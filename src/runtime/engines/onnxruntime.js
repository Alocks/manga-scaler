// ONNX Runtime adapter — model-agnostic engine. Configure the active model via onnxActiveModel (see onnx-models.js).

/** @type {Promise<any> | null} */
let onnxSessionPromise = null;
/** @type {any} */
let onnxSession = null;
let onnxInitialized = false;
let onnxSelectedProvider = null;
let onnxPreprocessCanvas = null;
let onnxPreprocessContext = null;
let onnxBackgroundWarmupPromise = null;
let onnxCanUseMainThreadTensorFromImage = null;
const ONNX_WORKER_LANE_FOREGROUND = 'foreground';
const ONNX_WORKER_LANE_BACKGROUND = 'background';
let onnxWorkingCanvasPool = {
    [ONNX_WORKER_LANE_FOREGROUND]: Object.create(null),
    [ONNX_WORKER_LANE_BACKGROUND]: Object.create(null)
};
const ONNX_WORKER_PENDING_WARNING_MS = 5000;
const ONNX_WORKER_MAX_INFLIGHT_RUN_REQUESTS = 1;
const ONNX_USE_MAIN_THREAD_TILED_PATH = false;
const ONNX_INIT_RETRY_BASE_DELAY_MS = 1000;
const ONNX_INIT_RETRY_MAX_DELAY_MS = 30000;

function createOnnxWorkerLaneState() {
    return {
        worker: null,
        readyPromise: null,
        requestId: 0,
        bootstrapUrl: null,
        pendingRequests: new Map(),
        activeRunRequests: 0,
        runSlotWaiters: [],
        initialized: false,
        selectedProvider: null,
        consecutiveInitFailures: 0,
        blockedUntilMs: 0
    };
}

const onnxWorkerLaneStates = {
    [ONNX_WORKER_LANE_FOREGROUND]: createOnnxWorkerLaneState(),
    [ONNX_WORKER_LANE_BACKGROUND]: createOnnxWorkerLaneState()
};

const ONNX_WORKER_PATH = 'src/workers/onnxruntime.worker.js';
const ONNX_CACHE_DB_NAME = 'manga-scaler-onnx-artifacts';
const ONNX_CACHE_STORE_NAME = 'artifacts';
const ONNX_TILE_YIELD_MS = 0;
const ONNX_TILE_YIELD_EVERY_TILES = 4;
const ONNX_BYTE_TO_UNIT_FLOAT = 0.00392156862745098;
const ONNX_MODEL_SWITCH_FALLBACK_KEY = 'realesrgan-2xplus';
let onnxCacheDbPromise = null;
let activeOnnxModelKey = null;

function getOnnxNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
}

function normalizeOnnxModelSelectionFromSettings(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    if (typeof normalizeOnnxModelKey === 'function') {
        return normalizeOnnxModelKey(settings.selectedOnnxModel);
    }
    return String(settings.selectedOnnxModel || ONNX_MODEL_SWITCH_FALLBACK_KEY);
}

function applyOnnxModelSelection(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const selectedModelKey = normalizeOnnxModelSelectionFromSettings(runtimeSettings);

    if (activeOnnxModelKey === selectedModelKey) {
        return;
    }

    const previousModelKey = activeOnnxModelKey;
    activeOnnxModelKey = selectedModelKey;

    if (typeof setOnnxActiveModelByKey === 'function') {
        setOnnxActiveModelByKey(selectedModelKey);
    }

    if (previousModelKey !== null) {
        resetOnnxAdapterState();
    }

    runtimeLog('onnx:model-selected', {
        previousModelKey,
        selectedModelKey,
        modelLabel: onnxActiveModel.label,
        modelPath: onnxActiveModel.modelPath
    });
}

function normalizeOnnxExecutionLane(lane) {
    if (lane === ONNX_WORKER_LANE_FOREGROUND) {
        return ONNX_WORKER_LANE_FOREGROUND;
    }

    if (typeof lane === 'string' && lane.startsWith(ONNX_WORKER_LANE_BACKGROUND)) {
        return ONNX_WORKER_LANE_BACKGROUND;
    }

    return ONNX_WORKER_LANE_FOREGROUND;
}

function getOnnxWorkerLaneState(lane = ONNX_WORKER_LANE_FOREGROUND) {
    const normalizedLane = normalizeOnnxExecutionLane(lane);
    if (!onnxWorkerLaneStates[normalizedLane]) {
        onnxWorkerLaneStates[normalizedLane] = createOnnxWorkerLaneState();
    }
    return onnxWorkerLaneStates[normalizedLane];
}

function serializeOnnxError(error) {
    if (!error) {
        return { message: 'Unknown ONNX error' };
    }

    const asAny = error;
    return {
        message: String(asAny.message || asAny),
        name: asAny.name ? String(asAny.name) : null,
        stack: typeof asAny.stack === 'string' ? asAny.stack.split('\n').slice(0, 4).join('\n') : null
    };
}

function isOnnxProtobufParseFailure(error) {
    const message = String(error?.message || error || '').toLowerCase();
    if (!message) return false;

    return (
        message.includes('protobuf parsing failed') ||
        message.includes('failed to load model because protobuf parsing failed') ||
        message.includes('error_code: 7')
    );
}

function formatOnnxDurationMs(startAt, endAt = getOnnxNow()) {
    return Number((endAt - startAt).toFixed(2));
}

function getOnnxWasmPathDiagnostics(lib) {
    const wasmConfig = lib?.env?.wasm;
    const wasmPaths = wasmConfig?.wasmPaths;

    if (!wasmPaths) {
        return { configured: false, kind: 'none', keys: [] };
    }

    if (typeof wasmPaths === 'string') {
        return { configured: true, kind: 'string', keys: [wasmPaths] };
    }

    if (typeof wasmPaths === 'object') {
        return { configured: true, kind: 'object', keys: Object.keys(wasmPaths) };
    }

    return { configured: true, kind: typeof wasmPaths, keys: [] };
}

function getOnnxLibrary() {
    const lib = window.ort;
    return lib && typeof lib === 'object' ? lib : null;
}

function isOnnxLibrarySupported(lib) {
    return (
        !!lib &&
        !!lib.InferenceSession &&
        typeof lib.InferenceSession.create === 'function' &&
        typeof lib.Tensor === 'function'
    );
}

function getOnnxModelUrl() {
    if (!chrome?.runtime?.getURL) {
        throw new Error('chrome.runtime.getURL is unavailable for ONNX model loading');
    }
    return chrome.runtime.getURL(onnxActiveModel.modelPath);
}

function getOnnxExternalDataUrl() {
    if (!onnxActiveModel.externalDataPath) {
        return null;
    }
    if (!chrome?.runtime?.getURL) {
        throw new Error('chrome.runtime.getURL is unavailable for ONNX external data loading');
    }
    return chrome.runtime.getURL(onnxActiveModel.externalDataPath);
}

function getOnnxWorkerUrl() {
    if (!chrome?.runtime?.getURL) {
        throw new Error('chrome.runtime.getURL is unavailable for ONNX worker loading');
    }
    return chrome.runtime.getURL(ONNX_WORKER_PATH);
}

function getOnnxOrtScriptUrl() {
    if (!chrome?.runtime?.getURL) {
        throw new Error('chrome.runtime.getURL is unavailable for ONNX Runtime script loading');
    }
    return chrome.runtime.getURL('node_modules/onnxruntime-web/dist/ort.all.min.js');
}

function getOnnxOrtDistUrl() {
    if (!chrome?.runtime?.getURL) {
        throw new Error('chrome.runtime.getURL is unavailable for ONNX Runtime asset loading');
    }
    return chrome.runtime.getURL('node_modules/onnxruntime-web/dist/');
}

function getOnnxOrtWasmModuleUrl() {
    return `${getOnnxOrtDistUrl()}ort-wasm-simd-threaded.jsep.mjs`;
}

function getOnnxOrtWasmBinaryUrl() {
    return `${getOnnxOrtDistUrl()}ort-wasm-simd-threaded.jsep.wasm`;
}

function cloneTypedArrayForTransfer(typedArray) {
    if (!typedArray || typeof typedArray.byteLength !== 'number') {
        throw new Error('Expected a typed array for transfer cloning');
    }

    // Always clone before transfer so lane initialization cannot detach shared artifact backing buffers.
    return typedArray.slice().buffer;
}

function rejectAllOnnxWorkerRequests(laneState, error) {
    if (!laneState || laneState.pendingRequests.size === 0) return;

    const pending = Array.from(laneState.pendingRequests.values());
    laneState.pendingRequests.clear();
    for (const entry of pending) {
        entry.reject(error);
    }
}

function drainOnnxWorkerRunSlotWaiters(laneState) {
    if (!laneState || !Array.isArray(laneState.runSlotWaiters)) return;

    while (laneState.runSlotWaiters.length > 0) {
        const resolve = laneState.runSlotWaiters.shift();
        if (typeof resolve === 'function') {
            resolve();
        }
    }
}

function releaseOnnxWorkerRunSlot(laneState) {
    if (!laneState) return;

    if (laneState.activeRunRequests > 0) {
        laneState.activeRunRequests -= 1;
    }

    if (
        laneState.activeRunRequests < ONNX_WORKER_MAX_INFLIGHT_RUN_REQUESTS &&
        Array.isArray(laneState.runSlotWaiters) &&
        laneState.runSlotWaiters.length > 0
    ) {
        const resolve = laneState.runSlotWaiters.shift();
        if (typeof resolve === 'function') {
            resolve();
        }
    }
}

function waitForOnnxWorkerRunSlot(laneState) {
    if (!laneState) {
        return Promise.resolve();
    }

    if (laneState.activeRunRequests < ONNX_WORKER_MAX_INFLIGHT_RUN_REQUESTS) {
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        laneState.runSlotWaiters.push(resolve);
    });
}

function createOnnxWorkerMessageHandler(laneState, lane) {
    return function handleOnnxWorkerMessage(event) {
    const message = event?.data;
    if (!message || typeof message !== 'object') {
        return;
    }

    if (message.type === 'log') {
        runtimeLog(message.event || 'onnx:worker-log', {
            lane,
            ...(message.payload || {})
        });
        return;
    }

    if (message.type !== 'response') {
        return;
    }

    const pending = laneState.pendingRequests.get(message.id);
    if (!pending) {
        return;
    }

    laneState.pendingRequests.delete(message.id);
    if (pending.warningTimeoutId) {
        window.clearTimeout(pending.warningTimeoutId);
    }

    if (pending.type === 'run') {
        releaseOnnxWorkerRunSlot(laneState);
    }

    if (message.ok) {
        pending.resolve(message.payload);
        return;
    }

    const errorPayload = message.error || {};
    const workerError = new Error(String(errorPayload.message || 'Unknown ONNX worker error'));
    workerError.name = errorPayload.name || 'OnnxWorkerError';
    if (errorPayload.stack) {
        workerError.stack = errorPayload.stack;
    }
    pending.reject(workerError);
}
}

function resetOnnxWorkerState(lane = ONNX_WORKER_LANE_FOREGROUND, terminate = true) {
    const normalizedLane = normalizeOnnxExecutionLane(lane);
    const laneState = getOnnxWorkerLaneState(normalizedLane);

    if (terminate && laneState.worker) {
        laneState.worker.terminate();
    }

    if (laneState.bootstrapUrl) {
        URL.revokeObjectURL(laneState.bootstrapUrl);
        laneState.bootstrapUrl = null;
    }

    laneState.worker = null;
    laneState.readyPromise = null;
    laneState.initialized = false;
    laneState.selectedProvider = null;
    laneState.activeRunRequests = 0;
    drainOnnxWorkerRunSlotWaiters(laneState);
    rejectAllOnnxWorkerRequests(laneState, new Error('ONNX worker was reset'));

    if (normalizedLane === ONNX_WORKER_LANE_FOREGROUND) {
        onnxInitialized = false;
        onnxSelectedProvider = null;
    }
}

function createOnnxWorkerBootstrapUrl() {
    if (!URL || typeof URL.createObjectURL !== 'function') {
        throw new Error('Blob worker bootstrap is unavailable in this runtime');
    }

    const workerScriptUrl = getOnnxWorkerUrl();
    const bootstrapSource = `importScripts(${JSON.stringify(workerScriptUrl)});`;
    const bootstrapBlob = new Blob([bootstrapSource], { type: 'application/javascript' });
    return URL.createObjectURL(bootstrapBlob);
}

function getOrCreateOnnxWorker(lane = ONNX_WORKER_LANE_FOREGROUND) {
    const normalizedLane = normalizeOnnxExecutionLane(lane);
    const laneState = getOnnxWorkerLaneState(normalizedLane);

    if (laneState.worker) {
        return laneState.worker;
    }

    if (typeof Worker !== 'function') {
        throw new Error('Dedicated workers are not supported in this runtime');
    }

    laneState.bootstrapUrl = createOnnxWorkerBootstrapUrl();
    const worker = new Worker(laneState.bootstrapUrl);
    runtimeLog('onnx:worker-bootstrap-created', {
        workerScriptUrl: getOnnxWorkerUrl(),
        lane: normalizedLane
    });
    worker.addEventListener('message', createOnnxWorkerMessageHandler(laneState, normalizedLane));
    worker.addEventListener('error', (event) => {
        const error = new Error(event?.message || 'Unknown ONNX worker error');
        runtimeLog('onnx:worker-error', {
            lane: normalizedLane,
            message: error.message,
            filename: event?.filename || null,
            lineno: event?.lineno || null,
            colno: event?.colno || null
        });
        resetOnnxWorkerState(normalizedLane, false);
        rejectAllOnnxWorkerRequests(laneState, error);
    });

    laneState.worker = worker;
    return worker;
}

async function postOnnxWorkerRequest(type, payload, transferList = [], lane = ONNX_WORKER_LANE_FOREGROUND) {
    const normalizedLane = normalizeOnnxExecutionLane(lane);
    const laneState = getOnnxWorkerLaneState(normalizedLane);
    const worker = getOrCreateOnnxWorker(normalizedLane);
    const requestId = ++laneState.requestId;
    const requestStartedAt = getOnnxNow();

    if (type === 'run') {
        if (laneState.activeRunRequests >= ONNX_WORKER_MAX_INFLIGHT_RUN_REQUESTS) {
            runtimeLog('onnx:worker-run-backpressure-wait', {
                lane: normalizedLane,
                requestId,
                activeRunRequests: laneState.activeRunRequests,
                pendingRequestCount: laneState.pendingRequests.size,
                maxInflightRunRequests: ONNX_WORKER_MAX_INFLIGHT_RUN_REQUESTS
            });
        }

        await waitForOnnxWorkerRunSlot(laneState);
        laneState.activeRunRequests += 1;
    }

    return new Promise((resolve, reject) => {
        const warningTimeoutId = window.setTimeout(() => {
            runtimeLog('onnx:worker-request-still-pending', {
                lane: normalizedLane,
                requestId,
                type,
                pendingRequestCount: laneState.pendingRequests.size,
                elapsedMs: formatOnnxDurationMs(requestStartedAt)
            });
        }, ONNX_WORKER_PENDING_WARNING_MS);

        laneState.pendingRequests.set(requestId, {
            resolve,
            reject,
            type,
            warningTimeoutId,
            requestStartedAt
        });

        try {
            worker.postMessage({ id: requestId, type, payload }, transferList);
        } catch (error) {
            laneState.pendingRequests.delete(requestId);
            window.clearTimeout(warningTimeoutId);
            if (type === 'run') {
                releaseOnnxWorkerRunSlot(laneState);
            }
            reject(error);
        }
    });
}

async function ensureOnnxWorkerReady(lane = ONNX_WORKER_LANE_FOREGROUND) {
    const normalizedLane = normalizeOnnxExecutionLane(lane);
    const laneState = getOnnxWorkerLaneState(normalizedLane);
    const now = getOnnxNow();

    if (laneState.blockedUntilMs > now) {
        throw new Error(`ONNX worker init cooldown active for lane ${normalizedLane}`);
    }

    if (laneState.initialized && laneState.worker) {
        return {
            provider: laneState.selectedProvider,
            inputNames: ['input'],
            outputNames: []
        };
    }

    if (laneState.readyPromise) {
        return laneState.readyPromise;
    }

    laneState.readyPromise = (async () => {
        const startedAt = getOnnxNow();
        async function dispatchWorkerInit(forceRefresh = false) {
            const artifacts = await loadOnnxArtifacts({ forceRefresh });
            const modelBuffer = cloneTypedArrayForTransfer(artifacts.modelBytes);
            const externalDataBuffer = artifacts.externalDataBytes
                ? cloneTypedArrayForTransfer(artifacts.externalDataBytes)
                : null;

            runtimeLog('onnx:worker-init-dispatch', {
                lane: normalizedLane,
                forceRefresh,
                modelBytes: artifacts.modelBytes.byteLength,
                externalDataBytes: artifacts.externalDataBytes ? artifacts.externalDataBytes.byteLength : 0,
                modelCacheHit: artifacts.modelCacheHit,
                externalDataCacheHit: artifacts.externalDataCacheHit
            });

            const transferList = [modelBuffer];
            if (externalDataBuffer) {
                transferList.push(externalDataBuffer);
            }

            return postOnnxWorkerRequest('init', {
                ortScriptUrl: getOnnxOrtScriptUrl(),
                ortDistUrl: getOnnxOrtDistUrl(),
                ortWasmModuleUrl: getOnnxOrtWasmModuleUrl(),
                ortWasmBinaryUrl: getOnnxOrtWasmBinaryUrl(),
                modelPath: onnxActiveModel.modelPath,
                modelUrl: artifacts.modelUrl,
                externalDataUrl: artifacts.externalDataUrl,
                externalDataPathAliases: onnxActiveModel.externalDataPathAliases,
                modelBytes: modelBuffer,
                externalDataBytes: externalDataBuffer
            }, transferList, normalizedLane);
        }

        try {
            let initResult;

            try {
                initResult = await dispatchWorkerInit(false);
            } catch (error) {
                if (!isOnnxProtobufParseFailure(error)) {
                    throw error;
                }

                runtimeLog('onnx:init-retry-after-cache-bypass', {
                    lane: normalizedLane,
                    modelPath: onnxActiveModel.modelPath,
                    reason: String(error?.message || error)
                });

                resetOnnxWorkerState(normalizedLane);
                initResult = await dispatchWorkerInit(true);
            }

            laneState.initialized = true;
            laneState.selectedProvider = initResult?.provider || 'wasm-worker';
            laneState.consecutiveInitFailures = 0;
            laneState.blockedUntilMs = 0;

            if (normalizedLane === ONNX_WORKER_LANE_FOREGROUND) {
                onnxInitialized = true;
                onnxSelectedProvider = laneState.selectedProvider;
            }

            runtimeLog('onnx:init', {
                lane: normalizedLane,
                model: onnxActiveModel.label,
                provider: laneState.selectedProvider,
                modelPath: onnxActiveModel.modelPath,
                durationMs: formatOnnxDurationMs(startedAt)
            });

            return initResult;
        } catch (error) {
            laneState.consecutiveInitFailures = (laneState.consecutiveInitFailures || 0) + 1;
            const delayMs = Math.min(
                ONNX_INIT_RETRY_MAX_DELAY_MS,
                ONNX_INIT_RETRY_BASE_DELAY_MS * Math.pow(2, Math.max(0, laneState.consecutiveInitFailures - 1))
            );
            laneState.blockedUntilMs = getOnnxNow() + delayMs;

            runtimeLog('onnx:init-failed', {
                lane: normalizedLane,
                model: onnxActiveModel.label,
                modelPath: onnxActiveModel.modelPath,
                durationMs: formatOnnxDurationMs(startedAt),
                consecutiveInitFailures: laneState.consecutiveInitFailures,
                retryBlockedForMs: delayMs,
                error: serializeOnnxError(error)
            });
            resetOnnxWorkerState(normalizedLane);
            throw error;
        } finally {
            laneState.readyPromise = null;
        }
    })();

    return laneState.readyPromise;
}

function openOnnxCacheDb() {
    if (!('indexedDB' in window)) {
        return Promise.resolve(null);
    }

    if (onnxCacheDbPromise) {
        return onnxCacheDbPromise;
    }

    onnxCacheDbPromise = new Promise((resolve) => {
        const request = indexedDB.open(ONNX_CACHE_DB_NAME, 1);

        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(ONNX_CACHE_STORE_NAME)) {
                db.createObjectStore(ONNX_CACHE_STORE_NAME, { keyPath: 'cacheKey' });
            }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => {
            runtimeLog('onnx:cache-db-open-error', { error: String(request.error) });
            resolve(null);
        };
        request.onblocked = () => {
            runtimeLog('onnx:cache-db-open-blocked');
            resolve(null);
        };
    });

    return onnxCacheDbPromise;
}

async function readOnnxCachedArtifact(cacheKey) {
    const db = await openOnnxCacheDb();
    if (!db) return null;

    return new Promise((resolve) => {
        const tx = db.transaction(ONNX_CACHE_STORE_NAME, 'readonly');
        const store = tx.objectStore(ONNX_CACHE_STORE_NAME);
        const request = store.get(cacheKey);

        request.onsuccess = () => {
            const payload = request.result?.payload;
            resolve(payload instanceof ArrayBuffer ? payload : null);
        };
        request.onerror = () => {
            runtimeLog('onnx:cache-read-error', { cacheKey, error: String(request.error) });
            resolve(null);
        };
    });
}

async function writeOnnxCachedArtifact(cacheKey, payload) {
    if (!(payload instanceof ArrayBuffer)) return false;

    const db = await openOnnxCacheDb();
    if (!db) return false;

    return new Promise((resolve) => {
        const tx = db.transaction(ONNX_CACHE_STORE_NAME, 'readwrite');
        const store = tx.objectStore(ONNX_CACHE_STORE_NAME);
        store.put({
            cacheKey,
            schema: onnxActiveModel.cacheSchemaVersion,
            payload,
            updatedAt: Date.now()
        });

        tx.oncomplete = () => resolve(true);
        tx.onerror = () => {
            runtimeLog('onnx:cache-write-error', { cacheKey, error: String(tx.error) });
            resolve(false);
        };
        tx.onabort = () => {
            runtimeLog('onnx:cache-write-abort', { cacheKey, error: String(tx.error) });
            resolve(false);
        };
    });
}

async function fetchOnnxArtifact(url, artifactKind, options = {}) {
    const forceRefresh = !!options?.forceRefresh;
    const cacheKey = `${onnxActiveModel.cacheSchemaVersion}|${artifactKind}|${url}`;
    if (!forceRefresh) {
        const cached = await readOnnxCachedArtifact(cacheKey);
        if (cached) {
            runtimeProfileLog('onnx:artifact-cache-hit', {
                artifactKind,
                url,
                bytes: cached.byteLength
            });
            return { bytes: new Uint8Array(cached), cacheHit: true };
        }
    } else {
        runtimeLog('onnx:artifact-cache-bypass', {
            artifactKind,
            url
        });
    }

    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch ONNX ${artifactKind}: ${response.status} ${response.statusText}`);
    }

    const buffer = await response.arrayBuffer();
    await writeOnnxCachedArtifact(cacheKey, buffer);

    runtimeProfileLog('onnx:artifact-cache-miss', {
        artifactKind,
        url,
        bytes: buffer.byteLength
    });

    return { bytes: new Uint8Array(buffer), cacheHit: false };
}

async function loadOnnxArtifacts(options = {}) {
    const forceRefresh = !!options?.forceRefresh;
    const modelUrl = getOnnxModelUrl();
    const externalDataUrl = getOnnxExternalDataUrl();

    const model = await fetchOnnxArtifact(modelUrl, 'model', { forceRefresh });
    let externalData = null;
    if (externalDataUrl) {
        externalData = await fetchOnnxArtifact(externalDataUrl, 'externalData', { forceRefresh });
    }

    return {
        modelUrl,
        externalDataUrl,
        modelBytes: model.bytes,
        externalDataBytes: externalData?.bytes || null,
        modelCacheHit: model.cacheHit,
        externalDataCacheHit: externalData?.cacheHit || false
    };
}

function configureOnnxEnvironment(lib) {
    if (!lib?.env || !chrome?.runtime?.getURL) return;

    const distPath = chrome.runtime.getURL('node_modules/onnxruntime-web/dist/');

    if (lib.env.wasm) {
        // Provide explicit module and wasm URLs so worker/blob contexts do not resolve relative .mjs files from about:blank.
        lib.env.wasm.wasmPaths = {
            mjs: `${distPath}ort-wasm-simd-threaded.jsep.mjs`,
            wasm: `${distPath}ort-wasm-simd-threaded.jsep.wasm`
        };
        lib.env.wasm.numThreads = 1;
        // Proxy worker bootstrap is unreliable in content scripts without a script URL context.
        lib.env.wasm.proxy = false;
    }

    runtimeLog('onnx:env-configured', {
        modelPath: onnxActiveModel.modelPath,
        distPath,
        wasm: getOnnxWasmPathDiagnostics(lib),
        hasNavigatorGpu: !!navigator?.gpu,
        executionProvider: 'webgpu'
    });
}

async function createOnnxSession() {
    const startedAt = getOnnxNow();
    const lib = getOnnxLibrary();
    if (!isOnnxLibrarySupported(lib)) {
        runtimeLog('onnx:library-unavailable', {
            hasOrt: !!window.ort,
            hasInferenceSession: !!window.ort?.InferenceSession,
            hasTensorCtor: typeof window.ort?.Tensor === 'function'
        });
        throw new Error('ONNX Runtime Web is not available on window.ort');
    }

    configureOnnxEnvironment(lib);

    const artifactsStartAt = getOnnxNow();
    let artifacts = await loadOnnxArtifacts();
    let artifactsEndAt = getOnnxNow();

    const sessionOptions = {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all'
    };

    if (artifacts.externalDataBytes && onnxActiveModel.externalDataPathAliases.length > 0) {
        sessionOptions.externalData = onnxActiveModel.externalDataPathAliases.map((path) => ({
            path,
            data: artifacts.externalDataBytes
        }));
    }

    runtimeLog('onnx:session-create-start', {
        providerCandidates: ['webgpu'],
        modelUrl: artifacts.modelUrl,
        externalDataUrl: artifacts.externalDataUrl,
        externalDataPathAliases: onnxActiveModel.externalDataPathAliases,
        modelBytes: artifacts.modelBytes.byteLength,
        externalDataBytes: artifacts.externalDataBytes ? artifacts.externalDataBytes.byteLength : 0,
        modelCacheHit: artifacts.modelCacheHit,
        externalDataCacheHit: artifacts.externalDataCacheHit,
        artifactLoadMs: formatOnnxDurationMs(artifactsStartAt, artifactsEndAt)
    });

    try {
        const session = await lib.InferenceSession.create(artifacts.modelBytes, sessionOptions);
        onnxSelectedProvider = 'webgpu-main';

        runtimeLog('onnx:session-create-success', {
            provider: onnxSelectedProvider,
            durationMs: formatOnnxDurationMs(startedAt),
            inputNames: Array.isArray(session?.inputNames) ? session.inputNames : [],
            outputNames: Array.isArray(session?.outputNames) ? session.outputNames : [],
            externalDataPathAliases: onnxActiveModel.externalDataPathAliases
        });

        return session;
    } catch (error) {
        if (isOnnxProtobufParseFailure(error) && !artifacts.forceRefreshed) {
            runtimeLog('onnx:session-retry-after-cache-bypass', {
                modelPath: onnxActiveModel.modelPath,
                reason: String(error?.message || error)
            });

            artifacts = await loadOnnxArtifacts({ forceRefresh: true });
            artifacts.forceRefreshed = true;
            artifactsEndAt = getOnnxNow();

            const retrySessionOptions = {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all'
            };

            if (artifacts.externalDataBytes && onnxActiveModel.externalDataPathAliases.length > 0) {
                retrySessionOptions.externalData = onnxActiveModel.externalDataPathAliases.map((path) => ({
                    path,
                    data: artifacts.externalDataBytes
                }));
            }

            const session = await lib.InferenceSession.create(artifacts.modelBytes, retrySessionOptions);
            onnxSelectedProvider = 'webgpu-main';

            runtimeLog('onnx:session-create-success', {
                provider: onnxSelectedProvider,
                durationMs: formatOnnxDurationMs(startedAt),
                inputNames: Array.isArray(session?.inputNames) ? session.inputNames : [],
                outputNames: Array.isArray(session?.outputNames) ? session.outputNames : [],
                externalDataPathAliases: onnxActiveModel.externalDataPathAliases,
                retriedAfterCacheBypass: true,
                artifactLoadMs: formatOnnxDurationMs(artifactsStartAt, artifactsEndAt)
            });

            return session;
        }

        runtimeLog('onnx:session-create-failed', {
            durationMs: formatOnnxDurationMs(startedAt),
            providerCandidates: ['webgpu'],
            modelUrl: artifacts.modelUrl,
            externalDataUrl: artifacts.externalDataUrl,
            externalDataPathAliases: onnxActiveModel.externalDataPathAliases,
            error: serializeOnnxError(error)
        });
        throw error;
    }
}

async function getOrCreateOnnxSession() {
    if (onnxSession) return onnxSession;
    if (onnxSessionPromise) return onnxSessionPromise;

    onnxSessionPromise = (async () => {
        const session = await createOnnxSession();
        onnxSession = session;
        onnxInitialized = true;

        runtimeLog('onnx:init', {
            model: onnxActiveModel.label,
            provider: onnxSelectedProvider,
            modelPath: onnxActiveModel.modelPath
        });

        return session;
    })();

    try {
        return await onnxSessionPromise;
    } catch (error) {
        runtimeLog('onnx:init-failed', {
            model: onnxActiveModel.label,
            modelPath: onnxActiveModel.modelPath,
            error: serializeOnnxError(error)
        });
        onnxSession = null;
        onnxInitialized = false;
        onnxSelectedProvider = null;
        onnxSessionPromise = null;
        throw error;
    } finally {
        onnxSessionPromise = null;
    }
}

function getOnnxPreprocessContext(width, height) {
    if (!onnxPreprocessCanvas) {
        if (typeof OffscreenCanvas === 'function') {
            onnxPreprocessCanvas = new OffscreenCanvas(width, height);
        } else {
            onnxPreprocessCanvas = document.createElement('canvas');
        }
    }

    if (onnxPreprocessCanvas.width !== width) {
        onnxPreprocessCanvas.width = width;
    }
    if (onnxPreprocessCanvas.height !== height) {
        onnxPreprocessCanvas.height = height;
    }

    if (!onnxPreprocessContext) {
        onnxPreprocessContext = onnxPreprocessCanvas.getContext('2d', {
            alpha: false,
            desynchronized: true,
            willReadFrequently: true
        });
    }

    if (!onnxPreprocessContext) {
        throw new Error('Failed to acquire 2D preprocessing context for ONNX Runtime');
    }

    return onnxPreprocessContext;
}

function createOnnxWorkingCanvas(width = 1, height = 1) {
    if (typeof OffscreenCanvas === 'function') {
        return new OffscreenCanvas(width, height);
    }

    const elementCanvas = document.createElement('canvas');
    elementCanvas.width = width;
    elementCanvas.height = height;
    return elementCanvas;
}

function getOnnxReusableWorkingCanvas(slot, width = 1, height = 1, lane = ONNX_WORKER_LANE_FOREGROUND) {
    const normalizedLane = normalizeOnnxExecutionLane(lane);

    if (!onnxWorkingCanvasPool[normalizedLane]) {
        onnxWorkingCanvasPool[normalizedLane] = Object.create(null);
    }

    const lanePool = onnxWorkingCanvasPool[normalizedLane];
    let canvas = lanePool[slot];

    if (!canvas) {
        canvas = createOnnxWorkingCanvas(width, height);
        lanePool[slot] = canvas;
    }

    if (canvas.width !== width) {
        canvas.width = width;
    }
    if (canvas.height !== height) {
        canvas.height = height;
    }

    return canvas;
}

function alignOnnxDimension(value, alignment = onnxActiveModel.inputAlignment) {
    if (!Number.isFinite(value) || value < 1) {
        return 1;
    }

    return Math.max(alignment, Math.ceil(value / alignment) * alignment);
}

function getOnnxInputPadding(width, height) {
    let paddedWidth = alignOnnxDimension(width);
    let paddedHeight = alignOnnxDimension(height);

    if (onnxActiveModel.fixedInputWidth > 0) {
        paddedWidth = Math.max(paddedWidth, onnxActiveModel.fixedInputWidth);
    }
    if (onnxActiveModel.fixedInputHeight > 0) {
        paddedHeight = Math.max(paddedHeight, onnxActiveModel.fixedInputHeight);
    }

    return {
        paddedWidth,
        paddedHeight,
        padRight: Math.max(0, paddedWidth - width),
        padBottom: Math.max(0, paddedHeight - height)
    };
}

function drawOnnxImageWithEdgePadding(sourceImage, destCanvas, width, height, paddedWidth, paddedHeight) {
    destCanvas.width = paddedWidth;
    destCanvas.height = paddedHeight;

    const destCtx = destCanvas.getContext('2d', { alpha: false, desynchronized: true });
    if (!destCtx) {
        throw new Error('Failed to acquire padded input canvas context for ONNX');
    }

    destCtx.clearRect(0, 0, paddedWidth, paddedHeight);
    destCtx.drawImage(sourceImage, 0, 0, width, height, 0, 0, width, height);

    if (paddedWidth > width) {
        destCtx.drawImage(sourceImage, width - 1, 0, 1, height, width, 0, paddedWidth - width, height);
    }

    if (paddedHeight > height) {
        destCtx.drawImage(sourceImage, 0, height - 1, width, 1, 0, height, width, paddedHeight - height);
    }

    if (paddedWidth > width && paddedHeight > height) {
        destCtx.drawImage(sourceImage, width - 1, height - 1, 1, 1, width, height, paddedWidth - width, paddedHeight - height);
    }

    return destCanvas;
}

function copyOnnxCanvasRegion(sourceCanvas, destCanvas, width, height) {
    destCanvas.width = width;
    destCanvas.height = height;

    const destCtx = destCanvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!destCtx) {
        throw new Error('Failed to acquire cropped output canvas context for ONNX');
    }

    destCtx.clearRect(0, 0, width, height);
    destCtx.drawImage(sourceCanvas, 0, 0, width, height, 0, 0, width, height);
}

function resolveOnnxTileEdgePixels() {
    const fromPixelBudget = Math.floor(Math.sqrt(onnxActiveModel.maxForegroundPixels));
    const overlapLimited = Math.max(onnxActiveModel.minTileEdgePixels, fromPixelBudget - onnxActiveModel.tileOverlapPixels * 2);
    return Math.min(onnxActiveModel.maxTileEdgePixels, overlapLimited);
}

function yieldOnnxTileLoop() {
    return new Promise((resolve) => {
        window.setTimeout(resolve, ONNX_TILE_YIELD_MS);
    });
}

function imageToOnnxWorkerInput(image) {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;

    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        throw new Error('Invalid source image dimensions for ONNX Runtime');
    }

    const ctx = getOnnxPreprocessContext(width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    return {
        width,
        height,
        rgbaBuffer: imageData.data.buffer
    };
}

function drawOnnxWorkerOutputToCanvas(workerResult, canvas) {
    const outWidth = workerResult?.outputWidth;
    const outHeight = workerResult?.outputHeight;
    const rgbaBuffer = workerResult?.rgbaBuffer;

    if (!Number.isFinite(outWidth) || !Number.isFinite(outHeight) || outWidth < 1 || outHeight < 1) {
        throw new Error('Worker returned invalid ONNX output dimensions');
    }

    if (!(rgbaBuffer instanceof ArrayBuffer)) {
        throw new Error('Worker returned invalid ONNX output pixels');
    }

    canvas.width = outWidth;
    canvas.height = outHeight;

    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) {
        throw new Error('Failed to acquire output canvas context for ONNX Runtime');
    }

    const imageData = new ImageData(new Uint8ClampedArray(rgbaBuffer), outWidth, outHeight);
    ctx.putImageData(imageData, 0, 0);
}

function resolveOnnxOutputTensor(runResult, outputNames = []) {
    if (!runResult || typeof runResult !== 'object') {
        throw new Error('ONNX Runtime returned an invalid inference result');
    }

    if (outputNames.length > 0 && runResult[outputNames[0]]) {
        return runResult[outputNames[0]];
    }

    const firstTensor = Object.values(runResult)[0];
    if (!firstTensor) {
        throw new Error('ONNX Runtime did not return any output tensors');
    }

    return firstTensor;
}

function inferOnnxOutputScale(data) {
    const sampleCount = Math.min(2048, data.length);
    let maxValue = 0;
    for (let i = 0; i < sampleCount; i++) {
        const value = data[i];
        if (value > maxValue) {
            maxValue = value;
        }
    }
    return maxValue <= 1.5 ? 255 : 1;
}

async function imageToOnnxMainThreadTensor(image) {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;

    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        throw new Error('Invalid source image dimensions for ONNX Runtime');
    }

    const ctx = getOnnxPreprocessContext(width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const lib = getOnnxLibrary();
    if (!isOnnxLibrarySupported(lib)) {
        throw new Error('ONNX Runtime Web is unavailable for main-thread inference');
    }

    if (
        onnxCanUseMainThreadTensorFromImage !== false &&
        typeof lib.Tensor.fromImage === 'function' &&
        typeof ImageData === 'function'
    ) {
        try {
            const tensor = await lib.Tensor.fromImage(imageData, {
                tensorLayout: 'NCHW',
                tensorFormat: 'RGB',
                dataType: 'float32'
            });
            onnxCanUseMainThreadTensorFromImage = true;
            return tensor;
        } catch (error) {
            onnxCanUseMainThreadTensorFromImage = false;
            runtimeLog('onnx:main-fromimage-fallback', {
                reason: String(error),
                disabledAfterFailure: true
            });
        }
    }

    const rgba = imageData.data;
    const pixelCount = width * height;
    const chw = new Float32Array(pixelCount * 3);
    const gBase = pixelCount;
    const bBase = pixelCount * 2;

    for (let i = 0; i < pixelCount; i++) {
        const srcOffset = i << 2;
        chw[i] = rgba[srcOffset] * ONNX_BYTE_TO_UNIT_FLOAT;
        chw[gBase + i] = rgba[srcOffset + 1] * ONNX_BYTE_TO_UNIT_FLOAT;
        chw[bBase + i] = rgba[srcOffset + 2] * ONNX_BYTE_TO_UNIT_FLOAT;
    }

    return new lib.Tensor('float32', chw, [1, 3, height, width]);
}

async function drawOnnxTensorToCanvas(outputTensor, canvas, cropWidth = null, cropHeight = null) {
    if (!outputTensor || !Array.isArray(outputTensor.dims) || outputTensor.dims.length !== 4) {
        throw new Error('Unexpected ONNX output tensor shape. Expected rank-4 [N,C,H,W]');
    }

    const [batch, channels, outHeight, outWidth] = outputTensor.dims;
    if (!Number.isFinite(batch) || batch < 1 || channels < 3 || outHeight < 1 || outWidth < 1) {
        throw new Error(`Invalid ONNX output tensor dims: ${JSON.stringify(outputTensor.dims)}`);
    }

    const finalWidth = Math.max(1, Math.min(outWidth, Math.round(cropWidth || outWidth)));
    const finalHeight = Math.max(1, Math.min(outHeight, Math.round(cropHeight || outHeight)));

    canvas.width = finalWidth;
    canvas.height = finalHeight;

    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) {
        throw new Error('Failed to acquire output canvas context for ONNX Runtime');
    }

    if (typeof outputTensor.toImageData === 'function') {
        try {
            const imageData = outputTensor.toImageData({
                tensorLayout: 'NCHW',
                format: 'RGB'
            });
            if (imageData && imageData.data && imageData.width === outWidth && imageData.height === outHeight) {
                ctx.putImageData(imageData, 0, 0, 0, 0, finalWidth, finalHeight);
                return;
            }
        } catch (error) {
            runtimeLog('onnx:main-toimagedata-fallback', {
                reason: String(error)
            });
        }
    }

    let data;
    try {
        data = outputTensor.data;
    } catch (error) {
        if (typeof outputTensor.getData === 'function') {
            data = await outputTensor.getData();
        } else {
            throw error;
        }
    }

    if (!data || typeof data.length !== 'number') {
        throw new Error('ONNX output tensor has no readable data buffer');
    }

    const pixelCount = outWidth * outHeight;
    const channelStride = pixelCount;
    const batchStride = channels * channelStride;
    const base = 0 * batchStride;
    const rStride = base;
    const gStride = base + channelStride;
    const bStride = base + channelStride * 2;
    const valueScale = inferOnnxOutputScale(data);
    const rgba = new Uint8ClampedArray(finalWidth * finalHeight * 4);

    let dst = 0;
    for (let y = 0; y < finalHeight; y++) {
        const srcRowOffset = y * outWidth;
        for (let x = 0; x < finalWidth; x++) {
            const src = srcRowOffset + x;
            rgba[dst] = Math.max(0, Math.min(255, Math.round(data[rStride + src] * valueScale)));
            rgba[dst + 1] = Math.max(0, Math.min(255, Math.round(data[gStride + src] * valueScale)));
            rgba[dst + 2] = Math.max(0, Math.min(255, Math.round(data[bStride + src] * valueScale)));
            rgba[dst + 3] = 255;
            dst += 4;
        }
    }

    ctx.putImageData(new ImageData(rgba, finalWidth, finalHeight), 0, 0);
}

async function runOnnxInferencePassMainThread(inputImage, outputCanvas, executionOptions = {}) {
    const executionLane = normalizeOnnxExecutionLane(executionOptions?.lane);
    const passStartAt = getOnnxNow();
    const inputWidth = inputImage.naturalWidth || inputImage.width || 0;
    const inputHeight = inputImage.naturalHeight || inputImage.height || 0;
    const inputPadding = getOnnxInputPadding(inputWidth, inputHeight);
    const needsInputPadding = inputPadding.padRight > 0 || inputPadding.padBottom > 0;

    runtimeLog('onnx:pass-main-start', {
        lane: executionLane,
        inputWidth,
        inputHeight,
        paddedWidth: inputPadding.paddedWidth,
        paddedHeight: inputPadding.paddedHeight,
        needsInputPadding
    });

    let inferenceSourceImage = inputImage;
    if (needsInputPadding) {
        inferenceSourceImage = drawOnnxImageWithEdgePadding(
            inputImage,
            getOnnxReusableWorkingCanvas(
                'padded-input-main',
                inputPadding.paddedWidth,
                inputPadding.paddedHeight,
                executionLane
            ),
            inputWidth,
            inputHeight,
            inputPadding.paddedWidth,
            inputPadding.paddedHeight
        );
    }

    const inferenceInputWidth = inferenceSourceImage.naturalWidth || inferenceSourceImage.width || inputPadding.paddedWidth;
    const inferenceInputHeight = inferenceSourceImage.naturalHeight || inferenceSourceImage.height || inputPadding.paddedHeight;

    const tensorBuildStartAt = getOnnxNow();
    const inputTensor = await imageToOnnxMainThreadTensor(inferenceSourceImage);
    const tensorBuildEndAt = getOnnxNow();

    const session = await getOrCreateOnnxSession();
    const inputName = session.inputNames?.[0];
    if (!inputName) {
        throw new Error('ONNX model has no input name');
    }

    const inferenceStartAt = getOnnxNow();
    const outputs = await session.run({ [inputName]: inputTensor });
    const inferenceEndAt = getOnnxNow();
    const outputTensor = resolveOnnxOutputTensor(outputs, session.outputNames || []);

    const outputWidth = outputTensor?.dims?.[3] || 0;
    const outputHeight = outputTensor?.dims?.[2] || 0;
    const scaleX = outputWidth > 0 ? outputWidth / inferenceInputWidth : 1;
    const scaleY = outputHeight > 0 ? outputHeight / inferenceInputHeight : 1;

    const canvasWriteStartAt = getOnnxNow();
    if (needsInputPadding) {
        const croppedOutputWidth = Math.max(1, Math.round(inputWidth * scaleX));
        const croppedOutputHeight = Math.max(1, Math.round(inputHeight * scaleY));
        await drawOnnxTensorToCanvas(outputTensor, outputCanvas, croppedOutputWidth, croppedOutputHeight);
    } else {
        await drawOnnxTensorToCanvas(outputTensor, outputCanvas);
    }
    const canvasWriteEndAt = getOnnxNow();

    runtimeLog('onnx:pass-main-success', {
        lane: executionLane,
        inputName,
        inputDims: inputTensor.dims,
        outputDims: outputTensor?.dims || null,
        tensorBuildMs: formatOnnxDurationMs(tensorBuildStartAt, tensorBuildEndAt),
        inferenceMs: formatOnnxDurationMs(inferenceStartAt, inferenceEndAt),
        canvasWriteMs: formatOnnxDurationMs(canvasWriteStartAt, canvasWriteEndAt),
        totalPassMs: formatOnnxDurationMs(passStartAt)
    });

    return {
        inputName,
        inputDims: inputTensor.dims,
        outputDims: outputTensor?.dims || null,
        inferenceMs: formatOnnxDurationMs(inferenceStartAt, inferenceEndAt),
        outputWidth: outputCanvas.width,
        outputHeight: outputCanvas.height
    };
}

function drawOnnxWorkerOutputCropToCanvas(workerResult, canvas, cropWidth, cropHeight) {
    const outWidth = workerResult?.outputWidth;
    const outHeight = workerResult?.outputHeight;
    const rgbaBuffer = workerResult?.rgbaBuffer;

    if (!Number.isFinite(outWidth) || !Number.isFinite(outHeight) || outWidth < 1 || outHeight < 1) {
        throw new Error('Worker returned invalid ONNX output dimensions');
    }

    if (!(rgbaBuffer instanceof ArrayBuffer)) {
        throw new Error('Worker returned invalid ONNX output pixels');
    }

    const finalCropWidth = Math.max(1, Math.min(outWidth, Math.round(cropWidth || outWidth)));
    const finalCropHeight = Math.max(1, Math.min(outHeight, Math.round(cropHeight || outHeight)));

    canvas.width = finalCropWidth;
    canvas.height = finalCropHeight;

    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) {
        throw new Error('Failed to acquire cropped output canvas context for ONNX Runtime');
    }

    const imageData = new ImageData(new Uint8ClampedArray(rgbaBuffer), outWidth, outHeight);
    ctx.putImageData(imageData, 0, 0, 0, 0, finalCropWidth, finalCropHeight);
}

async function runOnnxInferencePass(inputImage, outputCanvas, executionOptions = {}) {
    const executionLane = normalizeOnnxExecutionLane(executionOptions?.lane);
    const laneState = getOnnxWorkerLaneState(executionLane);
    const passStartAt = getOnnxNow();
    const inputWidth = inputImage.naturalWidth || inputImage.width || 0;
    const inputHeight = inputImage.naturalHeight || inputImage.height || 0;
    const inputPadding = getOnnxInputPadding(inputWidth, inputHeight);
    const needsInputPadding = inputPadding.padRight > 0 || inputPadding.padBottom > 0;
    runtimeLog('onnx:pass-start', {
        inputWidth,
        inputHeight,
        paddedWidth: inputPadding.paddedWidth,
        paddedHeight: inputPadding.paddedHeight,
        needsInputPadding
    });

    const tensorBuildStartAt = getOnnxNow();
    let workerSourceImage = inputImage;
    if (needsInputPadding) {
        workerSourceImage = drawOnnxImageWithEdgePadding(
            inputImage,
            getOnnxReusableWorkingCanvas(
                'padded-input',
                inputPadding.paddedWidth,
                inputPadding.paddedHeight,
                executionLane
            ),
            inputWidth,
            inputHeight,
            inputPadding.paddedWidth,
            inputPadding.paddedHeight
        );

        runtimeLog('onnx:input-padded', {
            inputWidth,
            inputHeight,
            paddedWidth: inputPadding.paddedWidth,
            paddedHeight: inputPadding.paddedHeight,
            padRight: inputPadding.padRight,
            padBottom: inputPadding.padBottom
        });
    }

    const workerInput = imageToOnnxWorkerInput(workerSourceImage);
    const tensorBuildEndAt = getOnnxNow();
    await ensureOnnxWorkerReady(executionLane);

    const inferenceStartAt = getOnnxNow();
    runtimeLog('onnx:pass-inference-start', {
        lane: executionLane,
        inputName: 'input',
        inputDims: [1, 3, workerInput.height, workerInput.width],
        tensorBuildMs: formatOnnxDurationMs(tensorBuildStartAt, tensorBuildEndAt)
    });
    runtimeLog('onnx:worker-run-dispatch', {
        lane: executionLane,
        inputWidth: workerInput.width,
        inputHeight: workerInput.height,
        inputPixels: workerInput.width * workerInput.height
    });
    const workerResult = await postOnnxWorkerRequest('run', workerInput, [workerInput.rgbaBuffer], executionLane);
    const inferenceEndAt = getOnnxNow();

    runtimeLog('onnx:worker-run-response', {
        lane: executionLane,
        outputDims: workerResult?.outputDims || null,
        outputWidth: workerResult?.outputWidth || 0,
        outputHeight: workerResult?.outputHeight || 0,
        provider: workerResult?.provider || laneState.selectedProvider || onnxSelectedProvider || null
    });

    const canvasWriteStartAt = getOnnxNow();
    if (needsInputPadding) {
        const scaleX = workerResult.outputWidth / workerInput.width;
        const scaleY = workerResult.outputHeight / workerInput.height;
        const croppedOutputWidth = Math.max(1, Math.round(inputWidth * scaleX));
        const croppedOutputHeight = Math.max(1, Math.round(inputHeight * scaleY));

        drawOnnxWorkerOutputCropToCanvas(workerResult, outputCanvas, croppedOutputWidth, croppedOutputHeight);

        runtimeLog('onnx:output-cropped', {
            lane: executionLane,
            paddedOutputWidth: workerResult.outputWidth,
            paddedOutputHeight: workerResult.outputHeight,
            croppedOutputWidth,
            croppedOutputHeight,
            scaleX,
            scaleY
        });
    } else {
        drawOnnxWorkerOutputToCanvas(workerResult, outputCanvas);
    }
    const canvasWriteEndAt = getOnnxNow();

    runtimeLog('onnx:pass-success', {
        lane: executionLane,
        inputName: workerResult.inputName || 'input',
        inputDims: workerResult.inputDims || [1, 3, workerInput.height, workerInput.width],
        outputDims: workerResult.outputDims || null,
        paddedInputWidth: workerInput.width,
        paddedInputHeight: workerInput.height,
        tensorBuildMs: formatOnnxDurationMs(tensorBuildStartAt, tensorBuildEndAt),
        inferenceMs: formatOnnxDurationMs(inferenceStartAt, inferenceEndAt),
        canvasWriteMs: formatOnnxDurationMs(canvasWriteStartAt, canvasWriteEndAt),
        totalPassMs: formatOnnxDurationMs(passStartAt)
    });

    return {
        inputName: workerResult.inputName || 'input',
        inputDims: workerResult.inputDims || [1, 3, workerInput.height, workerInput.width],
        outputDims: workerResult.outputDims || null,
        inferenceMs: formatOnnxDurationMs(inferenceStartAt, inferenceEndAt),
        outputWidth: outputCanvas.width,
        outputHeight: outputCanvas.height
    };
}

async function runOnnxUpscaleTiled(sourceImage, outputCanvas, sourceWidth, sourceHeight, executionOptions = {}) {
    const executionLane = normalizeOnnxExecutionLane(executionOptions?.lane);
    const preferMainThreadPath = ONNX_USE_MAIN_THREAD_TILED_PATH && executionLane === ONNX_WORKER_LANE_FOREGROUND;
    let fellBackToWorkerPath = false;
    const tileEdge = resolveOnnxTileEdgePixels();
    const overlap = onnxActiveModel.tileOverlapPixels;
    const coreStep = Math.max(64, tileEdge - overlap * 2);
    const tileColumns = Math.ceil(sourceWidth / coreStep);
    const tileRows = Math.ceil(sourceHeight / coreStep);

    const tileInputCanvas = createOnnxWorkingCanvas(1, 1);
    const tileOutputCanvas = createOnnxWorkingCanvas(1, 1);
    const destCtx = outputCanvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!destCtx) {
        throw new Error('Failed to acquire output canvas context for ONNX tiled compose');
    }

    let scaleX = null;
    let scaleY = null;
    let tileCount = 0;
    let totalInferenceMs = 0;

    runtimeLog('onnx:tiled-layout', {
        lane: executionLane,
        preferredMode: preferMainThreadPath ? 'main-thread-webgpu' : 'worker',
        sourceWidth,
        sourceHeight,
        tileEdge,
        overlap,
        coreStep,
        tileColumns,
        tileRows,
        estimatedTileCount: tileColumns * tileRows
    });

    for (let coreY = 0; coreY < sourceHeight; coreY += coreStep) {
        const coreBottom = Math.min(sourceHeight, coreY + coreStep);

        for (let coreX = 0; coreX < sourceWidth; coreX += coreStep) {
            const coreRight = Math.min(sourceWidth, coreX + coreStep);

            const tileInputX = Math.max(0, coreX - overlap);
            const tileInputY = Math.max(0, coreY - overlap);
            const tileInputRight = Math.min(sourceWidth, coreRight + overlap);
            const tileInputBottom = Math.min(sourceHeight, coreBottom + overlap);

            const tileInputWidth = tileInputRight - tileInputX;
            const tileInputHeight = tileInputBottom - tileInputY;
            const nextTileIndex = tileCount + 1;

            runtimeLog('onnx:tile-start', {
                tileIndex: nextTileIndex,
                coreX,
                coreY,
                coreRight,
                coreBottom,
                tileInputX,
                tileInputY,
                tileInputRight,
                tileInputBottom,
                tileInputWidth,
                tileInputHeight
            });

            tileInputCanvas.width = tileInputWidth;
            tileInputCanvas.height = tileInputHeight;

            const tileInputCtx = tileInputCanvas.getContext('2d', { alpha: false, desynchronized: true });
            if (!tileInputCtx) {
                throw new Error('Failed to acquire tile input canvas context for ONNX');
            }

            const tilePrepareStartAt = getOnnxNow();
            tileInputCtx.clearRect(0, 0, tileInputWidth, tileInputHeight);
            tileInputCtx.drawImage(
                sourceImage,
                tileInputX,
                tileInputY,
                tileInputWidth,
                tileInputHeight,
                0,
                0,
                tileInputWidth,
                tileInputHeight
            );
            const tilePrepareEndAt = getOnnxNow();

            runtimeLog('onnx:tile-prepared', {
                tileIndex: nextTileIndex,
                prepareMs: formatOnnxDurationMs(tilePrepareStartAt, tilePrepareEndAt),
                tileInputWidth,
                tileInputHeight
            });

            let pass;
            if (preferMainThreadPath && !fellBackToWorkerPath) {
                try {
                    pass = await runOnnxInferencePassMainThread(tileInputCanvas, tileOutputCanvas, executionOptions);
                } catch (error) {
                    fellBackToWorkerPath = true;
                    runtimeLog('onnx:tiled-mainthread-fallback', {
                        lane: executionLane,
                        tileIndex: nextTileIndex,
                        error: serializeOnnxError(error)
                    });
                    pass = await runOnnxInferencePass(tileInputCanvas, tileOutputCanvas, executionOptions);
                }
            } else {
                pass = await runOnnxInferencePass(tileInputCanvas, tileOutputCanvas, executionOptions);
            }
            totalInferenceMs += pass.inferenceMs;
            tileCount++;

            runtimeLog('onnx:tile-pass-success', {
                tileIndex: tileCount,
                inputDims: pass.inputDims,
                outputDims: pass.outputDims,
                outputWidth: pass.outputWidth,
                outputHeight: pass.outputHeight,
                inferenceMs: pass.inferenceMs
            });

            if (scaleX === null || scaleY === null) {
                scaleX = pass.outputWidth / tileInputWidth;
                scaleY = pass.outputHeight / tileInputHeight;

                outputCanvas.width = Math.max(1, Math.round(sourceWidth * scaleX));
                outputCanvas.height = Math.max(1, Math.round(sourceHeight * scaleY));
                destCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

                runtimeLog('onnx:tiled-output-init', {
                    tileIndex: tileCount,
                    scaleX,
                    scaleY,
                    outputWidth: outputCanvas.width,
                    outputHeight: outputCanvas.height
                });
            }

            const cropLeftIn = coreX - tileInputX;
            const cropTopIn = coreY - tileInputY;
            const cropRightIn = tileInputRight - coreRight;
            const cropBottomIn = tileInputBottom - coreBottom;

            const cropLeftOut = Math.max(0, Math.round(cropLeftIn * scaleX));
            const cropTopOut = Math.max(0, Math.round(cropTopIn * scaleY));
            const cropRightOut = Math.max(0, Math.round(cropRightIn * scaleX));
            const cropBottomOut = Math.max(0, Math.round(cropBottomIn * scaleY));

            const drawWidth = Math.max(1, pass.outputWidth - cropLeftOut - cropRightOut);
            const drawHeight = Math.max(1, pass.outputHeight - cropTopOut - cropBottomOut);
            const destX = Math.max(0, Math.round(coreX * scaleX));
            const destY = Math.max(0, Math.round(coreY * scaleY));

            runtimeLog('onnx:tile-compose-start', {
                tileIndex: tileCount,
                cropLeftOut,
                cropTopOut,
                cropRightOut,
                cropBottomOut,
                drawWidth,
                drawHeight,
                destX,
                destY
            });

            destCtx.drawImage(
                tileOutputCanvas,
                cropLeftOut,
                cropTopOut,
                drawWidth,
                drawHeight,
                destX,
                destY,
                drawWidth,
                drawHeight
            );

            runtimeLog('onnx:tile-compose-success', {
                tileIndex: tileCount,
                drawWidth,
                drawHeight,
                destX,
                destY
            });

            if (ONNX_TILE_YIELD_EVERY_TILES > 0 && tileCount % ONNX_TILE_YIELD_EVERY_TILES === 0) {
                await yieldOnnxTileLoop();

                runtimeLog('onnx:tile-yield-complete', {
                    tileIndex: tileCount,
                    yieldEveryTiles: ONNX_TILE_YIELD_EVERY_TILES
                });
            }
        }
    }

    runtimeLog('onnx:tiled-run-success', {
        lane: executionLane,
        mode: preferMainThreadPath && !fellBackToWorkerPath ? 'main-thread-webgpu' : 'worker',
        sourceWidth,
        sourceHeight,
        tileEdge,
        overlap,
        coreStep,
        tileCount,
        outputWidth: outputCanvas.width,
        outputHeight: outputCanvas.height,
        scaleX,
        scaleY,
        totalInferenceMs: Number(totalInferenceMs.toFixed(2))
    });

    return {
        inputName: 'input',
        inputDims: [1, 3, sourceHeight, sourceWidth],
        outputDims: [1, 3, outputCanvas.height, outputCanvas.width],
        inferenceMs: Number(totalInferenceMs.toFixed(2)),
        tiled: true,
        tileCount
    };
}

async function runOnnxUpscale(tempImg, canvas, runtimeSettings = getRuntimePreferenceSnapshot(), executionOptions = {}) {
    applyOnnxModelSelection(runtimeSettings);
    const executionLane = normalizeOnnxExecutionLane(executionOptions?.lane);
    const laneState = getOnnxWorkerLaneState(executionLane);
    await backendReadyPromise;

    const totalStartAt = getOnnxNow();
    const sourceWidth = tempImg.naturalWidth || tempImg.width;
    const sourceHeight = tempImg.naturalHeight || tempImg.height;
    const pixelCount = sourceWidth * sourceHeight;

    runtimeProfileLog('onnx:run-start', {
        sourceWidth,
        sourceHeight,
        provider: laneState.selectedProvider || onnxSelectedProvider,
        lane: executionLane
    });

    try {
        await ensureOnnxWorkerReady(executionLane);
        let runMeta;

        const useTiledPath =
            pixelCount > onnxActiveModel.maxSinglePassPixels ||
            (onnxActiveModel.fixedInputWidth > 0 && sourceWidth > onnxActiveModel.fixedInputWidth) ||
            (onnxActiveModel.fixedInputHeight > 0 && sourceHeight > onnxActiveModel.fixedInputHeight);

        if (useTiledPath) {
            runtimeLog('onnx:run-tiled-start', {
                sourceWidth,
                sourceHeight,
                pixelCount,
                threshold: onnxActiveModel.maxSinglePassPixels
            });
            runMeta = await runOnnxUpscaleTiled(tempImg, canvas, sourceWidth, sourceHeight, executionOptions);
        } else {
            runMeta = await runOnnxInferencePass(tempImg, canvas, executionOptions);
        }

        runtimeLog('onnx:run-success', {
            provider: laneState.selectedProvider || onnxSelectedProvider,
            lane: executionLane,
            inputName: runMeta.inputName,
            inputDims: runMeta.inputDims,
            outputDims: runMeta.outputDims,
            inferenceMs: runMeta.inferenceMs,
            tiled: !!runMeta.tiled,
            tileCount: runMeta.tileCount || 0,
            totalMs: formatOnnxDurationMs(totalStartAt),
            canvasWidth: canvas.width,
            canvasHeight: canvas.height
        });

        return {
            model: onnxActiveModel.label,
            runMode: laneState.selectedProvider || onnxSelectedProvider || 'onnx'
        };
    } catch (error) {
        runtimeLog('onnx:run-failed', {
            provider: laneState.selectedProvider || onnxSelectedProvider,
            lane: executionLane,
            sourceWidth,
            sourceHeight,
            totalMs: formatOnnxDurationMs(totalStartAt),
            error: serializeOnnxError(error)
        });
        throw error;
    }
}

async function prewarmOnnx(runtimeSettings = getRuntimePreferenceSnapshot()) {
    applyOnnxModelSelection(runtimeSettings);
    const startedAt = getOnnxNow();
    runtimeLog('onnx:prewarm-start', {
        model: onnxActiveModel.label,
        modelPath: onnxActiveModel.modelPath,
        mode: 'worker-init-non-blocking'
    });

    // Startup prewarm dispatches worker initialization without blocking the page thread.
    if (!onnxBackgroundWarmupPromise) {
        onnxBackgroundWarmupPromise = ensureOnnxWorkerReady()
            .then((workerStatus) => {
                runtimeLog('onnx:prewarm-worker-success', {
                    durationMs: formatOnnxDurationMs(startedAt),
                    provider: workerStatus?.provider || onnxSelectedProvider
                });
            })
            .catch((error) => {
                runtimeLog('onnx:prewarm-worker-failed', {
                    durationMs: formatOnnxDurationMs(startedAt),
                    error: serializeOnnxError(error)
                });
            })
            .finally(() => {
                onnxBackgroundWarmupPromise = null;
            });
    }

    runtimeLog('onnx:prewarm-dispatched', {
        durationMs: formatOnnxDurationMs(startedAt)
    });
}

function resetOnnxAdapterState() {
    const laneKeys = Object.keys(onnxWorkerLaneStates);
    for (const laneKey of laneKeys) {
        resetOnnxWorkerState(laneKey);
    }

    for (const laneKey of laneKeys) {
        delete onnxWorkerLaneStates[laneKey];
    }

    onnxWorkerLaneStates[ONNX_WORKER_LANE_FOREGROUND] = createOnnxWorkerLaneState();
    onnxWorkerLaneStates[ONNX_WORKER_LANE_BACKGROUND] = createOnnxWorkerLaneState();

    onnxSession = null;
    onnxSessionPromise = null;
    onnxInitialized = false;
    onnxSelectedProvider = null;
    onnxPreprocessCanvas = null;
    onnxPreprocessContext = null;
    onnxWorkingCanvasPool = {
        [ONNX_WORKER_LANE_FOREGROUND]: Object.create(null),
        [ONNX_WORKER_LANE_BACKGROUND]: Object.create(null)
    };
}

function isOnnxRuntimeSupported() {
    return typeof Worker === 'function' && !!chrome?.runtime?.getURL;
}

window.OnnxRuntimeAdapter = createLibraryBackedEngineAdapter({
    getLibrary: () => window,
    isLibrarySupported: isOnnxRuntimeSupported,
    upscale: runOnnxUpscale,
    ensureReady: prewarmOnnx,
    resetState: resetOnnxAdapterState,
    isInitialized: () => onnxInitialized,
    isRuntimeSupported: isOnnxRuntimeSupported
});
