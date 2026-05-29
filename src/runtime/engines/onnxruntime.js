// ONNX Runtime adapter — model-agnostic engine. Configure the active model via onnxActiveModel (see onnx-models.js).

let onnxInitialized = false;
let onnxSelectedProvider = null;
let onnxBackgroundWarmupPromise = null;
const ONNX_WORKER_LANE_FOREGROUND = 'foreground';
const ONNX_WORKER_LANE_BACKGROUND = 'background';
let onnxWorkingCanvasPool = {
    [ONNX_WORKER_LANE_FOREGROUND]: Object.create(null),
    [ONNX_WORKER_LANE_BACKGROUND]: Object.create(null)
};
const ONNX_WORKER_PENDING_WARNING_MS = 5000;
// ORT-Web sessions are not re-entrant; keep one run per worker/session.
const ONNX_WORKER_MAX_INFLIGHT_RUN_REQUESTS = 1;
const ONNX_INIT_RETRY_BASE_DELAY_MS = 1000;
const ONNX_INIT_RETRY_MAX_DELAY_MS = 30000;
const ONNX_TILE_MAX_PARALLELISM = 1;

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
const ONNX_INIT_WARMUP_TILE_EDGE_MAX = 512;
const ONNX_MODEL_SWITCH_FALLBACK_KEY = 'realesr-animevideov3';
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

    if (typeof lane === 'string' && lane.startsWith(`${ONNX_WORKER_LANE_FOREGROUND}-`)) {
        return lane;
    }

    if (typeof lane === 'string' && lane.startsWith(ONNX_WORKER_LANE_BACKGROUND)) {
        return lane;
    }

    return ONNX_WORKER_LANE_FOREGROUND;
}

function getOnnxParallelExecutionLane(baseLane, slotIndex = 0) {
    const normalizedLane = normalizeOnnxExecutionLane(baseLane);
    const normalizedSlotIndex = Number.isFinite(slotIndex) ? Math.max(0, Math.floor(slotIndex)) : 0;

    if (normalizedSlotIndex <= 0) {
        return normalizedLane;
    }

    return `${normalizedLane}-${normalizedSlotIndex}`;
}

function getOnnxExecutionLanes(baseLane, maxParallelism = 1) {
    const normalizedMaxParallelism = Number.isFinite(maxParallelism)
        ? Math.max(1, Math.floor(maxParallelism))
        : 1;

    return Array.from({ length: normalizedMaxParallelism }, (_unused, slotIndex) => {
        return getOnnxParallelExecutionLane(baseLane, slotIndex);
    });
}

function getOnnxBackgroundExecutionLane(slotIndex = 0) {
    return getOnnxParallelExecutionLane(ONNX_WORKER_LANE_BACKGROUND, slotIndex);
}

function getOnnxBackgroundQueueMaxConcurrency(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);

    let configured = 1;
    if (typeof resolveOnnxModelByKey === 'function') {
        const model = resolveOnnxModelByKey(settings.selectedOnnxModel);
        configured = Number(model?.queueParallelImages);
    } else if (typeof onnxActiveModel === 'object' && onnxActiveModel) {
        configured = Number(onnxActiveModel.queueParallelImages);
    }

    const normalized = Number.isFinite(configured) ? Math.floor(configured) : 1;
    return Math.max(1, normalized);
}

function getOnnxWorkerLaneState(lane = ONNX_WORKER_LANE_FOREGROUND) {
    const normalizedLane = normalizeOnnxExecutionLane(lane);
    if (!onnxWorkerLaneStates[normalizedLane]) {
        onnxWorkerLaneStates[normalizedLane] = createOnnxWorkerLaneState();
    }
    return onnxWorkerLaneStates[normalizedLane];
}

function getOnnxInitRetryDelayMs(consecutiveInitFailures) {
    const failures = Number.isFinite(consecutiveInitFailures)
        ? Math.max(1, Math.floor(consecutiveInitFailures))
        : 1;

    return Math.min(
        ONNX_INIT_RETRY_MAX_DELAY_MS,
        ONNX_INIT_RETRY_BASE_DELAY_MS * Math.pow(2, Math.max(0, failures - 1))
    );
}

function applyOnnxWorkerInitSuccess(laneState, lane, selectedProvider) {
    laneState.initialized = true;
    laneState.selectedProvider = selectedProvider || 'wasm-worker';
    laneState.consecutiveInitFailures = 0;
    laneState.blockedUntilMs = 0;

    if (lane === ONNX_WORKER_LANE_FOREGROUND) {
        onnxInitialized = true;
        onnxSelectedProvider = laneState.selectedProvider;
    }
}

function applyOnnxWorkerInitFailure(laneState, lane, startedAt, error) {
    laneState.consecutiveInitFailures = (laneState.consecutiveInitFailures || 0) + 1;
    const delayMs = getOnnxInitRetryDelayMs(laneState.consecutiveInitFailures);
    laneState.blockedUntilMs = getOnnxNow() + delayMs;

    runtimeLog('onnx:init-failed', {
        lane,
        model: onnxActiveModel.label,
        modelPath: onnxActiveModel.modelPath,
        durationMs: formatOnnxDurationMs(startedAt),
        consecutiveInitFailures: laneState.consecutiveInitFailures,
        retryBlockedForMs: delayMs,
        error: serializeOnnxError(error)
    });

    resetOnnxWorkerState(lane);
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

function getOnnxLibrary() {
    const lib = window.ort;
    return lib && typeof lib === 'object' ? lib : null;
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
    return typedArray.buffer;
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

function resetOnnxWorkerLanes(lanes = [], terminate = true) {
    for (const lane of lanes) {
        resetOnnxWorkerState(lane, terminate);
    }
}

function resetOnnxBackgroundWorkerLanes(maxParallelism = 1) {
    const lanes = getOnnxExecutionLanes(ONNX_WORKER_LANE_BACKGROUND, maxParallelism);
    resetOnnxWorkerLanes(lanes, true);
    return lanes.length;
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
        console.warn(`[NH] onnx-blocked lane=${normalizedLane}`, {
            remainingMs: Math.round(laneState.blockedUntilMs - now),
            consecutiveInitFailures: laneState.consecutiveInitFailures
        });
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
            const runtimeTileEdge = resolveOnnxTileEdgePixels();
            const initTileEdge = Math.max(
                onnxActiveModel.minTileEdgePixels,
                Math.min(runtimeTileEdge, ONNX_INIT_WARMUP_TILE_EDGE_MAX)
            );

            runtimeLog('onnx:worker-init-dispatch', {
                lane: normalizedLane,
                forceRefresh,
                modelBytes: artifacts.modelBytes.byteLength,
                externalDataBytes: artifacts.externalDataBytes ? artifacts.externalDataBytes.byteLength : 0,
                modelCacheHit: artifacts.modelCacheHit,
                runtimeTileEdge,
                initTileEdge,
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
                externalDataBytes: externalDataBuffer,
                onnxProfilingEnabled: !!window.MangaScalerProfiling?.isOnnxEnabled?.(),
                warmupWidth: initTileEdge,
                warmupHeight: initTileEdge,
                preferredInputWidth: initTileEdge,
                preferredInputHeight: initTileEdge,
                inputChannels: Number(onnxActiveModel.inputChannels || 3),
                outputChannels: Number(onnxActiveModel.outputChannels || 3),
                graphCaptureEnabled: false
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

            applyOnnxWorkerInitSuccess(laneState, normalizedLane, initResult?.provider);

            runtimeLog('onnx:init', {
                lane: normalizedLane,
                model: onnxActiveModel.label,
                provider: laneState.selectedProvider,
                modelPath: onnxActiveModel.modelPath,
                durationMs: formatOnnxDurationMs(startedAt)
            });

            return initResult;
        } catch (error) {
            applyOnnxWorkerInitFailure(laneState, normalizedLane, startedAt, error);
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

function getOnnxPreprocessContext(width, height) {
    const lib = getOnnxLibrary();
    if (!lib?.env || !chrome?.runtime?.getURL) return;

    const distPath = chrome.runtime.getURL('node_modules/onnxruntime-web/dist/');

    if (lib.env.wasm) {
        // Provide explicit module and wasm URLs so worker/blob contexts do not resolve relative .mjs files from about:blank.
        lib.env.wasm.wasmPaths = {
            mjs: `${distPath}ort-wasm-simd-threaded.jsep.mjs`,
            wasm: `${distPath}ort-wasm-simd-threaded.jsep.wasm`
        };
        const coreCount = navigator.hardwareConcurrency || 4;
        lib.env.wasm.numThreads = Math.min(4, coreCount);
        lib.env.wasm.proxy = false;
    }

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

    if (canvas.width < width) {
    canvas.width = width;
    }
    if (canvas.height < height) {
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

function getOnnxOutputColorCorrection() {
    const modelKey = String(onnxActiveModel?.key || '');
    if (!modelKey.endsWith('-unet1-only')) {
        return null;
    }

    // UNet1-only exports can appear slightly darker than the full UNet1+UNet2 path.
    // Apply a conservative RGB lift/gain to restore perceived luminance.
    return {
        gain: 1.06,
        lift: 1
    };
}

function applyOnnxRgbaColorCorrection(pixelBytes, correction) {
    if (!(pixelBytes instanceof Uint8ClampedArray) || !correction) {
        return;
    }

    const gain = Number(correction.gain);
    const lift = Number(correction.lift);
    if (!Number.isFinite(gain) || !Number.isFinite(lift)) {
        return;
    }

    for (let index = 0; index < pixelBytes.length; index += 4) {
        pixelBytes[index] = Math.max(0, Math.min(255, Math.round(pixelBytes[index] * gain + lift)));
        pixelBytes[index + 1] = Math.max(0, Math.min(255, Math.round(pixelBytes[index + 1] * gain + lift)));
        pixelBytes[index + 2] = Math.max(0, Math.min(255, Math.round(pixelBytes[index + 2] * gain + lift)));
    }
}

async function imageToOnnxWorkerInput(image) {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;

    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        throw new Error('Invalid source image dimensions for ONNX Runtime');
    }

    // Fast path: create a transferable ImageBitmap (zero-copy IPC) and let the worker
    // pass it directly to Tensor.fromImage — ORT's WebGPU backend can import it as a
    // GPU texture without a CPU round-trip.
    if (typeof createImageBitmap === 'function') {
        const imageBitmap = await createImageBitmap(image);
        return { width, height, imageBitmap };
    }

    // Fallback: extract RGBA pixels on the CPU (willReadFrequently canvas)
    const ctx = getOnnxPreprocessContext(width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    return { width, height, rgbaBuffer: imageData.data.buffer };
}

function drawOnnxWorkerOutputToCanvas(workerResult, canvas, cropWidth, cropHeight) {
    const outWidth = workerResult?.outputWidth;
    const outHeight = workerResult?.outputHeight;
    const rgbaBuffer = workerResult?.rgbaBuffer;

    if (!Number.isFinite(outWidth) || !Number.isFinite(outHeight) || outWidth < 1 || outHeight < 1) {
        throw new Error('Worker returned invalid ONNX output dimensions');
    }

    if (!(rgbaBuffer instanceof ArrayBuffer)) {
        throw new Error('Worker returned invalid ONNX output pixels');
    }

    const finalWidth = (cropWidth != null) ? Math.max(1, Math.min(outWidth, Math.round(cropWidth))) : outWidth;
    const finalHeight = (cropHeight != null) ? Math.max(1, Math.min(outHeight, Math.round(cropHeight))) : outHeight;

    canvas.width = finalWidth;
    canvas.height = finalHeight;

    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) {
        throw new Error('Failed to acquire output canvas context for ONNX Runtime');
    }

    const outputPixels = new Uint8ClampedArray(rgbaBuffer);
    const colorCorrection = getOnnxOutputColorCorrection();
    if (colorCorrection) {
        applyOnnxRgbaColorCorrection(outputPixels, colorCorrection);
    }

    const imageData = new ImageData(outputPixels, outWidth, outHeight);
    ctx.putImageData(imageData, 0, 0, 0, 0, finalWidth, finalHeight);
}

async function runOnnxInferencePass(inputImage, outputCanvas, executionOptions = {}) {
    const executionLane = normalizeOnnxExecutionLane(executionOptions?.lane);
    const canvasSlotSuffix = executionOptions?.canvasSlotSuffix != null
        ? `-${executionOptions.canvasSlotSuffix}`
        : '';
    const passStartAt = getOnnxNow();
    const inputWidth = inputImage.naturalWidth || inputImage.width || 0;
    const inputHeight = inputImage.naturalHeight || inputImage.height || 0;
    const inputPadding = getOnnxInputPadding(inputWidth, inputHeight);
    const needsInputPadding = inputPadding.padRight > 0 || inputPadding.padBottom > 0;

    const tensorBuildStartAt = getOnnxNow();
    let workerSourceImage = inputImage;
    if (needsInputPadding) {
        workerSourceImage = drawOnnxImageWithEdgePadding(
            inputImage,
            getOnnxReusableWorkingCanvas(
                `padded-input${canvasSlotSuffix}`,
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

    const workerInput = await imageToOnnxWorkerInput(workerSourceImage);
    workerInput.lane = executionLane;
    const tensorBuildEndAt = getOnnxNow();
    await ensureOnnxWorkerReady(executionLane);

    const inferenceStartAt = getOnnxNow();
    const workerResult = await postOnnxWorkerRequest(
        'run', workerInput,
        workerInput.imageBitmap != null ? [workerInput.imageBitmap]
            : workerInput.rgbaBuffer != null ? [workerInput.rgbaBuffer] : [],
        executionLane
    );
    const inferenceEndAt = getOnnxNow();

    const canvasWriteStartAt = getOnnxNow();
    if (needsInputPadding) {
        const scaleX = workerResult.outputWidth / workerInput.width;
        const scaleY = workerResult.outputHeight / workerInput.height;
        const croppedOutputWidth = Math.max(1, Math.round(inputWidth * scaleX));
        const croppedOutputHeight = Math.max(1, Math.round(inputHeight * scaleY));
        drawOnnxWorkerOutputToCanvas(workerResult, outputCanvas, croppedOutputWidth, croppedOutputHeight);
    } else {
        drawOnnxWorkerOutputToCanvas(workerResult, outputCanvas);
    }
    const canvasWriteEndAt = getOnnxNow();

    runtimeProfileLog('onnx:pass-profile', {
        lane: executionLane,
        tileIndex: executionOptions?.tileIndex ?? null,
        prepareMs: executionOptions?.tilePrepareMs ?? null,
        bitmapMs: formatOnnxDurationMs(tensorBuildStartAt, tensorBuildEndAt),
        workerBuildMs: workerResult.workerTensorBuildMs ?? null,
        gpuMs: workerResult.workerGpuTotalMs ?? workerResult.workerInferenceMs ?? null,
        gpuRunMs: workerResult.workerGpuFirstRunMs ?? null,
        gpuRetryMs: workerResult.workerGpuRetryRunMs ?? null,
        gpuOutputMs: workerResult.workerOutputConvertMs ?? null,
        gpuKernelCount: workerResult.workerGpuKernelCount ?? null,
        gpuKernelAvgMs: workerResult.workerGpuKernelAvgMs ?? null,
        gpuKernelTotalMs: workerResult.workerGpuKernelTotalMs ?? null,
        gpuKernelMaxMs: workerResult.workerGpuKernelMaxMs ?? null,
        gpuKernelTop: workerResult.workerGpuKernelTop ?? null,
        gpuKernelHotspots: workerResult.workerGpuKernelHotspots ?? null,
        gpuKernelSlowestHotspots: workerResult.workerGpuKernelSlowestHotspots ?? null,
        gpuKernelFirstEvent: workerResult.workerGpuKernelFirstEvent ?? null,
        gpuKernelLastEvent: workerResult.workerGpuKernelLastEvent ?? null,
        readbackMs: workerResult.workerDataReadMs ?? null,
        pixelMs: workerResult.workerPixelLoopMs ?? null,
        putMs: formatOnnxDurationMs(canvasWriteStartAt, canvasWriteEndAt),
        totalMs: formatOnnxDurationMs(passStartAt)
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
    const tileEdge = resolveOnnxTileEdgePixels();
    const overlap = onnxActiveModel.tileOverlapPixels;
    const coreStep = Math.max(64, tileEdge - overlap * 2);
    const tileColumns = Math.ceil(sourceWidth / coreStep);
    const tileRows = Math.ceil(sourceHeight / coreStep);
    const maxTileParallelism = Math.max(1, Math.floor(ONNX_TILE_MAX_PARALLELISM));
    const tileExecutionLanes = getOnnxExecutionLanes(executionLane, maxTileParallelism);

    await Promise.all(
        tileExecutionLanes.map((lane) => ensureOnnxWorkerReady(lane))
    );

    const destCtx = outputCanvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!destCtx) {
        throw new Error('Failed to acquire output canvas context for ONNX tiled compose');
    }

    let scaleX = null;
    let scaleY = null;
    let tileCount = 0;
    let totalInferenceMs = 0;
    const tileSpecs = [];

    runtimeLog('onnx:tiled-layout', {
        lane: executionLane,
        sourceWidth,
        sourceHeight,
        tileEdge,
        overlap,
        coreStep,
        tileColumns,
        tileRows,
        estimatedTileCount: tileColumns * tileRows,
        tileParallelism: maxTileParallelism
    });

    for (let coreY = 0; coreY < sourceHeight; coreY += coreStep) {
        const coreBottom = Math.min(sourceHeight, coreY + coreStep);

        for (let coreX = 0; coreX < sourceWidth; coreX += coreStep) {
            const coreRight = Math.min(sourceWidth, coreX + coreStep);

            // For the last column/row tile the naive start position (coreX - overlap)
            // leaves only a thin sliver of real content, requiring hundreds of pixels of
            // edge-replication padding.  RealCUGAN's UNet sees the whole 512-px input at
            // its deepest scale, so 90 % artificial padding pollutes the global features
            // and makes the core output look like a blurry bilinear upscale.
            // Fix: slide the tile's left/top edge back so it always spans a full tileEdge
            // of real source pixels.  The extra left/top overlap is simply discarded when
            // compositing, the core pixels remain correct.
            const isLastColumn = coreRight >= sourceWidth;
            const isLastRow    = coreBottom >= sourceHeight;

            const tileInputX = (isLastColumn && sourceWidth  >= tileEdge)
                ? sourceWidth  - tileEdge
                : Math.max(0, coreX - overlap);
            const tileInputY = (isLastRow    && sourceHeight >= tileEdge)
                ? sourceHeight - tileEdge
                : Math.max(0, coreY - overlap);
            const tileInputRight  = Math.min(sourceWidth,  coreRight  + overlap);
            const tileInputBottom = Math.min(sourceHeight, coreBottom + overlap);

            const tileInputWidth = tileInputRight - tileInputX;
            const tileInputHeight = tileInputBottom - tileInputY;
            tileSpecs.push({
                coreX,
                coreY,
                coreRight,
                coreBottom,
                tileInputX,
                tileInputY,
                tileInputRight,
                tileInputBottom,
                tileInputWidth,
                tileInputHeight,
                tileRow: Math.floor(coreY / coreStep),
                tileCol: Math.floor(coreX / coreStep)
            });
        }
    }

    async function processTile(tileSpec, slotIndex, tileIndex) {
        const tileExecutionLane = tileExecutionLanes[slotIndex] || executionLane;
        const tileInputCanvas = getOnnxReusableWorkingCanvas(`tile-input-${slotIndex}`, tileEdge, tileEdge, tileExecutionLane);
        const tileOutputCanvas = getOnnxReusableWorkingCanvas(`tile-output-${slotIndex}`, tileEdge, tileEdge, tileExecutionLane);

        tileInputCanvas.width = tileSpec.tileInputWidth;
        tileInputCanvas.height = tileSpec.tileInputHeight;

        const tileInputCtx = tileInputCanvas.getContext('2d', { alpha: false, desynchronized: true });
        if (!tileInputCtx) {
            throw new Error('Failed to acquire tile input canvas context for ONNX');
        }

        const tilePrepareStartAt = getOnnxNow();
        tileInputCtx.clearRect(0, 0, tileSpec.tileInputWidth, tileSpec.tileInputHeight);
        tileInputCtx.drawImage(
            sourceImage,
            tileSpec.tileInputX, tileSpec.tileInputY, tileSpec.tileInputWidth, tileSpec.tileInputHeight,
            0, 0, tileSpec.tileInputWidth, tileSpec.tileInputHeight
        );
        const tilePrepareEndAt = getOnnxNow();

        const pass = await runOnnxInferencePass(tileInputCanvas, tileOutputCanvas, {
            ...executionOptions,
            lane: tileExecutionLane,
            tileIndex,
            tileRow: tileSpec.tileRow,
            tileCol: tileSpec.tileCol,
            tilePrepareMs: formatOnnxDurationMs(tilePrepareStartAt, tilePrepareEndAt),
            canvasSlotSuffix: slotIndex
        });

        if (scaleX === null || scaleY === null) {
            scaleX = pass.outputWidth / tileSpec.tileInputWidth;
            scaleY = pass.outputHeight / tileSpec.tileInputHeight;
            outputCanvas.width = Math.max(1, Math.round(sourceWidth * scaleX));
            outputCanvas.height = Math.max(1, Math.round(sourceHeight * scaleY));
            destCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        }

        const cropLeftIn = tileSpec.coreX - tileSpec.tileInputX;
        const cropTopIn = tileSpec.coreY - tileSpec.tileInputY;
        const cropRightIn = tileSpec.tileInputRight - tileSpec.coreRight;
        const cropBottomIn = tileSpec.tileInputBottom - tileSpec.coreBottom;

        const cropLeftOut = Math.max(0, Math.round(cropLeftIn * scaleX));
        const cropTopOut = Math.max(0, Math.round(cropTopIn * scaleY));
        const cropRightOut = Math.max(0, Math.round(cropRightIn * scaleX));
        const cropBottomOut = Math.max(0, Math.round(cropBottomIn * scaleY));

        const drawWidth = Math.max(1, pass.outputWidth - cropLeftOut - cropRightOut);
        const drawHeight = Math.max(1, pass.outputHeight - cropTopOut - cropBottomOut);
        const destX = Math.max(0, Math.round(tileSpec.coreX * scaleX));
        const destY = Math.max(0, Math.round(tileSpec.coreY * scaleY));

        destCtx.drawImage(
            tileOutputCanvas,
            cropLeftOut, cropTopOut, drawWidth, drawHeight,
            destX, destY, drawWidth, drawHeight
        );

        const stepInferenceMs = typeof pass.inferenceMs === 'string'
            ? parseFloat(pass.inferenceMs)
            : (pass.inferenceMs || 0);
        totalInferenceMs += stepInferenceMs;
        tileCount += 1;

        if (ONNX_TILE_YIELD_EVERY_TILES > 0 && tileCount % ONNX_TILE_YIELD_EVERY_TILES === 0) {
            await yieldOnnxTileLoop();
        }
    }

    if (tileSpecs.length > 0) {
        await processTile(tileSpecs[0], 0, 1);

        let nextTileSpecIndex = 1;
        const tileWorkers = Array.from({ length: tileExecutionLanes.length }, async (_unused, slotIndex) => {
            while (nextTileSpecIndex < tileSpecs.length) {
                const tileIndex = nextTileSpecIndex + 1;
                const tileSpec = tileSpecs[nextTileSpecIndex];
                nextTileSpecIndex += 1;
                await processTile(tileSpec, slotIndex, tileIndex);
            }
        });

        await Promise.all(tileWorkers);
    }

    runtimeLog('onnx:tiled-run-success', {
        lane: executionLane,
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
            });
    }
    return onnxBackgroundWarmupPromise;
}


function resetOnnxAdapterState() {
    const laneKeys = Object.keys(onnxWorkerLaneStates);
    resetOnnxWorkerLanes(laneKeys, true);

    for (const laneKey of laneKeys) {
        delete onnxWorkerLaneStates[laneKey];
    }

    onnxWorkerLaneStates[ONNX_WORKER_LANE_FOREGROUND] = createOnnxWorkerLaneState();
    onnxWorkerLaneStates[ONNX_WORKER_LANE_BACKGROUND] = createOnnxWorkerLaneState();

    onnxInitialized = false;
    onnxSelectedProvider = null;
    onnxWorkingCanvasPool = {
        [ONNX_WORKER_LANE_FOREGROUND]: Object.create(null),
        [ONNX_WORKER_LANE_BACKGROUND]: Object.create(null)
    };
}

function isOnnxRuntimeSupported() {
    return typeof Worker === 'function' && !!chrome?.runtime?.getURL;
}

window.resetOnnxWorkerState = resetOnnxWorkerState;
window.resetOnnxBackgroundWorkerLanes = resetOnnxBackgroundWorkerLanes;

const onnxRuntimeAdapter = createLibraryBackedEngineAdapter({
    getLibrary: () => window,
    isLibrarySupported: isOnnxRuntimeSupported,
    upscale: runOnnxUpscale,
    ensureReady: prewarmOnnx,
    resetState: resetOnnxAdapterState,
    isInitialized: () => onnxInitialized,
    isRuntimeSupported: isOnnxRuntimeSupported
});

onnxRuntimeAdapter.getBackgroundExecutionLane = getOnnxBackgroundExecutionLane;
onnxRuntimeAdapter.getBackgroundQueueMaxConcurrency = getOnnxBackgroundQueueMaxConcurrency;
onnxRuntimeAdapter.resetBackgroundWorkers = resetOnnxBackgroundWorkerLanes;

window.OnnxRuntimeAdapter = onnxRuntimeAdapter;
