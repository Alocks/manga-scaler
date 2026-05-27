let workerOrt = null;
let workerSession = null;
let workerInitPromise = null;
let workerProvider = null;
let workerRunCounter = 0;
let workerCanUseTensorFromImage = null;
let workerCanUseBitmapFromImage = null;
// Output scale is detected once on the first tile and cached for all subsequent tiles.
let workerCachedOutputScale = null;
const WORKER_RUN_WARNING_MS = 5000;
const WORKER_BYTE_TO_UNIT_FLOAT = 0.00392156862745098;
let workerWarnFilterInstalled = false;
let workerOnnxProfilingEnabled = false;
let workerActiveGpuProfileCollector = null;

function createWorkerGpuProfileCollector() {
    return {
        totalMs: 0,
        kernelCount: 0,
        maxMs: 0,
        events: [],
        hotspots: Object.create(null)
    };
}

function getWorkerGpuProfileEventDurationMs(data) {
    if (!data || typeof data !== 'object') {
        return 0;
    }

    const startTime = Number(data.startTime);
    const endTime = Number(data.endTime);
    if (Number.isFinite(startTime) && Number.isFinite(endTime) && endTime >= startTime) {
        return Math.max(0, (endTime - startTime) / 1e6);
    }

    const directDuration = Number(data.durationMs ?? data.duration ?? data.elapsedMs ?? data.timeMs ?? data.totalMs ?? 0);
    return Number.isFinite(directDuration) && directDuration >= 0 ? directDuration : 0;
}

function finalizeWorkerGpuProfileSummary(collector) {
    if (!collector) {
        return null;
    }

    const topEvents = [...collector.events]
        .sort((left, right) => right.durationMs - left.durationMs || left.eventIndex - right.eventIndex)
        .slice(0, 10);

    const topHotspots = Object.values(collector.hotspots)
        .map((hotspot) => ({
            ...hotspot,
            avgMs: hotspot.count > 0 ? Number((hotspot.totalMs / hotspot.count).toFixed(3)) : 0
        }))
        .sort((left, right) => right.totalMs - left.totalMs || right.count - left.count || right.maxMs - left.maxMs)
        .slice(0, 12);

    const slowestHotspots = Object.values(collector.hotspots)
        .map((hotspot) => ({
            ...hotspot,
            avgMs: hotspot.count > 0 ? Number((hotspot.totalMs / hotspot.count).toFixed(3)) : 0
        }))
        .sort((left, right) => right.avgMs - left.avgMs || right.maxMs - left.maxMs || right.count - left.count)
        .slice(0, 12);

    return {
        kernelCount: collector.kernelCount,
        totalMs: collector.totalMs,
        maxMs: collector.maxMs,
        avgMs: collector.kernelCount > 0 ? Number((collector.totalMs / collector.kernelCount).toFixed(3)) : 0,
        topEvents,
        topHotspots,
        slowestHotspots,
        firstEvent: collector.events[0] || null,
        lastEvent: collector.events.length > 0 ? collector.events[collector.events.length - 1] : null
    };
}

function getWorkerNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
}

function formatWorkerDurationMs(startAt, endAt = getWorkerNow()) {
    return Number((endAt - startAt).toFixed(2));
}

function serializeWorkerError(error) {
    if (!error) {
        return { message: 'Unknown ONNX worker error' };
    }

    return {
        message: String(error.message || error),
        name: error.name ? String(error.name) : null,
        stack: typeof error.stack === 'string' ? error.stack.split('\n').slice(0, 6).join('\n') : null
    };
}

function postWorkerLog(event, payload = {}) {
    self.postMessage({
        type: 'log',
        event,
        payload
    });
}

function loadOrtRuntime(ortScriptUrl) {
    if (workerOrt) {
        return workerOrt;
    }

    importScripts(ortScriptUrl);
    workerOrt = self.ort;

    if (!workerOrt || !workerOrt.InferenceSession || typeof workerOrt.Tensor !== 'function') {
        throw new Error('ONNX Runtime Web failed to initialize inside worker');
    }

    return workerOrt;
}

function installOrtWarningFilter() {
    if (workerWarnFilterInstalled) {
        return;
    }

    const originalWarn = console.warn.bind(console);
    console.warn = (...args) => {
        const message = args.map((value) => {
            if (typeof value === 'string') return value;
            if (value && typeof value.message === 'string') return value.message;
            return String(value);
        }).join(' ');

        if (
            message.includes('VerifyEachNodeIsAssignedToAnEp') ||
            message.includes('session_state.cc:1367') ||
            message.includes('session_state.cc:1369')
        ) {
            return;
        }

        originalWarn(...args);
    };

    workerWarnFilterInstalled = true;
}

function installOrtGpuProfilingCollector(lib) {
    if (!lib?.env?.webgpu) {
        return;
    }

    lib.env.webgpu.profiling = {
        mode: workerOnnxProfilingEnabled ? 'default' : 'off',
        ondata(data) {
            const collector = workerActiveGpuProfileCollector;
            if (!collector || !data || typeof data !== 'object') {
                return;
            }

            if (!Array.isArray(collector.events)) {
                collector.events = [];
            }
            if (!collector.hotspots || typeof collector.hotspots !== 'object') {
                collector.hotspots = Object.create(null);
            }

            const durationMs = getWorkerGpuProfileEventDurationMs(data);
            const startTime = Number(data.startTime);
            const endTime = Number(data.endTime);
            const kernelName = String(data.kernelName || data.name || data.label || 'unknown');
            const programName = String(data.programName || data.shaderName || 'unknown');
            const kernelType = String(data.kernelType || data.type || data.category || 'unknown');
            const eventIndex = collector.events.length;
            const event = {
                eventIndex,
                kernelId: data.kernelId ?? null,
                kernelType,
                kernelName,
                programName,
                startTime: Number.isFinite(startTime) ? startTime : null,
                endTime: Number.isFinite(endTime) ? endTime : null,
                durationMs: Number(durationMs.toFixed(3))
            };

            collector.totalMs += durationMs;
            collector.kernelCount += 1;
            collector.maxMs = Math.max(collector.maxMs, durationMs);
            collector.events.push(event);

            const hotspotKey = `${kernelType}::${programName}::${kernelName}`;
            if (!collector.hotspots[hotspotKey]) {
                collector.hotspots[hotspotKey] = {
                    hotspotKey,
                    kernelType,
                    programName,
                    kernelName,
                    count: 0,
                    totalMs: 0,
                    maxMs: 0,
                    firstEventIndex: eventIndex,
                    lastEventIndex: eventIndex
                };
            }

            const hotspot = collector.hotspots[hotspotKey];
            hotspot.count += 1;
            hotspot.totalMs = Number((hotspot.totalMs + durationMs).toFixed(6));
            hotspot.maxMs = Math.max(hotspot.maxMs, durationMs);
            hotspot.lastEventIndex = eventIndex;
        }
    };
}

function configureWorkerEnvironment(lib, ortDistUrl) {
    if (!lib?.env?.wasm) {
        return;
    }

    installOrtWarningFilter();
    installOrtGpuProfilingCollector(lib);
    lib.env.logLevel = 'error';
    lib.env.wasm.wasmPaths = {
        mjs: `${ortDistUrl}ort-wasm-simd-threaded.jsep.mjs`,
        wasm: `${ortDistUrl}ort-wasm-simd-threaded.jsep.wasm`
    };
    lib.env.wasm.numThreads = (self.crossOriginIsolated && self.navigator?.hardwareConcurrency > 1)
        ? Math.min(4, self.navigator.hardwareConcurrency)
        : 1;
    lib.env.wasm.proxy = false;
}

function resolveWorkerOutputTensor(runResult, outputNames = []) {
    if (!runResult || typeof runResult !== 'object') {
        throw new Error('ONNX worker received an invalid inference result');
    }

    if (outputNames.length > 0 && runResult[outputNames[0]]) {
        return runResult[outputNames[0]];
    }

    const firstTensor = Object.values(runResult)[0];
    if (!firstTensor) {
        throw new Error('ONNX worker did not receive any output tensors');
    }

    return firstTensor;
}

function inferWorkerOutputScale(data) {
    const sampleCount = Math.min(2048, data.length);
    let maxValue = 0;
    for (let i = 0; i < sampleCount; i++) {
        const value = data[i];
        // Skip Inf/NaN — these are model artefacts, not a signal for [0,255] range.
        if (Number.isFinite(value) && value > maxValue) {
            maxValue = value;
        }
    }
    // Threshold is set to 10 rather than the previous 1.5.
    // RealCUGAN's UpCunet2x sums two network paths (unet1 + unet2), so its
    // output can naturally reach ~1.5–2.0 for saturated/bright pixels even
    // though the model is trained on [0,1] targets.  Any legitimate [0,255]
    // model will have peak values well above 10 throughout the tensor.
    return maxValue > 10 ? 1 : 255;
}

async function imageBufferToTensor(rgbaBuffer, width, height) {
    if (!(rgbaBuffer instanceof ArrayBuffer)) {
        throw new Error('ONNX worker received an invalid input pixel buffer');
    }

    if (
        workerCanUseTensorFromImage !== false &&
        typeof workerOrt?.Tensor?.fromImage === 'function' &&
        typeof ImageData === 'function'
    ) {
        try {
            const imageData = new ImageData(new Uint8ClampedArray(rgbaBuffer), width, height);
            const tensor = await workerOrt.Tensor.fromImage(imageData, {
                tensorLayout: 'NCHW',
                tensorFormat: 'RGB',
                dataType: 'float32'
            });
            workerCanUseTensorFromImage = true;
            return tensor;
        } catch (error) {
            workerCanUseTensorFromImage = false;
            postWorkerLog('onnx:worker-fromimage-fallback', {
                reason: String(error),
                disabledAfterFailure: true
            });
        }
    }

    const pixelCount = width * height;
    const chw = new Float32Array(pixelCount * 3);
    writeImageBufferToChw(rgbaBuffer, width, height, chw, 0);

    return new workerOrt.Tensor('float32', chw, [1, 3, height, width]);
}

async function imageBitmapToTensor(imageBitmap) {
    if (
        workerCanUseBitmapFromImage !== false &&
        typeof workerOrt?.Tensor?.fromImage === 'function'
    ) {
        try {
            const tensor = await workerOrt.Tensor.fromImage(imageBitmap, {
                tensorLayout: 'NCHW',
                tensorFormat: 'RGB',
                dataType: 'float32'
            });
            workerCanUseBitmapFromImage = true;
            imageBitmap.close();
            return tensor;
        } catch (bitmapError) {
            workerCanUseBitmapFromImage = false;
            postWorkerLog('onnx:worker-bitmap-fromimage-fallback', { reason: String(bitmapError) });
        }
    }

    // Fallback: draw to OffscreenCanvas and extract ImageData on the CPU
    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(imageBitmap, 0, 0);
    imageBitmap.close();
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return imageBufferToTensor(imageData.data.buffer, canvas.width, canvas.height);
}

function writeImageBufferToChw(rgbaBuffer, width, height, targetChw, targetOffset = 0) {
    if (!(rgbaBuffer instanceof ArrayBuffer)) {
        throw new Error('ONNX worker received an invalid input pixel buffer');
    }

    const rgba = new Uint8Array(rgbaBuffer);
    const pixelCount = width * height;
    const gBase = targetOffset + pixelCount;
    const bBase = targetOffset + pixelCount * 2;

    for (let i = 0; i < pixelCount; i++) {
        const srcOffset = i << 2;
        targetChw[targetOffset + i] = rgba[srcOffset] * WORKER_BYTE_TO_UNIT_FLOAT;
        targetChw[gBase + i] = rgba[srcOffset + 1] * WORKER_BYTE_TO_UNIT_FLOAT;
        targetChw[bBase + i] = rgba[srcOffset + 2] * WORKER_BYTE_TO_UNIT_FLOAT;
    }
}

function tensorToImageBuffer(outputTensor, batchIndex = 0) {
    if (!outputTensor || !Array.isArray(outputTensor.dims) || outputTensor.dims.length !== 4) {
        throw new Error('Unexpected worker ONNX output tensor shape. Expected rank-4 [N,C,H,W]');
    }

    const [batch, channels, outHeight, outWidth] = outputTensor.dims;
    if (!Number.isFinite(batch) || batch < 1 || channels < 3 || outHeight < 1 || outWidth < 1) {
        throw new Error(`Invalid worker ONNX output dims: ${JSON.stringify(outputTensor.dims)}`);
    }

    // Time the GPU->CPU readback (.data access triggers download for WebGPU tensors)
    const dataReadStartAt = getWorkerNow();
    const data = outputTensor.data;
    const dataReadEndAt = getWorkerNow();

    if (!data || typeof data.length !== 'number') {
        throw new Error('Worker ONNX output tensor has no readable data buffer');
    }

    const pixelCount = outWidth * outHeight;
    const channelStride = pixelCount;
    const base = batchIndex * channels * pixelCount;
    const rgba = new Uint8ClampedArray(pixelCount * 4);

    // Cache output scale on first tile — avoids 2048-sample scan on every subsequent tile
    if (!workerCachedOutputScale) {
        workerCachedOutputScale = inferWorkerOutputScale(data);
    }
    const valueScale = workerCachedOutputScale;

    const rBase = base;
    const gBase = base + channelStride;
    const bBase = base + channelStride * 2;

    const pixelLoopStartAt = getWorkerNow();
    // Pre-fill alpha; three sequential channel passes keep reads contiguous in NCHW memory
    // (vs interleaved reads that jump channelStride elements per pixel for large tiles)
    rgba.fill(255);
    for (let i = 0; i < pixelCount; i++) {
        const v = (data[rBase + i] * valueScale + 0.5) | 0;
        rgba[i * 4] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
    for (let i = 0; i < pixelCount; i++) {
        const v = (data[gBase + i] * valueScale + 0.5) | 0;
        rgba[i * 4 + 1] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
    for (let i = 0; i < pixelCount; i++) {
        const v = (data[bBase + i] * valueScale + 0.5) | 0;
        rgba[i * 4 + 2] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
    const pixelLoopEndAt = getWorkerNow();

    return {
        outputWidth: outWidth,
        outputHeight: outHeight,
        outputDims: [1, channels, outHeight, outWidth],
        rgbaBuffer: rgba.buffer,
        dataReadMs: formatWorkerDurationMs(dataReadStartAt, dataReadEndAt),
        pixelLoopMs: formatWorkerDurationMs(pixelLoopStartAt, pixelLoopEndAt)
    };
}

// Reusable single-element views for f32→u32 bit reinterpretation — avoids O(n) allocations in the hot loop.
const _f32TempBuf = new Float32Array(1);
const _u32TempBuf = new Uint32Array(_f32TempBuf.buffer);

function float32ToFloat16Array(float32Array) {
    const len = float32Array.length;
    const out = new Uint16Array(len);

    for (let i = 0; i < len; i++) {
        const val = float32Array[i];

        if (Object.is(val, NaN)) {
            out[i] = 0x7E00; // Quiet NaN
            continue;
        }
        if (val === Infinity) {
            out[i] = 0x7C00;
            continue;
        }
        if (val === -Infinity) {
            out[i] = 0xFC00;
            continue;
        }

        // Reinterpret f32 bits via shared buffer — no per-element heap allocations.
        _f32TempBuf[0] = val;
        const f32Bits = _u32TempBuf[0];
        const sign = (f32Bits >> 16) & 0x8000;
        let exponent = ((f32Bits >> 23) & 0xFF) - 127;
        let mantissa = f32Bits & 0x007FFFFF;

        if (exponent === 128) { // Overflow or NaN
            out[i] = sign | 0x7C00 | (mantissa ? 0x0200 : 0);
        } else if (exponent > 15) { // Overflow to Infinity
            out[i] = sign | 0x7C00;
        } else if (exponent > -15) { // Normal numbers
            out[i] = sign | ((exponent + 15) << 10) | (mantissa >> 13);
        } else if (exponent >= -24) { // Subnormal numbers
            out[i] = sign | ((mantissa | 0x00800000) >> (-14 - exponent));
        } else { // Underflow to Zero
            out[i] = sign;
        }
    }
    return out;
}

async function runWorkerInferenceBatch(batchEntries) {
    if (!Array.isArray(batchEntries) || batchEntries.length === 0) {
        return;
    }

    const entry = batchEntries[0];
    const payload = entry.payload;
    const runId = entry.runId;
    const runStartedAt = getWorkerNow();
    const session = await ensureWorkerSession(payload.init);
    const width = payload.width;
    const height = payload.height;
    const inputName = session.inputNames?.[0];

    if (!inputName) {
        throw new Error('Worker ONNX model has no input name');
    }

    const tensorStartAt = getWorkerNow();
    let inputTensor = payload.imageBitmap != null
        ? await imageBitmapToTensor(payload.imageBitmap)
        : await imageBufferToTensor(payload.rgbaBuffer, width, height);

    // If this session previously required float16 input, convert eagerly before the run.
    if (session._inferredInputType === 'float16' && inputTensor.type === 'float32') {
        const prevTensor = inputTensor;
        inputTensor = new workerOrt.Tensor('float16', float32ToFloat16Array(prevTensor.data), prevTensor.dims);
        if (typeof prevTensor.dispose === 'function') {
            prevTensor.dispose();
        }
    }
    const tensorEndAt = getWorkerNow();

    const runWarningTimer = self.setTimeout(() => {
        postWorkerLog('onnx:worker-pass-still-running', {
            runId,
            inputName,
            inputDims: inputTensor.dims,
            elapsedMs: formatWorkerDurationMs(runStartedAt)
        });
    }, WORKER_RUN_WARNING_MS);

    const inferenceStartAt = getWorkerNow();
    let gpuFirstRunMs = null;
    let gpuRetryRunMs = 0;
    workerActiveGpuProfileCollector = workerOnnxProfilingEnabled ? createWorkerGpuProfileCollector() : null;
    let outputs;
    try {
        try {
            const firstRunStartAt = getWorkerNow();
            outputs = await session.run({ [inputName]: inputTensor });
            const firstRunEndAt = getWorkerNow();
            gpuFirstRunMs = formatWorkerDurationMs(firstRunStartAt, firstRunEndAt);
        } catch (runError) {
            // Auto-detect float16 input requirement. ORT-Web has no public API for
            // inspecting session input metadata, so we catch the type-mismatch error on
            // the first run and cache the result on the session object for all subsequent
            // runs (avoiding the retry overhead after the first tile).
            const errMsg = String(runError?.message || runError);
            if (errMsg.includes('float16') && inputTensor.type === 'float32') {
                console.warn('[Manga Scaler] float16 input required — converting and retrying', { errMsg });
                session._inferredInputType = 'float16';
                const prevTensor = inputTensor;
                inputTensor = new workerOrt.Tensor('float16', float32ToFloat16Array(prevTensor.data), prevTensor.dims);
                if (typeof prevTensor.dispose === 'function') {
                    prevTensor.dispose();
                }
                const retryRunStartAt = getWorkerNow();
                outputs = await session.run({ [inputName]: inputTensor });
                const retryRunEndAt = getWorkerNow();
                gpuRetryRunMs = formatWorkerDurationMs(retryRunStartAt, retryRunEndAt);
                if (gpuFirstRunMs === null) {
                    gpuFirstRunMs = 0;
                }
            } else {
                throw runError;
            }
        }
    } finally {
        self.clearTimeout(runWarningTimer);
        if (typeof inputTensor.dispose === 'function') {
            inputTensor.dispose();
        }
    }
    const inferenceEndAt = getWorkerNow();
    const gpuTotalMs = formatWorkerDurationMs(inferenceStartAt, inferenceEndAt);
    const gpuProfileSummary = finalizeWorkerGpuProfileSummary(workerActiveGpuProfileCollector);
    workerActiveGpuProfileCollector = null;

    postWorkerLog('onnx:worker-pass-inference-complete', {
        runId,
        inputName,
        inferenceMs: gpuTotalMs,
        gpuFirstRunMs,
        gpuRetryRunMs,
        gpuTotalMs
    });

    const outputConvertStartAt = getWorkerNow();

    const outputTensor = resolveWorkerOutputTensor(outputs, session.outputNames || []);
    let outputImage;
    try {
        outputImage = tensorToImageBuffer(outputTensor, 0);
    } finally {
        if (typeof outputTensor.dispose === 'function') {
            outputTensor.dispose();
        }
    }
    const outputConvertEndAt = getWorkerNow();
    const outputConvertMs = formatWorkerDurationMs(outputConvertStartAt, outputConvertEndAt);

    entry.resolve({
        inputName,
        inputDims: [1, 3, height, width],
        outputDims: outputImage.outputDims,
        outputWidth: outputImage.outputWidth,
        outputHeight: outputImage.outputHeight,
        rgbaBuffer: outputImage.rgbaBuffer,
        provider: workerProvider,
        workerTensorBuildMs: formatWorkerDurationMs(tensorStartAt, tensorEndAt),
        workerInferenceMs: gpuTotalMs,
        workerGpuFirstRunMs: gpuFirstRunMs,
        workerGpuRetryRunMs: gpuRetryRunMs,
        workerGpuTotalMs: gpuTotalMs,
        workerGpuKernelCount: gpuProfileSummary?.kernelCount ?? null,
        workerGpuKernelAvgMs: gpuProfileSummary?.avgMs ?? null,
        workerGpuKernelTotalMs: gpuProfileSummary ? Number(gpuProfileSummary.totalMs.toFixed(3)) : null,
        workerGpuKernelMaxMs: gpuProfileSummary ? Number(gpuProfileSummary.maxMs.toFixed(3)) : null,
        workerGpuKernelTop: gpuProfileSummary?.topEvents ?? null,
        workerGpuKernelHotspots: gpuProfileSummary?.topHotspots ?? null,
        workerGpuKernelSlowestHotspots: gpuProfileSummary?.slowestHotspots ?? null,
        workerGpuKernelFirstEvent: gpuProfileSummary?.firstEvent ?? null,
        workerGpuKernelLastEvent: gpuProfileSummary?.lastEvent ?? null,
        workerOutputConvertMs: outputConvertMs,
        workerDataReadMs: outputImage.dataReadMs,
        workerPixelLoopMs: outputImage.pixelLoopMs
    });
}

async function ensureWorkerSession(payload) {
    if (workerSession) {
        return workerSession;
    }

    if (workerInitPromise) {
        return workerInitPromise;
    }

    workerInitPromise = (async () => {
        const startedAt = getWorkerNow();
        const lib = loadOrtRuntime(payload.ortScriptUrl);
        workerOnnxProfilingEnabled = !!payload.onnxProfilingEnabled;
        configureWorkerEnvironment(lib, payload.ortDistUrl);

        if (payload.ortWasmModuleUrl && payload.ortWasmBinaryUrl) {
            lib.env.wasm.wasmPaths = {
                mjs: payload.ortWasmModuleUrl,
                wasm: payload.ortWasmBinaryUrl
            };
        }

        postWorkerLog('onnx:worker-init-start', {
            modelPath: payload.modelPath,
            modelUrl: payload.modelUrl,
            externalDataUrl: payload.externalDataUrl,
            hasNavigatorGpu: !!navigator?.gpu,
            executionProvider: 'webgpu'
        });

        const sessionOptions = {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: false,
            enableMemPattern: false
        };

        if (payload.externalDataBytes && Array.isArray(payload.externalDataPathAliases) && payload.externalDataPathAliases.length > 0) {
            const externalDataView = new Uint8Array(payload.externalDataBytes);
            sessionOptions.externalData = payload.externalDataPathAliases.map((path) => ({
                path,
                data: externalDataView
            }));
        }

        workerSession = await lib.InferenceSession.create(new Uint8Array(payload.modelBytes), sessionOptions);
        payload.modelBytes = null;
        payload.externalDataBytes = null;
        workerProvider = 'webgpu-worker';

        // Warmup: run a tiny inference to pre-compile all WebGPU shaders for this model.
        // This moves shader compilation cost from the first real tile to session init time.
        const warmupStartedAt = getWorkerNow();
        try {
            const warmupInputName = workerSession.inputNames?.[0];
            if (warmupInputName) {
                const wW = 32, wH = 32;
                const warmupTensor = new workerOrt.Tensor('float32', new Float32Array(wW * wH * 3), [1, 3, wH, wW]);
                const warmupOutputs = await workerSession.run({ [warmupInputName]: warmupTensor });
                if (typeof warmupTensor.dispose === 'function') warmupTensor.dispose();
                for (const t of Object.values(warmupOutputs || {})) {
                    if (typeof t?.dispose === 'function') t.dispose();
                }
            }
            postWorkerLog('onnx:worker-warmup', { durationMs: formatWorkerDurationMs(warmupStartedAt) });
        } catch (warmupErr) {
            postWorkerLog('onnx:worker-warmup-failed', { reason: String(warmupErr) });
        }

        postWorkerLog('onnx:worker-init-success', {
            provider: workerProvider,
            durationMs: formatWorkerDurationMs(startedAt),
            inputNames: Array.isArray(workerSession?.inputNames) ? workerSession.inputNames : [],
            outputNames: Array.isArray(workerSession?.outputNames) ? workerSession.outputNames : []
        });

        return workerSession;
    })();

    try {
        return await workerInitPromise;
    } finally {
        workerInitPromise = null;
    }
}

function runWorkerInference(payload) {
    return new Promise((resolve, reject) => {
        runWorkerInferenceBatch([{ runId: ++workerRunCounter, payload, resolve, reject }]).catch(reject);
    });
}

self.onmessage = async (event) => {
    const message = event?.data;
    const response = {
        type: 'response',
        id: message?.id,
        ok: false
    };

    try {
        if (!message || typeof message !== 'object') {
            throw new Error('Worker received an invalid message');
        }

        if (message.type === 'init') {
            const session = await ensureWorkerSession(message.payload);
            response.ok = true;
            response.payload = {
                provider: workerProvider,
                inputNames: Array.isArray(session?.inputNames) ? session.inputNames : [],
                outputNames: Array.isArray(session?.outputNames) ? session.outputNames : []
            };
            self.postMessage(response);
            return;
        }

        if (message.type === 'run') {
            const result = await runWorkerInference(message.payload);
            response.ok = true;
            response.payload = result;
            self.postMessage(response, [result.rgbaBuffer]);
            return;
        }

        throw new Error(`Unsupported worker message type: ${String(message.type)}`);
    } catch (error) {
        response.ok = false;
        response.error = serializeWorkerError(error);
        self.postMessage(response);
    }
};