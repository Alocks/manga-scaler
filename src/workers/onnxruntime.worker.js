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
let workerGraphCaptureEnabled = false;
let workerGraphCaptureActive = false;
let workerGraphCaptureFailureReason = null;
let workerPreferredInputWidth = 0;
let workerPreferredInputHeight = 0;
let workerGpuDevice = null;
let workerGpuInputBuffer = null;
let workerGpuInputBufferByteLength = 0;
let workerGpuInputBufferDims = null;
let workerExpectedInputChannels = 3;
let workerExpectedOutputChannels = 3;

function normalizeWorkerChannelCount(value, fallback = 3) {
    const numeric = Number(value);
    if (numeric === 1 || numeric === 3) {
        return numeric;
    }

    if (fallback == null) {
        return null;
    }

    return Number(fallback) === 1 ? 1 : 3;
}

function resolveWorkerValueInfoMetadata(metadataCollection, tensorName) {
    if (!metadataCollection || !tensorName) {
        return null;
    }

    if (typeof metadataCollection.get === 'function') {
        return metadataCollection.get(tensorName) || null;
    }

    if (typeof metadataCollection === 'object') {
        return metadataCollection[tensorName] || null;
    }

    return null;
}

function resolveWorkerChannelsFromTensorMetadata(valueInfo) {
    const dims = valueInfo?.dimensions || valueInfo?.dims;
    if (Array.isArray(dims) && dims.length >= 2) {
        return normalizeWorkerChannelCount(dims[1], null);
    }

    return null;
}

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

async function loadOrtRuntime(ortScriptUrl) {
    if (workerOrt) {
        return workerOrt;
    }

    if (globalThis.workerOrtPreload && globalThis.workerOrtPreload.InferenceSession && typeof globalThis.workerOrtPreload.Tensor === 'function') {
        workerOrt = globalThis.workerOrtPreload;
        return workerOrt;
    }

    const ortModule = await import(ortScriptUrl);
    workerOrt = ortModule?.default || ortModule?.ort || ortModule;

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
            message.includes('session_state.cc:1369') ||
            message.includes('CleanUnusedInitializersAndNodeArgs')
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
        workerExpectedInputChannels === 3 &&
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
    const chw = new Float32Array(pixelCount * workerExpectedInputChannels);
    writeImageBufferToNchw(rgbaBuffer, width, height, chw, 0, workerExpectedInputChannels);

    return new workerOrt.Tensor('float32', chw, [1, workerExpectedInputChannels, height, width]);
}

async function tensorFromImageSource(imageSource, width, height, sourceLabel = 'image-source') {
    if (workerExpectedInputChannels !== 3) {
        return null;
    }

    if (typeof workerOrt?.Tensor?.fromImage !== 'function') {
        return null;
    }

    try {
        return await workerOrt.Tensor.fromImage(imageSource, {
            tensorLayout: 'NCHW',
            tensorFormat: 'RGB',
            dataType: 'float32',
            resizedWidth: width,
            resizedHeight: height
        });
    } catch (error) {
        postWorkerLog('onnx:worker-fromimage-source-fallback', {
            source: sourceLabel,
            reason: String(error)
        });
        return null;
    }
}

async function buildWorkerCpuInputTensor(payload, width, height) {
    if (payload.imageBitmap != null) {
        return imageBitmapToTensor(payload.imageBitmap);
    }
    return imageBufferToTensor(payload.rgbaBuffer, width, height);
}

function getWorkerImageBitmapRgbaBuffer(imageBitmap, width, height) {
    // Fallback path for graph-capture input upload: extract RGBA on CPU and then write CHW to a GPU buffer.
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(imageBitmap, 0, 0, width, height);
    imageBitmap.close();
    const imageData = ctx.getImageData(0, 0, width, height);
    return imageData.data.buffer;
}

function ensureWorkerGpuInputBuffer(byteLength, dims) {
    if (!workerGpuDevice || typeof workerGpuDevice.createBuffer !== 'function') {
        return null;
    }

    if (
        workerGpuInputBuffer &&
        workerGpuInputBufferByteLength === byteLength &&
        workerGpuInputBufferDims &&
        workerGpuInputBufferDims.width === dims.width &&
        workerGpuInputBufferDims.height === dims.height
    ) {
        return workerGpuInputBuffer;
    }

    if (workerGpuInputBuffer && typeof workerGpuInputBuffer.destroy === 'function') {
        workerGpuInputBuffer.destroy();
    }

    const gpuBufferUsage = self.GPUBufferUsage || {};
    const usageFlags = (gpuBufferUsage.COPY_DST || 0x0008) | (gpuBufferUsage.STORAGE || 0x0080);

    workerGpuInputBuffer = workerGpuDevice.createBuffer({
        size: byteLength,
        usage: usageFlags
    });
    workerGpuInputBufferByteLength = byteLength;
    workerGpuInputBufferDims = { ...dims };
    return workerGpuInputBuffer;
}

async function tryBuildWorkerGpuInputTensor(payload, width, height, expectedTensorType) {
    if (!workerGraphCaptureEnabled || !workerGraphCaptureActive) {
        return null;
    }

    if (expectedTensorType === 'float16') {
        return null;
    }

    if (
        !Number.isFinite(workerPreferredInputWidth) ||
        !Number.isFinite(workerPreferredInputHeight) ||
        workerPreferredInputWidth < 1 ||
        workerPreferredInputHeight < 1
    ) {
        return null;
    }

    if (width !== workerPreferredInputWidth || height !== workerPreferredInputHeight) {
        return null;
    }

    if (typeof workerOrt?.Tensor?.fromGpuBuffer !== 'function') {
        workerGraphCaptureActive = false;
        workerGraphCaptureFailureReason = 'Tensor.fromGpuBuffer is unavailable';
        return null;
    }

    if (!workerGpuDevice || typeof workerGpuDevice?.queue?.writeBuffer !== 'function') {
        workerGraphCaptureActive = false;
        workerGraphCaptureFailureReason = 'WebGPU device queue is unavailable';
        return null;
    }

    const pixelCount = width * height;
    const chw = new Float32Array(pixelCount * workerExpectedInputChannels);
    let rgbaBuffer = payload.rgbaBuffer;

    if (!(rgbaBuffer instanceof ArrayBuffer) && payload.imageBitmap != null) {
        rgbaBuffer = getWorkerImageBitmapRgbaBuffer(payload.imageBitmap, width, height);
        payload.rgbaBuffer = rgbaBuffer;
        payload.imageBitmap = null;
    }

    if (!(rgbaBuffer instanceof ArrayBuffer)) {
        return null;
    }

    writeImageBufferToNchw(rgbaBuffer, width, height, chw, 0, workerExpectedInputChannels);
    const byteLength = chw.byteLength;
    const gpuInputBuffer = ensureWorkerGpuInputBuffer(byteLength, { width, height });
    if (!gpuInputBuffer) {
        workerGraphCaptureActive = false;
        workerGraphCaptureFailureReason = 'Failed to allocate reusable WebGPU input buffer';
        return null;
    }

    workerGpuDevice.queue.writeBuffer(gpuInputBuffer, 0, chw.buffer, chw.byteOffset, byteLength);

    return workerOrt.Tensor.fromGpuBuffer(gpuInputBuffer, {
        dims: [1, workerExpectedInputChannels, height, width],
        dataType: 'float32'
    });
}

async function imageBitmapToTensor(imageBitmap) {
    // Preferred path: hand the ImageBitmap directly to ORT so WebGPU can import texture data.
    if (workerCanUseBitmapFromImage !== false) {
        const directTensor = await tensorFromImageSource(
            imageBitmap,
            imageBitmap.width,
            imageBitmap.height,
            'imageBitmap'
        );
        if (directTensor) {
            workerCanUseBitmapFromImage = true;
            imageBitmap.close();
            return directTensor;
        }

        // Do not permanently disable after a single failure; runtime support can vary by source and frame.
        workerCanUseBitmapFromImage = null;
    }

    // Secondary fast path: draw once to OffscreenCanvas and pass canvas directly to Tensor.fromImage.
    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(imageBitmap, 0, 0);

    const canvasTensor = await tensorFromImageSource(
        canvas,
        canvas.width,
        canvas.height,
        'offscreenCanvas'
    );
    if (canvasTensor) {
        imageBitmap.close();
        return canvasTensor;
    }

    // Last resort: CPU readback + conversion path.
    imageBitmap.close();
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return imageBufferToTensor(imageData.data.buffer, canvas.width, canvas.height);
}

function writeImageBufferToNchw(rgbaBuffer, width, height, targetChw, targetOffset = 0, channels = 3) {
    if (!(rgbaBuffer instanceof ArrayBuffer)) {
        throw new Error('ONNX worker received an invalid input pixel buffer');
    }

    const rgba = new Uint8Array(rgbaBuffer);
    const pixelCount = width * height;

    if (channels === 1) {
        for (let i = 0; i < pixelCount; i++) {
            const srcOffset = i << 2;
            const r = rgba[srcOffset];
            const g = rgba[srcOffset + 1];
            const b = rgba[srcOffset + 2];
            // BT.601 luma preserves detail better than simple average for monochrome models.
            const luma = (0.299 * r + 0.587 * g + 0.114 * b) * WORKER_BYTE_TO_UNIT_FLOAT;
            targetChw[targetOffset + i] = luma;
        }
        return;
    }

    if (channels !== 3) {
        throw new Error(`Unsupported worker ONNX input channel count: ${channels}`);
    }

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
    if (!Number.isFinite(batch) || batch < 1 || (channels !== 1 && channels !== 3) || outHeight < 1 || outWidth < 1) {
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
    if (channels === 1) {
        for (let i = 0; i < pixelCount; i++) {
            const v = (data[rBase + i] * valueScale + 0.5) | 0;
            const clamped = v < 0 ? 0 : v > 255 ? 255 : v;
            const rgbaIndex = i * 4;
            rgba[rgbaIndex] = clamped;
            rgba[rgbaIndex + 1] = clamped;
            rgba[rgbaIndex + 2] = clamped;
        }
    } else {
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
    let inputTensor = await tryBuildWorkerGpuInputTensor(payload, width, height, session._inferredInputType || 'float32');
    if (!inputTensor) {
        inputTensor = await buildWorkerCpuInputTensor(payload, width, height);
    }

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

                // Graph capture in ORT-Web requires stable tensor creation paths and shapes.
                // If the model expects float16 inputs, fall back to CPU tensor creation.
                if (workerGraphCaptureActive) {
                    workerGraphCaptureActive = false;
                    workerGraphCaptureFailureReason = 'Model requires float16 input tensor';
                }

                const prevTensor = inputTensor;
                const canReusePrevData = prevTensor && prevTensor.data && prevTensor.dims;
                if (canReusePrevData) {
                    inputTensor = new workerOrt.Tensor('float16', float32ToFloat16Array(prevTensor.data), prevTensor.dims);
                } else {
                    const cpuTensor = await buildWorkerCpuInputTensor(payload, width, height);
                    inputTensor = new workerOrt.Tensor('float16', float32ToFloat16Array(cpuTensor.data), cpuTensor.dims);
                    if (typeof cpuTensor.dispose === 'function') {
                        cpuTensor.dispose();
                    }
                }
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
        inputDims: [1, workerExpectedInputChannels, height, width],
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
        const lib = await loadOrtRuntime(payload.ortScriptUrl);
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

        workerPreferredInputWidth = Number.isFinite(payload.preferredInputWidth)
            ? Math.max(1, Math.floor(payload.preferredInputWidth))
            : 0;
        workerPreferredInputHeight = Number.isFinite(payload.preferredInputHeight)
            ? Math.max(1, Math.floor(payload.preferredInputHeight))
            : 0;
        workerGraphCaptureEnabled = payload.graphCaptureEnabled !== false;
        workerGraphCaptureActive = workerGraphCaptureEnabled;
        workerGraphCaptureFailureReason = null;
        workerGpuDevice = null;
        workerGpuInputBuffer = null;
        workerGpuInputBufferByteLength = 0;
        workerGpuInputBufferDims = null;
        workerExpectedInputChannels = normalizeWorkerChannelCount(payload.inputChannels, 3);
        workerExpectedOutputChannels = normalizeWorkerChannelCount(payload.outputChannels, 3);

        const sessionOptions = {
            executionProviders: [{
                name: 'webgpu',
                preferredLayout: 'NHWC',
                powerPreference: 'high-performance'
            }],
            graphOptimizationLevel: 'all',
            enableGraphCapture: workerGraphCaptureEnabled,
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

        try {
            workerSession = await lib.InferenceSession.create(new Uint8Array(payload.modelBytes), sessionOptions);
        } catch (sessionCreateError) {
            postWorkerLog('onnx:worker-session-create-retry', {
                reason: String(sessionCreateError),
                graphCaptureEnabled: workerGraphCaptureEnabled
            });

            workerGraphCaptureActive = false;
            workerGraphCaptureFailureReason = String(sessionCreateError);

            const fallbackSessionOptions = {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all',
                enableCpuMemArena: false,
                enableMemPattern: false
            };
            if (sessionOptions.externalData) {
                fallbackSessionOptions.externalData = sessionOptions.externalData;
            }
            workerSession = await lib.InferenceSession.create(new Uint8Array(payload.modelBytes), fallbackSessionOptions);
        }

        const sessionInputName = workerSession.inputNames?.[0] || null;
        const sessionOutputName = workerSession.outputNames?.[0] || null;
        const inputMetadata = resolveWorkerValueInfoMetadata(workerSession.inputMetadata, sessionInputName);
        const outputMetadata = resolveWorkerValueInfoMetadata(workerSession.outputMetadata, sessionOutputName);
        workerExpectedInputChannels = normalizeWorkerChannelCount(
            resolveWorkerChannelsFromTensorMetadata(inputMetadata),
            workerExpectedInputChannels
        );
        workerExpectedOutputChannels = normalizeWorkerChannelCount(
            resolveWorkerChannelsFromTensorMetadata(outputMetadata),
            workerExpectedOutputChannels
        );

        if (workerExpectedInputChannels !== 1 && workerExpectedInputChannels !== 3) {
            throw new Error(`Unsupported ONNX worker input channels: ${workerExpectedInputChannels}`);
        }

        if (workerExpectedOutputChannels !== 1 && workerExpectedOutputChannels !== 3) {
            throw new Error(`Unsupported ONNX worker output channels: ${workerExpectedOutputChannels}`);
        }

        payload.modelBytes = null;
        payload.externalDataBytes = null;
        workerProvider = 'webgpu-worker';

        if (workerGraphCaptureActive) {
            workerGpuDevice = lib?.env?.webgpu?.device || null;
            if (!workerGpuDevice) {
                workerGraphCaptureActive = false;
                workerGraphCaptureFailureReason = 'ORT WebGPU device handle is unavailable after session init';
            }
        }

        // Warmup: run a model-size inference to pre-compile shader variants used by real tiles.
        const warmupStartedAt = getWorkerNow();
        try {
            const warmupInputName = workerSession.inputNames?.[0];
            if (warmupInputName) {
                const wW = Number.isFinite(payload.warmupWidth) ? Math.max(1, Math.floor(payload.warmupWidth)) : 32;
                const wH = Number.isFinite(payload.warmupHeight) ? Math.max(1, Math.floor(payload.warmupHeight)) : 32;
                const warmupTensor = new workerOrt.Tensor(
                    'float32',
                    new Float32Array(wW * wH * workerExpectedInputChannels),
                    [1, workerExpectedInputChannels, wH, wW]
                );
                const warmupOutputs = await workerSession.run({ [warmupInputName]: warmupTensor });
                if (typeof warmupTensor.dispose === 'function') warmupTensor.dispose();
                for (const t of Object.values(warmupOutputs || {})) {
                    if (typeof t?.dispose === 'function') t.dispose();
                }
            }
            postWorkerLog('onnx:worker-warmup', {
                durationMs: formatWorkerDurationMs(warmupStartedAt),
                warmupWidth: Number.isFinite(payload.warmupWidth) ? payload.warmupWidth : 32,
                warmupHeight: Number.isFinite(payload.warmupHeight) ? payload.warmupHeight : 32,
                inputChannels: workerExpectedInputChannels,
                outputChannels: workerExpectedOutputChannels,
                graphCaptureActive: workerGraphCaptureActive,
                graphCaptureFailureReason: workerGraphCaptureFailureReason
            });
        } catch (warmupErr) {
            postWorkerLog('onnx:worker-warmup-failed', { reason: String(warmupErr) });
        }

        postWorkerLog('onnx:worker-init-success', {
            provider: workerProvider,
            durationMs: formatWorkerDurationMs(startedAt),
            graphCaptureEnabled: workerGraphCaptureEnabled,
            graphCaptureActive: workerGraphCaptureActive,
            graphCaptureFailureReason: workerGraphCaptureFailureReason,
            inputChannels: workerExpectedInputChannels,
            outputChannels: workerExpectedOutputChannels,
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