let workerOrt = null;
let workerSession = null;
let workerInitPromise = null;
let workerProvider = null;
let workerRunCounter = 0;
let workerRunQueueDepth = 0;
let workerRunDrainPromise = null;
let workerRunDrainScheduled = false;
const workerPendingRuns = [];
let workerCanUseTensorFromImage = null;
const WORKER_RUN_WARNING_MS = 5000;
const WORKER_MAX_BUFFERED_RUNS = 8;
const WORKER_DYNAMIC_BATCH_SIZE = 2;
const WORKER_BYTE_TO_UNIT_FLOAT = 0.00392156862745098;

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

function configureWorkerEnvironment(lib, ortDistUrl) {
    if (!lib?.env?.wasm) {
        return;
    }

    lib.env.wasm.wasmPaths = {
        mjs: `${ortDistUrl}ort-wasm-simd-threaded.jsep.mjs`,
        wasm: `${ortDistUrl}ort-wasm-simd-threaded.jsep.wasm`
    };
    lib.env.wasm.numThreads = 1;
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
        if (value > maxValue) {
            maxValue = value;
        }
    }
    return maxValue <= 1.5 ? 255 : 1;
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

function imageBufferBatchToTensor(batchPayloads, width, height) {
    const batchSize = batchPayloads.length;
    if (!Number.isFinite(batchSize) || batchSize < 1) {
        throw new Error('ONNX worker received an invalid batch payload');
    }

    const pixelCount = width * height;
    const channels = 3;
    const perImageStride = channels * pixelCount;
    const chw = new Float32Array(batchSize * perImageStride);

    for (let i = 0; i < batchSize; i++) {
        writeImageBufferToChw(batchPayloads[i].rgbaBuffer, width, height, chw, i * perImageStride);
    }

    return new workerOrt.Tensor('float32', chw, [batchSize, channels, height, width]);
}

function tensorToImageBuffer(outputTensor, batchIndex = 0) {
    if (!outputTensor || !Array.isArray(outputTensor.dims) || outputTensor.dims.length !== 4) {
        throw new Error('Unexpected worker ONNX output tensor shape. Expected rank-4 [N,C,H,W]');
    }

    const [batch, channels, outHeight, outWidth] = outputTensor.dims;
    if (!Number.isFinite(batch) || batch < 1 || channels < 3 || outHeight < 1 || outWidth < 1) {
        throw new Error(`Invalid worker ONNX output dims: ${JSON.stringify(outputTensor.dims)}`);
    }

    if (!Number.isFinite(batchIndex) || batchIndex < 0 || batchIndex >= batch) {
        throw new Error(`Invalid worker ONNX output batch index: ${batchIndex} for batch size ${batch}`);
    }

    if (batch === 1 && batchIndex === 0 && typeof outputTensor?.toImageData === 'function') {
        try {
            const imageData = outputTensor.toImageData({
                tensorLayout: 'NCHW',
                format: 'RGB'
            });

            if (imageData && imageData.data && imageData.width === outWidth && imageData.height === outHeight) {
                return {
                    outputWidth: outWidth,
                    outputHeight: outHeight,
                    outputDims: outputTensor.dims,
                    rgbaBuffer: imageData.data.buffer.slice(
                        imageData.data.byteOffset,
                        imageData.data.byteOffset + imageData.data.byteLength
                    )
                };
            }
        } catch (error) {
            postWorkerLog('onnx:worker-toimagedata-fallback', {
                reason: String(error)
            });
        }
    }

    const data = outputTensor.data;
    if (!data || typeof data.length !== 'number') {
        throw new Error('Worker ONNX output tensor has no readable data buffer');
    }

    const pixelCount = outWidth * outHeight;
    const perBatchStride = channels * pixelCount;
    const channelStride = pixelCount;
    const base = batchIndex * perBatchStride;
    const rgba = new Uint8ClampedArray(pixelCount * 4);
    const valueScale = inferWorkerOutputScale(data);

    for (let i = 0; i < pixelCount; i++) {
        const dst = i * 4;
        rgba[dst] = Math.max(0, Math.min(255, Math.round(data[base + i] * valueScale)));
        rgba[dst + 1] = Math.max(0, Math.min(255, Math.round(data[base + channelStride + i] * valueScale)));
        rgba[dst + 2] = Math.max(0, Math.min(255, Math.round(data[base + channelStride * 2 + i] * valueScale)));
        rgba[dst + 3] = 255;
    }

    return {
        outputWidth: outWidth,
        outputHeight: outHeight,
        outputDims: [1, channels, outHeight, outWidth],
        rgbaBuffer: rgba.buffer
    };
}

function scheduleWorkerRunDrain() {
    if (workerRunDrainScheduled) {
        return;
    }

    workerRunDrainScheduled = true;
    Promise.resolve().then(() => {
        workerRunDrainScheduled = false;
        if (!workerRunDrainPromise) {
            workerRunDrainPromise = drainWorkerRunQueue().finally(() => {
                workerRunDrainPromise = null;
                if (workerPendingRuns.length > 0) {
                    scheduleWorkerRunDrain();
                }
            });
        }
    });
}

function takeNextWorkerBatch() {
    if (workerPendingRuns.length === 0) {
        return [];
    }

    const batch = [workerPendingRuns.shift()];
    const headPayload = batch[0].payload;

    for (let i = 0; i < workerPendingRuns.length && batch.length < WORKER_DYNAMIC_BATCH_SIZE; ) {
        const candidate = workerPendingRuns[i];
        if (candidate.payload.width === headPayload.width && candidate.payload.height === headPayload.height) {
            batch.push(candidate);
            workerPendingRuns.splice(i, 1);
            continue;
        }
        i += 1;
    }

    return batch;
}

async function runWorkerInferenceBatch(batchEntries) {
    if (!Array.isArray(batchEntries) || batchEntries.length === 0) {
        return;
    }

    const headPayload = batchEntries[0].payload;
    const runId = batchEntries[0].runId;
    const batchSize = batchEntries.length;
    const runStartedAt = getWorkerNow();
    const session = await ensureWorkerSession(headPayload.init);
    const width = headPayload.width;
    const height = headPayload.height;
    const inputName = session.inputNames?.[0];

    postWorkerLog('onnx:worker-run-start', {
        runId,
        batchSize,
        width,
        height,
        pixelCount: width * height,
        queueDepth: workerRunQueueDepth
    });

    if (!inputName) {
        throw new Error('Worker ONNX model has no input name');
    }

    const tensorStartAt = getWorkerNow();
    const inputTensor = batchSize > 1
        ? imageBufferBatchToTensor(batchEntries.map((entry) => entry.payload), width, height)
        : await imageBufferToTensor(headPayload.rgbaBuffer, width, height);
    const tensorEndAt = getWorkerNow();

    postWorkerLog('onnx:worker-pass-inference-start', {
        runId,
        batchSize,
        inputName,
        inputDims: inputTensor.dims,
        tensorBuildMs: formatWorkerDurationMs(tensorStartAt, tensorEndAt)
    });

    const runWarningTimer = self.setTimeout(() => {
        postWorkerLog('onnx:worker-pass-still-running', {
            runId,
            batchSize,
            inputName,
            inputDims: inputTensor.dims,
            elapsedMs: formatWorkerDurationMs(runStartedAt)
        });
    }, WORKER_RUN_WARNING_MS);

    const inferenceStartAt = getWorkerNow();
    let outputs;
    try {
        outputs = await session.run({ [inputName]: inputTensor });
    } finally {
        self.clearTimeout(runWarningTimer);
    }
    const inferenceEndAt = getWorkerNow();

    postWorkerLog('onnx:worker-pass-inference-complete', {
        runId,
        batchSize,
        inputName,
        inferenceMs: formatWorkerDurationMs(inferenceStartAt, inferenceEndAt)
    });

    const outputResolveStartAt = getWorkerNow();
    const outputTensor = resolveWorkerOutputTensor(outputs, session.outputNames || []);
    const outputResolveEndAt = getWorkerNow();

    postWorkerLog('onnx:worker-output-resolved', {
        runId,
        batchSize,
        outputDims: outputTensor?.dims || null,
        resolveMs: formatWorkerDurationMs(outputResolveStartAt, outputResolveEndAt)
    });

    const outputBatchSize = Array.isArray(outputTensor?.dims) ? Number(outputTensor.dims[0]) : 0;
    if (!Number.isFinite(outputBatchSize) || outputBatchSize < batchSize) {
        throw new Error(`Worker ONNX output batch size mismatch. Expected at least ${batchSize}, received ${outputBatchSize}`);
    }

    for (let i = 0; i < batchEntries.length; i++) {
        const outputImage = tensorToImageBuffer(outputTensor, i);
        batchEntries[i].resolve({
            inputName,
            inputDims: [1, 3, height, width],
            outputDims: outputImage.outputDims,
            outputWidth: outputImage.outputWidth,
            outputHeight: outputImage.outputHeight,
            rgbaBuffer: outputImage.rgbaBuffer,
            provider: workerProvider
        });
    }

    postWorkerLog('onnx:worker-pass-success', {
        runId,
        batchSize,
        inputName,
        inputDims: inputTensor.dims,
        outputDims: outputTensor?.dims || null,
        inferenceMs: formatWorkerDurationMs(inferenceStartAt, inferenceEndAt),
        totalRunMs: formatWorkerDurationMs(runStartedAt)
    });
}

async function drainWorkerRunQueue() {
    while (workerPendingRuns.length > 0) {
        const batchEntries = takeNextWorkerBatch();
        if (batchEntries.length === 0) {
            return;
        }

        try {
            await runWorkerInferenceBatch(batchEntries);
        } catch (error) {
            for (const entry of batchEntries) {
                entry.reject(error);
            }
        } finally {
            workerRunQueueDepth = Math.max(0, workerRunQueueDepth - batchEntries.length);
            postWorkerLog('onnx:worker-run-finished', {
                runId: batchEntries[0]?.runId || null,
                batchSize: batchEntries.length,
                remainingQueueDepth: workerRunQueueDepth
            });
        }
    }
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
            graphOptimizationLevel: 'all'
        };

        if (payload.externalDataBytes && Array.isArray(payload.externalDataPathAliases) && payload.externalDataPathAliases.length > 0) {
            sessionOptions.externalData = payload.externalDataPathAliases.map((path) => ({
                path,
                data: new Uint8Array(payload.externalDataBytes)
            }));
        }

        workerSession = await lib.InferenceSession.create(new Uint8Array(payload.modelBytes), sessionOptions);
        workerProvider = 'webgpu-worker';

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

async function runWorkerInference(payload) {
    return new Promise((resolve, reject) => {
        const runId = ++workerRunCounter;
        const queueDepthAfterEnqueue = workerRunQueueDepth + 1;

        workerPendingRuns.push({
            runId,
            payload,
            resolve,
            reject
        });
        workerRunQueueDepth = queueDepthAfterEnqueue;

        postWorkerLog('onnx:worker-run-queued', {
            runId,
            width: payload.width,
            height: payload.height,
            pixelCount: payload.width * payload.height,
            queueDepth: queueDepthAfterEnqueue,
            maxBatchSize: WORKER_DYNAMIC_BATCH_SIZE
        });

        scheduleWorkerRunDrain();
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

        postWorkerLog('onnx:worker-message-received', {
            id: message.id || null,
            type: message.type || null,
            queueDepth: workerRunQueueDepth
        });

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
            if (workerRunQueueDepth >= WORKER_MAX_BUFFERED_RUNS) {
                const backpressureError = new Error('ONNX worker queue is saturated; rejecting run to prevent memory growth');
                backpressureError.name = 'OnnxWorkerBackpressureError';

                postWorkerLog('onnx:worker-run-rejected-backpressure', {
                    id: message.id || null,
                    queueDepth: workerRunQueueDepth,
                    maxBufferedRuns: WORKER_MAX_BUFFERED_RUNS
                });

                throw backpressureError;
            }

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