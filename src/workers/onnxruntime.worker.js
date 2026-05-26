let workerOrt = null;
let workerSession = null;
let workerInitPromise = null;
let workerProvider = null;
let workerRunCounter = 0;
let workerRunQueueDepth = 0;
let workerRunSerial = Promise.resolve();
const WORKER_RUN_WARNING_MS = 5000;

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

function imageBufferToTensor(rgbaBuffer, width, height) {
    if (!(rgbaBuffer instanceof ArrayBuffer)) {
        throw new Error('ONNX worker received an invalid input pixel buffer');
    }

    const rgba = new Uint8ClampedArray(rgbaBuffer);
    const pixelCount = width * height;
    const chw = new Float32Array(pixelCount * 3);

    for (let i = 0; i < pixelCount; i++) {
        const srcOffset = i * 4;
        chw[i] = rgba[srcOffset] / 255;
        chw[pixelCount + i] = rgba[srcOffset + 1] / 255;
        chw[pixelCount * 2 + i] = rgba[srcOffset + 2] / 255;
    }

    return new workerOrt.Tensor('float32', chw, [1, 3, height, width]);
}

function tensorToImageBuffer(outputTensor) {
    if (!outputTensor || !Array.isArray(outputTensor.dims) || outputTensor.dims.length !== 4) {
        throw new Error('Unexpected worker ONNX output tensor shape. Expected rank-4 [N,C,H,W]');
    }

    const [batch, channels, outHeight, outWidth] = outputTensor.dims;
    if (!Number.isFinite(batch) || batch < 1 || channels < 3 || outHeight < 1 || outWidth < 1) {
        throw new Error(`Invalid worker ONNX output dims: ${JSON.stringify(outputTensor.dims)}`);
    }

    const data = outputTensor.data;
    if (!data || typeof data.length !== 'number') {
        throw new Error('Worker ONNX output tensor has no readable data buffer');
    }

    const pixelCount = outWidth * outHeight;
    const channelStride = pixelCount;
    const batchStride = channels * channelStride;
    const base = 0 * batchStride;
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
        outputDims: outputTensor.dims,
        rgbaBuffer: rgba.buffer
    };
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
            graphOptimizationLevel: 'all',
                extra: {
                    session: {
                        'session.disable_cpu_ep_fallback': '1'
                    }
                },
            externalData: payload.externalDataPathAliases.map((path) => ({
                path,
                data: new Uint8Array(payload.externalDataBytes)
            }))
        };

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
    const runId = ++workerRunCounter;
    const queueDepthAfterEnqueue = ++workerRunQueueDepth;

    postWorkerLog('onnx:worker-run-queued', {
        runId,
        width: payload.width,
        height: payload.height,
        pixelCount: payload.width * payload.height,
        queueDepth: queueDepthAfterEnqueue
    });

    const scheduledRun = workerRunSerial.then(async () => {
        const runStartedAt = getWorkerNow();
        const session = await ensureWorkerSession(payload.init);
        const width = payload.width;
        const height = payload.height;
        const inputName = session.inputNames?.[0];

        postWorkerLog('onnx:worker-run-start', {
            runId,
            width,
            height,
            pixelCount: width * height,
            queueDepth: workerRunQueueDepth
        });

        if (!inputName) {
            throw new Error('Worker ONNX model has no input name');
        }

        const tensorStartAt = getWorkerNow();
        const inputTensor = imageBufferToTensor(payload.rgbaBuffer, width, height);
        const tensorEndAt = getWorkerNow();

        postWorkerLog('onnx:worker-pass-inference-start', {
            runId,
            inputName,
            inputDims: inputTensor.dims,
            tensorBuildMs: formatWorkerDurationMs(tensorStartAt, tensorEndAt)
        });

        const runWarningTimer = self.setTimeout(() => {
            postWorkerLog('onnx:worker-pass-still-running', {
                runId,
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
            inputName,
            inferenceMs: formatWorkerDurationMs(inferenceStartAt, inferenceEndAt)
        });

        const outputResolveStartAt = getWorkerNow();
        const outputTensor = resolveWorkerOutputTensor(outputs, session.outputNames || []);
        const outputResolveEndAt = getWorkerNow();

        postWorkerLog('onnx:worker-output-resolved', {
            runId,
            outputDims: outputTensor?.dims || null,
            resolveMs: formatWorkerDurationMs(outputResolveStartAt, outputResolveEndAt)
        });

        const outputImageStartAt = getWorkerNow();
        const outputImage = tensorToImageBuffer(outputTensor);
        const outputImageEndAt = getWorkerNow();

        postWorkerLog('onnx:worker-output-buffer-ready', {
            runId,
            outputDims: outputImage.outputDims,
            outputWidth: outputImage.outputWidth,
            outputHeight: outputImage.outputHeight,
            outputBufferMs: formatWorkerDurationMs(outputImageStartAt, outputImageEndAt)
        });

        postWorkerLog('onnx:worker-pass-success', {
            runId,
            inputName,
            inputDims: inputTensor.dims,
            outputDims: outputImage.outputDims,
            inferenceMs: formatWorkerDurationMs(inferenceStartAt, inferenceEndAt),
            totalRunMs: formatWorkerDurationMs(runStartedAt)
        });

        return {
            inputName,
            inputDims: inputTensor.dims,
            outputDims: outputImage.outputDims,
            outputWidth: outputImage.outputWidth,
            outputHeight: outputImage.outputHeight,
            rgbaBuffer: outputImage.rgbaBuffer,
            provider: workerProvider
        };
    });

    workerRunSerial = scheduledRun.catch(() => {});

    try {
        return await scheduledRun;
    } finally {
        workerRunQueueDepth = Math.max(0, workerRunQueueDepth - 1);
        postWorkerLog('onnx:worker-run-finished', {
            runId,
            remainingQueueDepth: workerRunQueueDepth
        });
    }
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