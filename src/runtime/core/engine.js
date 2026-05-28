// Dispatcher — adapter implementations are in src/runtime/engines/

const REQUIRED_ADAPTER_METHODS = ['isSupported', 'upscale', 'prewarm', 'reset'];
const DEFAULT_BACKGROUND_QUEUE_MAX_CONCURRENCY = 1;

function getDefaultBackgroundExecutionLane(slotIndex) {
    const normalizedSlot = Number.isFinite(slotIndex) ? Math.max(0, Math.floor(slotIndex)) : 0;
    return normalizedSlot <= 0 ? 'background' : `background-${normalizedSlot}`;
}

function normalizeUpscaleResult(result) {
    if (result && typeof result === 'object' && !Array.isArray(result)) {
        return {
            model: typeof result.model === 'string' ? result.model : 'unknown',
            runMode: typeof result.runMode === 'string' ? result.runMode : null
        };
    }

    return {
        model: typeof result === 'string' ? result : 'unknown',
        runMode: null
    };
}

function getValidatedAdapter(adapterName) {
    const adapter = window[adapterName];
    if (!adapter || typeof adapter !== 'object') {
        throw new Error(`${adapterName} is missing from window`);
    }

    for (const methodName of REQUIRED_ADAPTER_METHODS) {
        if (typeof adapter[methodName] !== 'function') {
            throw new Error(`${adapterName}.${methodName} is not a function`);
        }
    }

    return adapter;
}

function tryGetValidatedAdapter(adapterName) {
    try {
        return getValidatedAdapter(adapterName);
    } catch (error) {
        runtimeLog('adapter:invalid', { adapterName, error: String(error) });
        return null;
    }
}

function getEffectiveBackend(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    if (settings.selectedEngineBackend === 'off') return 'off';
    const onnxAdapter = tryGetValidatedAdapter('OnnxRuntimeAdapter');
    if (settings.selectedEngineBackend === 'onnx' && onnxAdapter && onnxAdapter.isSupported()) {
        return 'onnx';
    }
    const webGpuAdapter = tryGetValidatedAdapter('WebGPUAdapter');
    if (settings.selectedEngineBackend === 'webgpu' && webGpuAdapter && webGpuAdapter.isSupported()) {
        return 'webgpu';
    }
    return 'webgl';
}

async function upscaleWithSelectedBackend(tempImg, canvas, runtimeSettings = getRuntimePreferenceSnapshot(), executionOptions = {}) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const backend = getEffectiveBackend(settings);
    if (backend === 'onnx') {
        try {
            const onnxAdapter = getValidatedAdapter('OnnxRuntimeAdapter');
            const result = await onnxAdapter.upscale(tempImg, canvas, settings, executionOptions);
            const normalized = normalizeUpscaleResult(result);
            return { backend: 'onnx', model: normalized.model, runMode: normalized.runMode };
        } catch (err) {
            runtimeLog('onnx:upscale-error', { error: String(err) });
            throw err;
        }
    }

    if (backend === 'webgpu') {
        try {
            const webGpuAdapter = getValidatedAdapter('WebGPUAdapter');
            const result = await webGpuAdapter.upscale(tempImg, canvas, settings);
            const normalized = normalizeUpscaleResult(result);
            return { backend: 'webgpu', model: normalized.model, runMode: normalized.runMode };
        } catch (err) {
            runtimeLog('webgpu:fallback-to-webgl', { error: String(err) });
        }
    }

    const webGlAdapter = getValidatedAdapter('WebGLAdapter');
    const webGlResult = await webGlAdapter.upscale(tempImg, canvas, settings);
    const normalized = normalizeUpscaleResult(webGlResult);
    return { backend: 'webgl', model: normalized.model, runMode: normalized.runMode };
}

async function prewarmSelectedBackend() {
    const settings = getRuntimePreferenceSnapshot();
    const backend = getEffectiveBackend(settings);
    if (backend === 'off') return;
    if (backend === 'onnx') {
        const onnxAdapter = getValidatedAdapter('OnnxRuntimeAdapter');
        await onnxAdapter.prewarm(settings);
        return;
    }
    if (backend === 'webgpu') {
        const webGpuAdapter = getValidatedAdapter('WebGPUAdapter');
        await webGpuAdapter.prewarm(settings);
        return;
    }
    const webGlAdapter = getValidatedAdapter('WebGLAdapter');
    await webGlAdapter.prewarm(settings);
}

function resetBackendRuntimeState() {
    const onnxAdapter = tryGetValidatedAdapter('OnnxRuntimeAdapter');
    if (onnxAdapter) {
        onnxAdapter.reset();
    }

    const webGlAdapter = tryGetValidatedAdapter('WebGLAdapter');
    if (webGlAdapter) {
        webGlAdapter.reset();
    }

    const webGpuAdapter = tryGetValidatedAdapter('WebGPUAdapter');
    if (webGpuAdapter) {
        webGpuAdapter.reset();
    }
}

function resolveBackgroundQueueExecution(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const backend = getEffectiveBackend(settings);

    const defaultPlan = {
        backend,
        maxConcurrency: DEFAULT_BACKGROUND_QUEUE_MAX_CONCURRENCY,
        getExecutionLane: getDefaultBackgroundExecutionLane,
        dispose() {}
    };

    if (backend !== 'onnx') {
        return defaultPlan;
    }

    const onnxAdapter = tryGetValidatedAdapter('OnnxRuntimeAdapter');
    if (!onnxAdapter) {
        return defaultPlan;
    }

    const adapterConcurrency = typeof onnxAdapter.getBackgroundQueueMaxConcurrency === 'function'
        ? Number(onnxAdapter.getBackgroundQueueMaxConcurrency(settings))
        : DEFAULT_BACKGROUND_QUEUE_MAX_CONCURRENCY;
    const maxConcurrency = Number.isFinite(adapterConcurrency)
        ? Math.max(1, Math.floor(adapterConcurrency))
        : DEFAULT_BACKGROUND_QUEUE_MAX_CONCURRENCY;

    return {
        backend,
        maxConcurrency,
        getExecutionLane(slotIndex) {
            if (typeof onnxAdapter.getBackgroundExecutionLane === 'function') {
                return onnxAdapter.getBackgroundExecutionLane(slotIndex);
            }
            return getDefaultBackgroundExecutionLane(slotIndex);
        },
        dispose() {
            if (typeof onnxAdapter.resetBackgroundWorkers === 'function') {
                onnxAdapter.resetBackgroundWorkers(maxConcurrency);
            }
        }
    };
}
