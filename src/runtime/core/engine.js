// Dispatcher — adapter implementations are in src/runtime/engines/

const REQUIRED_ADAPTER_METHODS = ['isSupported', 'upscale', 'prewarm', 'reset'];

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
    const webGpuAdapter = tryGetValidatedAdapter('WebGPUAdapter');
    if (settings.selectedEngineBackend === 'webgpu' && webGpuAdapter && webGpuAdapter.isSupported()) {
        return 'webgpu';
    }
    return 'webgl';
}

async function upscaleWithSelectedBackend(tempImg, canvas, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const backend = getEffectiveBackend(settings);
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
    if (backend === 'webgpu') {
        const webGpuAdapter = getValidatedAdapter('WebGPUAdapter');
        await webGpuAdapter.prewarm(settings);
        return;
    }
    const webGlAdapter = getValidatedAdapter('WebGLAdapter');
    await webGlAdapter.prewarm(settings);
}

function resetBackendRuntimeState() {
    const webGlAdapter = tryGetValidatedAdapter('WebGLAdapter');
    if (webGlAdapter) {
        webGlAdapter.reset();
    }

    const webGpuAdapter = tryGetValidatedAdapter('WebGPUAdapter');
    if (webGpuAdapter) {
        webGpuAdapter.reset();
    }
}
