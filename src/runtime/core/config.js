// Settings constants, preference loaders, and ready promises

const SIMPLE_PRESET_KEY = 'simplePreset';
const ENGINE_BACKEND_KEY = 'engineBackend';
const WEBGPU_MODEL_KEY = 'webgpuModel';
const WEBGPU_SCALE_KEY = 'webgpuScale';
const DEFAULT_SIMPLE_PRESET = 'M';
const DEFAULT_ENGINE_BACKEND = 'webgl';
const DEFAULT_WEBGPU_MODEL = 'ModeA';
const DEFAULT_WEBGPU_SCALE = 2;
const SIMPLE_PRESET_VALUES = new Set(['S', 'M', 'L', 'UL', 'VL']);
const ENGINE_BACKEND_VALUES = new Set(['off', 'webgl', 'webgpu', 'onnx']);
const WEBGPU_MODEL_VALUES = new Set([
    'ModeA', 'ModeAA', 'ModeB', 'ModeBB', 'ModeC', 'ModeCA'
]);
const WEBGPU_SCALE_VALUES = new Set([2, 3, 4]);

let selectedSimplePreset = DEFAULT_SIMPLE_PRESET;
let selectedEngineBackend = DEFAULT_ENGINE_BACKEND;
let selectedWebGpuModel = DEFAULT_WEBGPU_MODEL;
let selectedWebGpuScale = DEFAULT_WEBGPU_SCALE;
let presetReadyPromise = Promise.resolve();
let backendReadyPromise = Promise.resolve();
let webgpuModelReadyPromise = Promise.resolve();
let webgpuScaleReadyPromise = Promise.resolve();
let backendPreferenceLoaded = false;

function runtimeLog(label, data = {}) {
    if (typeof window.MangaScalerLog === 'function') {
        window.MangaScalerLog(label, data);
        return;
    }
    if (!window.MangaScalerDebugEnabled) {
        return;
    }
    console.log('[Manga Scaler]', label, { ts: new Date().toISOString(), ...data });
}

function isRuntimeProfilingEnabled() {
    return !!window.MangaScalerProfiling?.isEnabled?.();
}

function runtimeProfileLog(label, data = {}) {
    if (!isRuntimeProfilingEnabled()) return;
    runtimeLog(`profile:${label}`, data);
}

function normalizeSimplePreset(value) {
    const normalized = String(value || '').toUpperCase();
    return SIMPLE_PRESET_VALUES.has(normalized) ? normalized : DEFAULT_SIMPLE_PRESET;
}

function normalizeEngineBackend(value) {
    const normalized = String(value || '').toLowerCase();
    return ENGINE_BACKEND_VALUES.has(normalized) ? normalized : DEFAULT_ENGINE_BACKEND;
}

function normalizeWebGpuModel(value) {
    const normalized = String(value || '').trim();
    return WEBGPU_MODEL_VALUES.has(normalized) ? normalized : DEFAULT_WEBGPU_MODEL;
}

function normalizeWebGpuScale(value) {
    const normalized = Number(value);
    return WEBGPU_SCALE_VALUES.has(normalized) ? normalized : DEFAULT_WEBGPU_SCALE;
}

function getRuntimePreferenceSnapshot() {
    return {
        selectedSimplePreset,
        selectedEngineBackend,
        selectedWebGpuModel,
        selectedWebGpuScale
    };
}

function getNormalizedRuntimePreferenceSnapshot(snapshot) {
    const source = snapshot && typeof snapshot === 'object' ? snapshot : {};
    return {
        selectedSimplePreset: normalizeSimplePreset(source.selectedSimplePreset),
        selectedEngineBackend: normalizeEngineBackend(source.selectedEngineBackend),
        selectedWebGpuModel: normalizeWebGpuModel(source.selectedWebGpuModel),
        selectedWebGpuScale: normalizeWebGpuScale(source.selectedWebGpuScale)
    };
}

function applyRuntimePreferenceStorageChanges(changes) {
    const hasPresetChange = !!changes[SIMPLE_PRESET_KEY];
    const hasBackendChange = !!changes[ENGINE_BACKEND_KEY];
    const hasWebGpuModelChange = !!changes[WEBGPU_MODEL_KEY];
    const hasWebGpuScaleChange = !!changes[WEBGPU_SCALE_KEY];
    if (!hasPresetChange && !hasBackendChange && !hasWebGpuModelChange && !hasWebGpuScaleChange) {
        return false;
    }

    let didChange = false;

    if (hasPresetChange) {
        const nextPreset = normalizeSimplePreset(changes[SIMPLE_PRESET_KEY].newValue);
        if (nextPreset !== selectedSimplePreset) {
            selectedSimplePreset = nextPreset;
            didChange = true;
        }
    }

    if (hasBackendChange) {
        const nextBackend = normalizeEngineBackend(changes[ENGINE_BACKEND_KEY].newValue);
        if (nextBackend !== selectedEngineBackend) {
            selectedEngineBackend = nextBackend;
            didChange = true;
        }
    }

    if (hasWebGpuModelChange) {
        const nextWebGpuModel = normalizeWebGpuModel(changes[WEBGPU_MODEL_KEY].newValue);
        if (nextWebGpuModel !== selectedWebGpuModel) {
            selectedWebGpuModel = nextWebGpuModel;
            didChange = true;
        }
    }

    if (hasWebGpuScaleChange) {
        const nextWebGpuScale = normalizeWebGpuScale(changes[WEBGPU_SCALE_KEY].newValue);
        if (nextWebGpuScale !== selectedWebGpuScale) {
            selectedWebGpuScale = nextWebGpuScale;
            didChange = true;
        }
    }

    return didChange;
}

function loadSimplePresetPreference() {
    if (!chrome?.storage?.sync) {
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        chrome.storage.sync.get({ [SIMPLE_PRESET_KEY]: DEFAULT_SIMPLE_PRESET }, (result) => {
            selectedSimplePreset = normalizeSimplePreset(result?.[SIMPLE_PRESET_KEY]);
            runtimeLog('preset:loaded', { selectedSimplePreset });
            resolve();
        });
    });
}

function loadEngineBackendPreference() {
    if (!chrome?.storage?.sync) {
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        chrome.storage.sync.get({ [ENGINE_BACKEND_KEY]: DEFAULT_ENGINE_BACKEND }, (result) => {
            selectedEngineBackend = normalizeEngineBackend(result?.[ENGINE_BACKEND_KEY]);
            backendPreferenceLoaded = true;
            runtimeLog('backend:loaded', { selectedEngineBackend });
            resolve();
        });
    });
}

function loadWebGpuModelPreference() {
    if (!chrome?.storage?.sync) {
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        chrome.storage.sync.get({ [WEBGPU_MODEL_KEY]: DEFAULT_WEBGPU_MODEL }, (result) => {
            selectedWebGpuModel = normalizeWebGpuModel(result?.[WEBGPU_MODEL_KEY]);
            runtimeLog('webgpu-model:loaded', { selectedWebGpuModel });
            resolve();
        });
    });
}

function loadWebGpuScalePreference() {
    if (!chrome?.storage?.sync) {
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        chrome.storage.sync.get({ [WEBGPU_SCALE_KEY]: DEFAULT_WEBGPU_SCALE }, (result) => {
            selectedWebGpuScale = normalizeWebGpuScale(result?.[WEBGPU_SCALE_KEY]);
            runtimeLog('webgpu-scale:loaded', { selectedWebGpuScale });
            resolve();
        });
    });
}

presetReadyPromise = loadSimplePresetPreference();
backendReadyPromise = loadEngineBackendPreference();
webgpuModelReadyPromise = loadWebGpuModelPreference();
webgpuScaleReadyPromise = loadWebGpuScalePreference();
