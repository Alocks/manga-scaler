// Settings constants, preference loaders, and ready promises

const SIMPLE_PRESET_KEY = 'simplePreset';
const ENGINE_BACKEND_KEY = 'engineBackend';
const WEBGPU_MODEL_KEY = 'webgpuModel';
const WEBGPU_SCALE_KEY = 'webgpuScale';
const ONNX_MODEL_KEY = 'onnxModel';
const DEFAULT_SIMPLE_PRESET = 'M';
const DEFAULT_ENGINE_BACKEND = 'webgl';
const DEFAULT_WEBGPU_MODEL = 'ModeA';
const DEFAULT_WEBGPU_SCALE = 2;
const DEFAULT_ONNX_MODEL = 'realesr-animevideov3';
const SIMPLE_PRESET_VALUES = new Set(['S', 'M', 'L', 'UL', 'VL']);
const ENGINE_BACKEND_VALUES = new Set(['off', 'webgl', 'webgpu', 'onnx']);
const WEBGPU_MODEL_VALUES = new Set([
    'ModeA', 'ModeAA', 'ModeB', 'ModeBB', 'ModeC', 'ModeCA'
]);
const WEBGPU_SCALE_VALUES = new Set([2, 3, 4]);
const FALLBACK_ONNX_MODEL_VALUES = new Set([
    'realesr-animevideov3',
    'up2x-latest-conservative',
    'up2x-latest-denoise1x',
    'animejanai-hd-v3sharp1-compact',
    'animejanai-hd-v3sharp1-superultracompact',
    'mangajanai-1200p-v1',
    'mangajanai-1600p-v1',
    'illustrationjanai-v1'
]);

let selectedSimplePreset = DEFAULT_SIMPLE_PRESET;
let selectedEngineBackend = DEFAULT_ENGINE_BACKEND;
let selectedWebGpuModel = DEFAULT_WEBGPU_MODEL;
let selectedWebGpuScale = DEFAULT_WEBGPU_SCALE;
let selectedOnnxModel = DEFAULT_ONNX_MODEL;
let presetReadyPromise = Promise.resolve();
let backendReadyPromise = Promise.resolve();
let webgpuModelReadyPromise = Promise.resolve();
let webgpuScaleReadyPromise = Promise.resolve();
let onnxModelReadyPromise = Promise.resolve();
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

function isOnnxProfilingEnabled() {
    return !!window.MangaScalerProfiling?.isOnnxEnabled?.();
}

function runtimeProfileLog(label, data = {}) {
    const normalizedLabel = String(label || '');

    if (normalizedLabel.startsWith('onnx:')) {
        if (!isOnnxProfilingEnabled()) return;
    } else if (!isRuntimeProfilingEnabled()) {
        return;
    }

    runtimeLog(`profile:${normalizedLabel}`, data);
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

function normalizeOnnxModel(value) {
    const normalized = String(value || '').trim();
    if (typeof globalThis.normalizeOnnxModelKey === 'function') {
        return globalThis.normalizeOnnxModelKey(normalized);
    }

    if (typeof globalThis.getOnnxModelOptions === 'function') {
        const options = globalThis.getOnnxModelOptions();
        if (Array.isArray(options) && options.some((option) => option?.key === normalized)) {
            return normalized;
        }
    }

    return FALLBACK_ONNX_MODEL_VALUES.has(normalized) ? normalized : DEFAULT_ONNX_MODEL;
}

function getRuntimePreferenceSnapshot() {
    return {
        selectedSimplePreset,
        selectedEngineBackend,
        selectedWebGpuModel,
        selectedWebGpuScale,
        selectedOnnxModel
    };
}

function getNormalizedRuntimePreferenceSnapshot(snapshot) {
    const source = snapshot && typeof snapshot === 'object' ? snapshot : {};
    return {
        selectedSimplePreset: normalizeSimplePreset(source.selectedSimplePreset),
        selectedEngineBackend: normalizeEngineBackend(source.selectedEngineBackend),
        selectedWebGpuModel: normalizeWebGpuModel(source.selectedWebGpuModel),
        selectedWebGpuScale: normalizeWebGpuScale(source.selectedWebGpuScale),
        selectedOnnxModel: normalizeOnnxModel(source.selectedOnnxModel)
    };
}

function applyNormalizedPreferenceChange(changes, key, normalize, currentValue, setValue) {
    if (!changes[key]) {
        return false;
    }

    const nextValue = normalize(changes[key].newValue);
    if (nextValue === currentValue) {
        return false;
    }

    setValue(nextValue);
    return true;
}

function applyRuntimePreferenceStorageChanges(changes) {
    let didChange = false;

    didChange = applyNormalizedPreferenceChange(
        changes,
        SIMPLE_PRESET_KEY,
        normalizeSimplePreset,
        selectedSimplePreset,
        (value) => { selectedSimplePreset = value; }
    ) || didChange;

    didChange = applyNormalizedPreferenceChange(
        changes,
        ENGINE_BACKEND_KEY,
        normalizeEngineBackend,
        selectedEngineBackend,
        (value) => { selectedEngineBackend = value; }
    ) || didChange;

    didChange = applyNormalizedPreferenceChange(
        changes,
        WEBGPU_MODEL_KEY,
        normalizeWebGpuModel,
        selectedWebGpuModel,
        (value) => { selectedWebGpuModel = value; }
    ) || didChange;

    didChange = applyNormalizedPreferenceChange(
        changes,
        WEBGPU_SCALE_KEY,
        normalizeWebGpuScale,
        selectedWebGpuScale,
        (value) => { selectedWebGpuScale = value; }
    ) || didChange;

    didChange = applyNormalizedPreferenceChange(
        changes,
        ONNX_MODEL_KEY,
        normalizeOnnxModel,
        selectedOnnxModel,
        (value) => { selectedOnnxModel = value; }
    ) || didChange;

    return didChange;
}

function loadNormalizedPreference({ key, defaultValue, normalize, logLabel, afterLoad, assign, logField }) {
    if (!chrome?.storage?.sync) {
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        chrome.storage.sync.get({ [key]: defaultValue }, (result) => {
            const normalized = normalize(result?.[key]);
            assign(normalized);
            if (typeof afterLoad === 'function') {
                afterLoad(normalized);
            }
            runtimeLog(logLabel, { [logField || key]: normalized });
            resolve();
        });
    });
}

function loadSimplePresetPreference() {
    return loadNormalizedPreference({
        key: SIMPLE_PRESET_KEY,
        defaultValue: DEFAULT_SIMPLE_PRESET,
        normalize: normalizeSimplePreset,
        logLabel: 'preset:loaded',
        logField: 'selectedSimplePreset',
        assign: (value) => {
            selectedSimplePreset = value;
        }
    });
}

function loadEngineBackendPreference() {
    return loadNormalizedPreference({
        key: ENGINE_BACKEND_KEY,
        defaultValue: DEFAULT_ENGINE_BACKEND,
        normalize: normalizeEngineBackend,
        logLabel: 'backend:loaded',
        logField: 'selectedEngineBackend',
        assign: (value) => {
            selectedEngineBackend = value;
        },
        afterLoad: () => {
            backendPreferenceLoaded = true;
        }
    });
}

function loadWebGpuModelPreference() {
    return loadNormalizedPreference({
        key: WEBGPU_MODEL_KEY,
        defaultValue: DEFAULT_WEBGPU_MODEL,
        normalize: normalizeWebGpuModel,
        logLabel: 'webgpu-model:loaded',
        logField: 'selectedWebGpuModel',
        assign: (value) => {
            selectedWebGpuModel = value;
        }
    });
}

function loadWebGpuScalePreference() {
    return loadNormalizedPreference({
        key: WEBGPU_SCALE_KEY,
        defaultValue: DEFAULT_WEBGPU_SCALE,
        normalize: normalizeWebGpuScale,
        logLabel: 'webgpu-scale:loaded',
        logField: 'selectedWebGpuScale',
        assign: (value) => {
            selectedWebGpuScale = value;
        }
    });
}

function loadOnnxModelPreference() {
    return loadNormalizedPreference({
        key: ONNX_MODEL_KEY,
        defaultValue: DEFAULT_ONNX_MODEL,
        normalize: normalizeOnnxModel,
        logLabel: 'onnx-model:loaded',
        logField: 'selectedOnnxModel',
        assign: (value) => {
            selectedOnnxModel = value;
        }
    });
}

presetReadyPromise = loadSimplePresetPreference();
backendReadyPromise = loadEngineBackendPreference();
webgpuModelReadyPromise = loadWebGpuModelPreference();
webgpuScaleReadyPromise = loadWebGpuScalePreference();
onnxModelReadyPromise = loadOnnxModelPreference();
