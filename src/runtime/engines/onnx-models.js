// ONNX model descriptors.
// Each descriptor fully specifies a model's paths, cache identity, and inference geometry.
// To add a new model: define a new descriptor constant and set onnxActiveModel to it.
//
// Descriptor fields:
//   label                — identifier string used in logs
//   modelPath            — extension-relative path to the .onnx file
//   externalDataPath     — path to external weight sidecar (.onnx.data), or null for single-file models
//   externalDataPathAliases — basenames the protobuf may reference for externalData linking
//   cacheSchemaVersion   — cache namespace; bump when model or format changes to invalidate stale cache
//   inputAlignment       — required input dimension alignment (each dim padded to nearest multiple)
//   fixedInputWidth      — required input width (0 = dynamic axes, model accepts any size)
//   fixedInputHeight     — required input height (0 = dynamic axes, model accepts any size)
//   maxSinglePassPixels  — below this pixel count, run the full image in one pass; above it, use tiles
//   maxForegroundPixels  — pixel budget used to derive the target tile edge size
//   tileOverlapPixels    — overlap border added on each side of a tile to reduce seam artifacts
//   minTileEdgePixels    — minimum tile edge in pixels
//   maxTileEdgePixels    — maximum tile edge in pixels

const REALESRGAN_2XPLUS_ONNX_MODEL = {
    key: 'realesrgan-2xplus',
    title: 'RealESRGAN 2x+',
    label: 'REALESRGAN_2XPLUS_ONNX',
    modelPath: 'models/realesrgan_2xplus.onnx',
    externalDataPath: 'models/realesrgan_2xplus.onnx.data',
    externalDataPathAliases: [
        'realesrgan_2xplus.onnx.data',
        'realesrgan_2xplus_dynamic.onnx.data'
    ],
    cacheSchemaVersion: 'v1',
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 900000,
    maxForegroundPixels: 1600000,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const REALESRGAN_GENEREAL_X4_V3_ONNX_MODEL = {
    key: 'realesr-general-x4v3',
    title: 'RealESR General x4 v3',
    label: 'REALESRGAN_GENEREAL_X4_V3_ONNX',
    modelPath: 'models/realesr-general-x4v3.onnx',
    externalDataPath: 'models/realesr-general-x4v3.onnx.data',
    externalDataPathAliases: [
        'realesr-general-x4v3.onnx.data',
        'realesr-general-x4v3_dynamic.onnx.data'
    ],
    cacheSchemaVersion: 'v2',
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 900000,
    maxForegroundPixels: 1600000,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const UP2X_LATEST_CONSERVATIVE_ONNX_MODEL = {
    key: 'up2x-latest-conservative',
    title: 'RealCUGAN Latest Conservative',
    label: 'UP2X_LATEST_CONSERVATIVE_ONNX',
    modelPath: 'models/up2x-latest-conservative.onnx',
    externalDataPath: 'models/up2x-latest-conservative.onnx.data',
    externalDataPathAliases: [
        'up2x-latest-conservative.onnx.data',
        './up2x-latest-conservative.onnx.data',
        'models/up2x-latest-conservative.onnx.data',
        '"up2x-latest-conservative.onnx.data"'
    ],
    cacheSchemaVersion: 'v2',
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 900000,
    maxForegroundPixels: 1600000,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const UP2X_LATEST_DENOISE1X_ONNX_MODEL = {
    key: 'up2x-latest-denoise1x',
    title: 'RealCUGAN 2X Latest Denoise 1x',
    label: 'UP2X_LATEST_DENOISE1X_ONNX',
    modelPath: 'models/up2x-latest-denoise1x.onnx',
    externalDataPath: 'models/up2x-latest-denoise1x.onnx.data',
    externalDataPathAliases: [
        'up2x-latest-denoise1x.onnx.data',
        './up2x-latest-denoise1x.onnx.data',
        'models/up2x-latest-denoise1x.onnx.data',
        '"up2x-latest-denoise1x.onnx.data"'
    ],
    cacheSchemaVersion: 'v2',
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 900000,
    maxForegroundPixels: 1600000,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const DEFAULT_ONNX_MODEL_KEY = REALESRGAN_2XPLUS_ONNX_MODEL.key;

const ONNX_MODEL_REGISTRY = Object.freeze({
    [REALESRGAN_2XPLUS_ONNX_MODEL.key]: Object.freeze(REALESRGAN_2XPLUS_ONNX_MODEL),
    [REALESRGAN_GENEREAL_X4_V3_ONNX_MODEL.key]: Object.freeze(REALESRGAN_GENEREAL_X4_V3_ONNX_MODEL),
    [UP2X_LATEST_CONSERVATIVE_ONNX_MODEL.key]: Object.freeze(UP2X_LATEST_CONSERVATIVE_ONNX_MODEL),
    [UP2X_LATEST_DENOISE1X_ONNX_MODEL.key]: Object.freeze(UP2X_LATEST_DENOISE1X_ONNX_MODEL)
});

function normalizeOnnxModelKey(value) {
    const normalized = String(value || '').trim();
    return ONNX_MODEL_REGISTRY[normalized] ? normalized : DEFAULT_ONNX_MODEL_KEY;
}

function resolveOnnxModelByKey(value) {
    return ONNX_MODEL_REGISTRY[normalizeOnnxModelKey(value)];
}

function setOnnxActiveModelByKey(value) {
    onnxActiveModel = resolveOnnxModelByKey(value);
    return onnxActiveModel;
}

function getOnnxModelOptions() {
    return Object.values(ONNX_MODEL_REGISTRY).map((model) => ({
        key: model.key,
        title: model.title,
        label: model.label
    }));
}

// Expose registry helpers for popup/settings UIs that run outside the runtime bundle.
globalThis.DEFAULT_ONNX_MODEL_KEY = DEFAULT_ONNX_MODEL_KEY;
globalThis.ONNX_MODEL_REGISTRY = ONNX_MODEL_REGISTRY;
globalThis.normalizeOnnxModelKey = normalizeOnnxModelKey;
globalThis.resolveOnnxModelByKey = resolveOnnxModelByKey;
globalThis.setOnnxActiveModelByKey = setOnnxActiveModelByKey;
globalThis.getOnnxModelOptions = getOnnxModelOptions;


// Active model used by the ONNX engine. Reassign to switch models at runtime.
let onnxActiveModel = REALESRGAN_2XPLUS_ONNX_MODEL;
