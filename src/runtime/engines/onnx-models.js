// ONNX model descriptors.
// Each descriptor fully specifies a model's paths, cache identity, and inference geometry.
// To add a new model: define a new descriptor constant and set onnxActiveModel to it.
//
// Descriptor fields:
//   label                 Eidentifier string used in logs
//   modelPath             Eextension-relative path to the .onnx file
//   externalDataPath      Epath to external weight sidecar (.onnx.data), or null for single-file models
//   externalDataPathAliases  Ebasenames the protobuf may reference for externalData linking
//   cacheSchemaVersion    Ecache namespace; bump when model or format changes to invalidate stale cache
//   inputAlignment        Erequired input dimension alignment (each dim padded to nearest multiple)
//   fixedInputWidth       Erequired input width (0 = dynamic axes, model accepts any size)
//   fixedInputHeight      Erequired input height (0 = dynamic axes, model accepts any size)
//   maxSinglePassPixels   Ebelow this pixel count, run the full image in one pass; above it, use tiles
//   maxForegroundPixels   Epixel budget used to derive the target tile edge size
//   tileOverlapPixels     Eoverlap border added on each side of a tile to reduce seam artifacts
//   minTileEdgePixels     Eminimum tile edge in pixels
//   maxTileEdgePixels     Emaximum tile edge in pixels

const REALESRGAN_2XPLUS_ONNX_MODEL = {
    key: 'realesrgan-2xplus',
    title: 'RealESRGAN 2x+',
    label: 'REALESRGAN_2XPLUS_ONNX',
    modelPath: 'models/RealESRGAN_x2plus.onnx',
    externalDataPath: null,
    externalDataPathAliases: [],
    cacheSchemaVersion: 'v3',  // dynamic axes  Ere-export required
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 1048576,
    maxForegroundPixels: 3145728,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const UP2X_LATEST_CONSERVATIVE_ONNX_MODEL = {
    key: 'up2x-latest-conservative',
    title: 'RealCUGAN Conservative',
    label: 'UP2X_LATEST_CONSERVATIVE_ONNX',
    modelPath: 'models/up2x-latest-conservative.onnx',
    externalDataPath: null,
    externalDataPathAliases: [],
    cacheSchemaVersion: 'v3',  // dynamic axes  Ere-export required
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 1048576,
    maxForegroundPixels: 3145728,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const UP2X_LATEST_DENOISE1X_ONNX_MODEL = {
    key: 'up2x-latest-denoise1x',
    title: 'RealCUGAN 2X Denoise 1x',
    label: 'UP2X_LATEST_DENOISE1X_ONNX',
    modelPath: 'models/up2x-latest-denoise1x.onnx',
    externalDataPath: null,
    externalDataPathAliases: [],
    cacheSchemaVersion: 'v3',  // dynamic axes  Ere-export required
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 1048576,
    maxForegroundPixels: 3145728,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const ANIMESHARP_V2_MOSR_ONNX_MODEL = {
    key: 'animesharp-v2-mosr-sharp',
    title: 'AnimeSharp V2 MoSR Sharp',
    label: 'ANIMESHARP_V2_MOSR_SHARP_ONNX',
    modelPath: 'models/2x-AnimeSharpV2_MoSR_Sharp.onnx',
    externalDataPath: null,
    externalDataPathAliases: [],
    cacheSchemaVersion: 'v2',  // dynamic axes  Ere-export required
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 1048576,
    maxForegroundPixels: 3145728,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const ANIMESHARP_V2_RPLKSR_ONNX_MODEL = {
    key: 'animesharp-v2-rplksr-sharp',
    title: 'AnimeSharp V2 RPLKSR Sharp',
    label: 'ANIMESHARP_V2_RPLKSR_SHARP_ONNX',
    modelPath: 'models/2x-AnimeSharpV2_RPLKSR_Sharp.onnx',
    externalDataPath: null,
    externalDataPathAliases: [],
    cacheSchemaVersion: 'v2',  // dynamic axes  Ere-export required
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 1048576,
    maxForegroundPixels: 3145728,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const MANGAJANAI_1200P_V1_ONNX_MODEL = {
    key: 'mangajanai-1200p-v1',
    title: 'MangaJaNai 2x 1200p V1',
    label: 'MANGAJANAI_1200P_V1_ONNX',
    modelPath: 'models/2x_MangaJaNai_1200p_V1_ESRGAN_70k.onnx',
    externalDataPath: null,
    externalDataPathAliases: [],
    cacheSchemaVersion: 'v2',  // dynamic axes  Ere-export required
    inputAlignment: 2,
    fixedInputWidth: 0,
    fixedInputHeight: 0,
    maxSinglePassPixels: 1048576,
    maxForegroundPixels: 3145728,
    tileOverlapPixels: 16,
    minTileEdgePixels: 128,
    maxTileEdgePixels: 512
};

const DEFAULT_ONNX_MODEL_KEY = REALESRGAN_2XPLUS_ONNX_MODEL.key;

const ONNX_MODEL_REGISTRY = Object.freeze({
    [REALESRGAN_2XPLUS_ONNX_MODEL.key]: Object.freeze(REALESRGAN_2XPLUS_ONNX_MODEL),
    [UP2X_LATEST_CONSERVATIVE_ONNX_MODEL.key]: Object.freeze(UP2X_LATEST_CONSERVATIVE_ONNX_MODEL),
    [UP2X_LATEST_DENOISE1X_ONNX_MODEL.key]: Object.freeze(UP2X_LATEST_DENOISE1X_ONNX_MODEL),
    [MANGAJANAI_1200P_V1_ONNX_MODEL.key]: Object.freeze(MANGAJANAI_1200P_V1_ONNX_MODEL),
    [ANIMESHARP_V2_MOSR_ONNX_MODEL.key]: Object.freeze(ANIMESHARP_V2_MOSR_ONNX_MODEL),
    [ANIMESHARP_V2_RPLKSR_ONNX_MODEL.key]: Object.freeze(ANIMESHARP_V2_RPLKSR_ONNX_MODEL)
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
