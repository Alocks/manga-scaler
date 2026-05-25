// WebGL adapter using anime4k-webgl library

/** @type {any} */
let scaler = null;
/** @type {Promise<any> | null} */
let scalerPromise = null;
let scalerCacheKey = null;
let scalerPromiseKey = null;
const WEBGL_FALLBACK_MAX_CANVAS_DIMENSION = 16384;
const WEBGL_UPSCALER_CTOR_NAMES = ['ImageUpscaler'];
let fallbackResizeCanvas = null;
let fallbackResizeContext = null;
let fallbackResizeCanvasWidth = 0;
let fallbackResizeCanvasHeight = 0;
const fallbackOutputContextCache = new WeakMap();
let webglProcessingCanvas = null;
let webglAttachedEngine = null;

function getWebGlLibrary() {
    const lib = window.Anime4KJS || window.Anime4K;
    return lib && typeof lib === 'object' ? lib : null;
}

function isWebGlLibrarySupported(lib) {
    if (typeof engineLibraryHasAnyFunctions === 'function') {
        return engineLibraryHasAnyFunctions(lib, WEBGL_UPSCALER_CTOR_NAMES);
    }
    return WEBGL_UPSCALER_CTOR_NAMES.some((name) => typeof lib?.[name] === 'function');
}

function getRestoreUpscalePreset(lib, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const preferredSimple = {
        S: lib.ANIME4KJS_SIMPLE_S_2X,
        M: lib.ANIME4KJS_SIMPLE_M_2X,
        L: lib.ANIME4KJS_SIMPLE_L_2X,
        VL: lib.ANIME4KJS_SIMPLE_VL_2X,
        UL: lib.ANIME4KJS_SIMPLE_UL_2X
    };

    if (preferredSimple[settings.selectedSimplePreset]) {
        return preferredSimple[settings.selectedSimplePreset];
    }

    const fallbackOrder = ['M', 'L', 'S', 'UL', 'VL'];
    for (const key of fallbackOrder) {
        if (preferredSimple[key]) return preferredSimple[key];
    }

    return lib.ANIME4KJS_EMPTY || [];
}

function getWebGlScalerCacheKey(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    return `preset:${settings.selectedSimplePreset}`;
}

function getOrCreateFallbackOutputContext(canvas) {
    let ctx = fallbackOutputContextCache.get(canvas);
    if (!ctx) {
        ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
        if (!ctx) {
            throw new Error('2D fallback failed to acquire output context');
        }
        fallbackOutputContextCache.set(canvas, ctx);
    }
    return ctx;
}

function getOrCreateFallbackResizeContext(targetWidth, targetHeight) {
    if (
        !fallbackResizeCanvas ||
        !fallbackResizeContext ||
        fallbackResizeCanvasWidth !== targetWidth ||
        fallbackResizeCanvasHeight !== targetHeight
    ) {
        fallbackResizeCanvas = createResizeCanvas(targetWidth, targetHeight);
        fallbackResizeContext = fallbackResizeCanvas.getContext('2d', { alpha: true, desynchronized: true });
        if (!fallbackResizeContext) {
            throw new Error('2D fallback failed to acquire resize context');
        }
        fallbackResizeCanvasWidth = targetWidth;
        fallbackResizeCanvasHeight = targetHeight;
    }

    return { canvas: fallbackResizeCanvas, ctx: fallbackResizeContext };
}

function getOrCreateWebGlProcessingCanvas() {
    if (!webglProcessingCanvas) {
        webglProcessingCanvas = createResizeCanvas(1, 1);
    }
    return webglProcessingCanvas;
}

function ensureWebGlEngineAttachment(engine, sourceImage) {
    const processingCanvas = getOrCreateWebGlProcessingCanvas();

    if (webglAttachedEngine !== engine) {
        try {
            webglAttachedEngine?.detachSource?.();
        } catch {}

        engine.attachSource(sourceImage, processingCanvas);
        webglAttachedEngine = engine;
        return processingCanvas;
    }

    engine.source = sourceImage;
    engine.canvas = processingCanvas;
    return processingCanvas;
}

function copyWebGlOutputToTargetCanvas(sourceCanvas, targetCanvas) {
    const width = sourceCanvas.width || 0;
    const height = sourceCanvas.height || 0;
    if (width <= 0 || height <= 0) return;

    if (targetCanvas.width !== width) {
        targetCanvas.width = width;
    }
    if (targetCanvas.height !== height) {
        targetCanvas.height = height;
    }

    const ctx = getOrCreateFallbackOutputContext(targetCanvas);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(sourceCanvas, 0, 0, width, height);
}

async function getScaler(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const cacheKey = getWebGlScalerCacheKey(settings);

    if (scaler && scalerCacheKey === cacheKey) return scaler;

    if (scalerPromise) {
        const pendingScaler = await scalerPromise;
        if (scalerPromiseKey === cacheKey) {
            return pendingScaler;
        }
    }

    scalerPromise = (async () => {
        await Promise.all([presetReadyPromise, backendReadyPromise]);

        const lib = getWebGlLibrary();
        if (!lib) {
            throw new Error('Anime4KJS WebGL runtime not found on window');
        }

        const preset = getRestoreUpscalePreset(lib, settings);
        runtimeLog('scaler:init', {
            backend: 'webgl',
            hasImageUpscaler: typeof lib.ImageUpscaler === 'function',
            presetLength: Array.isArray(preset) ? preset.length : null,
            selectedSimplePreset: settings.selectedSimplePreset
        });

        const instance = new lib.ImageUpscaler(preset);

        if (instance && instance.supported !== undefined) {
            runtimeLog('scaler:webgl-supported', { supported: instance.supported });
        }

        if (instance && instance.renderer) {
            runtimeLog('scaler:renderer-info', {
                hasRenderer: !!instance.renderer,
                rendererType: instance.renderer?.constructor?.name
            });
        }

        return instance;
    })();
    scalerPromiseKey = cacheKey;

    try {
        scaler = await scalerPromise;
        scalerCacheKey = cacheKey;
        return scaler;
    } finally {
        scalerPromise = null;
        scalerPromiseKey = null;
    }
}

function resetWebGlAdapterState() {
    try {
        webglAttachedEngine?.detachSource?.();
    } catch {}

    scaler = null;
    scalerPromise = null;
    scalerCacheKey = null;
    scalerPromiseKey = null;
    fallbackResizeCanvas = null;
    fallbackResizeContext = null;
    fallbackResizeCanvasWidth = 0;
    fallbackResizeCanvasHeight = 0;
    webglProcessingCanvas = null;
    webglAttachedEngine = null;
}

function resolveWebGlFallbackScale(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const presetScaleMap = {
        S: 2,
        M: 2,
        L: 2,
        VL: 2,
        UL: 2
    };
    return presetScaleMap[settings.selectedSimplePreset] || 2;
}

function createResizeCanvas(width, height) {
    if (typeof OffscreenCanvas === 'function') {
        return new OffscreenCanvas(width, height);
    }

    const elementCanvas = document.createElement('canvas');
    elementCanvas.width = width;
    elementCanvas.height = height;
    return elementCanvas;
}

async function tryImageBitmapHighQualityResize(tempImg, canvas, targetWidth, targetHeight) {
    if (typeof createImageBitmap !== 'function') return false;

    let bitmap = null;
    try {
        bitmap = await createImageBitmap(tempImg, {
            resizeWidth: targetWidth,
            resizeHeight: targetHeight,
            resizeQuality: 'high'
        });

        if (canvas.width !== targetWidth) {
            canvas.width = targetWidth;
        }
        if (canvas.height !== targetHeight) {
            canvas.height = targetHeight;
        }

        const ctx = getOrCreateFallbackOutputContext(canvas);

        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.clearRect(0, 0, targetWidth, targetHeight);
        ctx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
        return true;
    } catch {
        return false;
    } finally {
        if (bitmap && typeof bitmap.close === 'function') {
            bitmap.close();
        }
    }
}

async function upscaleWith2dFallback(tempImg, canvas, scale) {
    const sourceWidth = tempImg.naturalWidth || tempImg.width;
    const sourceHeight = tempImg.naturalHeight || tempImg.height;

    if (
        !Number.isFinite(sourceWidth) ||
        !Number.isFinite(sourceHeight) ||
        sourceWidth <= 0 ||
        sourceHeight <= 0
    ) {
        throw new Error('2D fallback cannot resolve valid source dimensions');
    }

    const maxCanvasDimension = WEBGL_FALLBACK_MAX_CANVAS_DIMENSION;
    if (sourceWidth > maxCanvasDimension || sourceHeight > maxCanvasDimension) {
        throw new Error(`2D fallback source exceeds canvas limits: source=${sourceWidth}x${sourceHeight}, max=${maxCanvasDimension}`);
    }

    const maxScaleByDimension = Math.min(maxCanvasDimension / sourceWidth, maxCanvasDimension / sourceHeight);
    const effectiveScale = Math.max(1, Math.min(scale, maxScaleByDimension));

    const targetWidth = Math.max(1, Math.min(maxCanvasDimension, Math.floor(sourceWidth * effectiveScale)));
    const targetHeight = Math.max(1, Math.min(maxCanvasDimension, Math.floor(sourceHeight * effectiveScale)));

    const usedImageBitmapResize = await tryImageBitmapHighQualityResize(tempImg, canvas, targetWidth, targetHeight);
    if (usedImageBitmapResize) {
        return;
    }

    const resizeBuffer = getOrCreateFallbackResizeContext(targetWidth, targetHeight);
    const intermediateCanvas = resizeBuffer.canvas;
    const intermediateCtx = resizeBuffer.ctx;

    intermediateCtx.imageSmoothingEnabled = true;
    intermediateCtx.imageSmoothingQuality = 'high';
    intermediateCtx.clearRect(0, 0, targetWidth, targetHeight);
    intermediateCtx.drawImage(tempImg, 0, 0, targetWidth, targetHeight);

    if (canvas.width !== targetWidth) {
        canvas.width = targetWidth;
    }
    if (canvas.height !== targetHeight) {
        canvas.height = targetHeight;
    }
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        throw new Error(`2D fallback canvas size rejected: ${targetWidth}x${targetHeight}`);
    }

    const outputCtx = getOrCreateFallbackOutputContext(canvas);

    outputCtx.imageSmoothingEnabled = true;
    outputCtx.imageSmoothingQuality = 'high';
    outputCtx.clearRect(0, 0, targetWidth, targetHeight);
    outputCtx.drawImage(intermediateCanvas, 0, 0, targetWidth, targetHeight);
}

async function runAnime4KWebGl(tempImg, canvas, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const engine = await getScaler(settings);
    if (!engine.supported) {
        throw new Error('Anime4KJS WebGL pipeline not supported');
    }

    const fallbackScale = resolveWebGlFallbackScale(settings);

    try {
        const processingCanvas = ensureWebGlEngineAttachment(engine, tempImg);
        engine.upscale();
        copyWebGlOutputToTargetCanvas(processingCanvas, canvas);
    } catch (error) {
        runtimeLog('webgl:fallback-2d', {
            reason: 'webgl-error',
            error: String(error),
            sourceWidth: tempImg.naturalWidth || tempImg.width,
            sourceHeight: tempImg.naturalHeight || tempImg.height,
            fallbackScale
        });
        await upscaleWith2dFallback(tempImg, canvas, fallbackScale);
        return `2D_FALLBACK_${fallbackScale}X`;
    }

    if (canvas.width > 0 && canvas.height > 0) {
        return `SIMPLE_${settings.selectedSimplePreset}`;
    }

    runtimeLog('webgl:fallback-2d', {
        reason: 'empty-canvas',
        sourceWidth: tempImg.naturalWidth || tempImg.width,
        sourceHeight: tempImg.naturalHeight || tempImg.height,
        fallbackScale
    });
    await upscaleWith2dFallback(tempImg, canvas, fallbackScale);
    return `2D_FALLBACK_${fallbackScale}X`;
}

async function prewarmWebGl(runtimeSettings = getRuntimePreferenceSnapshot()) {
    await getScaler(runtimeSettings);
}

function isWebGlRuntimeSupported() {
    return !!scaler && scaler.supported === true;
}

window.WebGLAdapter = createLibraryBackedEngineAdapter({
    getLibrary: getWebGlLibrary,
    isLibrarySupported: isWebGlLibrarySupported,
    upscale: runAnime4KWebGl,
    ensureReady: prewarmWebGl,
    resetState: resetWebGlAdapterState,
    isInitialized: () => !!scaler,
    isRuntimeSupported: isWebGlRuntimeSupported
});
