// WebGL adapter using anime4k-webgl library

/** @type {any} */
let scaler = null;
/** @type {Promise<any> | null} */
let scalerPromise = null;
const WEBGL_FALLBACK_MAX_CANVAS_DIMENSION = 16384;
const WEBGL_UPSCALER_CTOR_NAMES = ['ImageUpscaler'];

function getWebGlLibrary() {
    const lib = window.Anime4KJS || window.Anime4K;
    return lib && typeof lib === 'object' ? lib : null;
}

function isWebGlLibrarySupported(lib) {
    if (typeof window.engineLibraryHasAnyFunctions === 'function') {
        return window.engineLibraryHasAnyFunctions(lib, WEBGL_UPSCALER_CTOR_NAMES);
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

async function getScaler(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    if (scaler) return scaler;
    if (scalerPromise) return scalerPromise;

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

    try {
        scaler = await scalerPromise;
        return scaler;
    } finally {
        scalerPromise = null;
    }
}

function resetWebGlAdapterState() {
    scaler = null;
    scalerPromise = null;
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

function upscaleWith2dFallback(tempImg, canvas, scale) {
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

    canvas.width = targetWidth;
    canvas.height = targetHeight;

    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        throw new Error(`2D fallback canvas size rejected: ${targetWidth}x${targetHeight}`);
    }

    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) {
        throw new Error('2D fallback failed to acquire canvas context');
    }

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, targetWidth, targetHeight);
    ctx.drawImage(tempImg, 0, 0, targetWidth, targetHeight);
}

async function runAnime4KWebGl(tempImg, canvas, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const engine = await getScaler(settings);
    if (!engine.supported) {
        throw new Error('Anime4KJS WebGL pipeline not supported');
    }

    const fallbackScale = resolveWebGlFallbackScale(settings);

    try {
        engine.attachSource(tempImg, canvas);
        engine.upscale();
    } catch (error) {
        runtimeLog('webgl:fallback-2d', {
            reason: 'webgl-error',
            error: String(error),
            sourceWidth: tempImg.naturalWidth || tempImg.width,
            sourceHeight: tempImg.naturalHeight || tempImg.height,
            fallbackScale
        });
        upscaleWith2dFallback(tempImg, canvas, fallbackScale);
        return `2D_FALLBACK_${fallbackScale}X`;
    } finally {
        engine.detachSource();
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
    upscaleWith2dFallback(tempImg, canvas, fallbackScale);
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
