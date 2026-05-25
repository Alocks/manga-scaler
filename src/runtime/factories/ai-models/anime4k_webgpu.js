// Anime4K WebGPU model factory.

const WEBGPU_ANIME4K_MODEL_CTOR_NAMES = ['Anime4K', 'ModeA', 'ModeAA', 'ModeB', 'ModeBB', 'ModeC', 'ModeCA'];
const WEBGPU_ANIME4K_PROVIDER_NAME = 'anime4k-webgpu';

function getAnime4kWebGpuLibrary() {
    const lib = window['anime4k-webgpu'];
    return lib && typeof lib === 'object' ? lib : null;
}

function getAnime4kWebGpuLibraryForAdapter() {
    if (!navigator?.gpu) return null;
    return getAnime4kWebGpuLibrary();
}

function isAnime4kWebGpuLibrarySupported(lib) {
    if (typeof engineLibraryHasAnyFunctions === 'function') {
        return engineLibraryHasAnyFunctions(lib, WEBGPU_ANIME4K_MODEL_CTOR_NAMES);
    }
    return WEBGPU_ANIME4K_MODEL_CTOR_NAMES.some((name) => typeof lib?.[name] === 'function');
}

function getAnime4kWebGpuPresetCtor(lib, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const explicitCtor = lib[settings.selectedWebGpuModel];
    if (typeof explicitCtor === 'function') {
        return explicitCtor;
    }

    const presetByLevel = {
        S: [lib.ModeC, lib.ModeB, lib.ModeA],
        M: [lib.ModeB, lib.ModeA, lib.ModeC],
        L: [lib.ModeA, lib.ModeAA, lib.ModeB],
        VL: [lib.ModeAA, lib.ModeA, lib.ModeCA],
        UL: [lib.ModeCA, lib.ModeAA, lib.ModeA]
    };

    const ordered = presetByLevel[settings.selectedSimplePreset] || [lib.ModeA, lib.ModeB, lib.ModeC];
    for (const ctor of ordered) {
        if (typeof ctor === 'function') return ctor;
    }

    const fallback = [lib.ModeA, lib.ModeAA, lib.ModeB, lib.ModeBB, lib.ModeC, lib.ModeCA];
    for (const ctor of fallback) {
        if (typeof ctor === 'function') return ctor;
    }

    return null;
}

function getAnime4kWebGpuProcessingFactory(lib, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const presetCtor = typeof lib.Anime4K === 'function' ? null : getAnime4kWebGpuPresetCtor(lib, settings);
    if (!lib.Anime4K && !presetCtor) {
        throw new Error('No compatible anime4k-webgpu preset class is exported');
    }

    const kind = typeof lib.Anime4K === 'function' ? 'anime4k' : 'preset';
    const modelName = kind === 'anime4k'
        ? 'Anime4K'
        : (presetCtor.name || settings.selectedWebGpuModel);

    return createAiModelFactory({
        providerName: WEBGPU_ANIME4K_PROVIDER_NAME,
        kind,
        modelName,
        createResources({
            device,
            inputTexture,
            nativeWidth,
            nativeHeight,
            targetWidth,
            targetHeight
        }) {
            if (kind === 'anime4k') {
                const outputTexture = device.createTexture({
                    size: [targetWidth, targetHeight, 1],
                    format: 'rgba16float',
                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
                });
                const anime = new lib.Anime4K(device, inputTexture);

                return {
                    modelUsed: modelName,
                    outputTexture,
                    runPass(encoder) {
                        anime.render(outputTexture, encoder);
                    },
                    dispose: createAiModelDisposeFromMethods(anime, ['destroy', 'dispose'])
                };
            }

            const pipeline = new presetCtor({
                device,
                inputTexture,
                nativeDimensions: { width: nativeWidth, height: nativeHeight },
                targetDimensions: { width: targetWidth, height: targetHeight }
            });

            if (typeof pipeline.pass !== 'function' || typeof pipeline.getOutputTexture !== 'function') {
                throw new Error('Invalid anime4k-webgpu pipeline interface');
            }

            return {
                modelUsed: modelName,
                outputTexture: pipeline.getOutputTexture(),
                runPass(encoder) {
                    pipeline.pass(encoder);
                },
                dispose: createAiModelDisposeFromMethods(pipeline, ['destroy', 'dispose'])
            };
        }
    });
}
