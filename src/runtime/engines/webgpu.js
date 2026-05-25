// WebGPU adapter using anime4k-webgpu library

let webgpuDevicePromise = null;
let webgpuRenderPipeline = null;
let webgpuRenderPipelineFormat = null;
let webgpuRenderBindGroupLayout = null;
let webgpuSampler = null;
let webgpuVertexModule = null;
let webgpuFragmentModule = null;
let webgpuRenderPipelineLayout = null;
const webgpuCanvasContextCache = new WeakMap();
const WEBGPU_WAIT_FOR_SUBMISSION = false;
let webgpuProcessingCache = null;
let webgpuTileScratchCache = null;
const WEBGPU_MODEL_CTOR_NAMES = ['Anime4K', 'ModeA', 'ModeAA', 'ModeB', 'ModeBB', 'ModeC', 'ModeCA'];

function getWebGpuLibrary() {
    const lib = window['anime4k-webgpu'];
    return lib && typeof lib === 'object' ? lib : null;
}

function isWebGpuLibrarySupported(lib) {
    if (typeof engineLibraryHasAnyFunctions === 'function') {
        return engineLibraryHasAnyFunctions(lib, WEBGPU_MODEL_CTOR_NAMES);
    }
    return WEBGPU_MODEL_CTOR_NAMES.some((name) => typeof lib?.[name] === 'function');
}

function getWebGpuPresetCtor(lib, runtimeSettings = getRuntimePreferenceSnapshot()) {
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

async function getWebGpuDevice() {
    if (webgpuDevicePromise) return webgpuDevicePromise;

    webgpuDevicePromise = (async () => {
        if (!navigator?.gpu) {
            throw new Error('WebGPU API unavailable in this browser context');
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter available');
        }

        const requiredLimits = {};
        const adapterLimits = adapter.limits;
        const defaultMaxBufferSize = 268435456;
        const defaultMaxStorageBufferBindingSize = 134217728;
        const defaultMaxTextureDimension2D = 8192;

        if (adapterLimits) {
            if (
                typeof adapterLimits.maxBufferSize === 'number' &&
                Number.isFinite(adapterLimits.maxBufferSize) &&
                adapterLimits.maxBufferSize > defaultMaxBufferSize
            ) {
                requiredLimits.maxBufferSize = adapterLimits.maxBufferSize;
            }
            if (
                typeof adapterLimits.maxStorageBufferBindingSize === 'number' &&
                Number.isFinite(adapterLimits.maxStorageBufferBindingSize) &&
                adapterLimits.maxStorageBufferBindingSize > defaultMaxStorageBufferBindingSize
            ) {
                requiredLimits.maxStorageBufferBindingSize = adapterLimits.maxStorageBufferBindingSize;
            }
            if (
                typeof adapterLimits.maxTextureDimension2D === 'number' &&
                Number.isFinite(adapterLimits.maxTextureDimension2D) &&
                adapterLimits.maxTextureDimension2D > defaultMaxTextureDimension2D
            ) {
                requiredLimits.maxTextureDimension2D = adapterLimits.maxTextureDimension2D;
            }
        }

        try {
            if (Object.keys(requiredLimits).length > 0) {
                return await adapter.requestDevice({ requiredLimits });
            }
            return await adapter.requestDevice();
        } catch {
            return adapter.requestDevice();
        }
    })();

    try {
        return await webgpuDevicePromise;
    } catch (err) {
        webgpuDevicePromise = null;
        throw err;
    }
}

function getWebGpuLibraryForAdapter() {
    if (!navigator?.gpu) return null;
    return getWebGpuLibrary();
}

function resetWebGpuAdapterState() {
    webgpuDevicePromise = null;
    webgpuRenderPipeline = null;
    webgpuRenderPipelineFormat = null;
    webgpuRenderBindGroupLayout = null;
    webgpuSampler = null;
    webgpuVertexModule = null;
    webgpuFragmentModule = null;
    webgpuRenderPipelineLayout = null;
    disposeWebGpuProcessingCache();
    disposeWebGpuTileScratchCache();
}

function disposeWebGpuProcessingCache() {
    if (!webgpuProcessingCache) return;

    try {
        webgpuProcessingCache.inputTexture?.destroy?.();
    } catch {}

    try {
        webgpuProcessingCache.outputTexture?.destroy?.();
    } catch {}

    webgpuProcessingCache = null;
}

function disposeWebGpuTileScratchCache() {
    if (!webgpuTileScratchCache) return;

    webgpuTileScratchCache.sourceCanvas.width = 0;
    webgpuTileScratchCache.sourceCanvas.height = 0;
    webgpuTileScratchCache.outputCanvas.width = 0;
    webgpuTileScratchCache.outputCanvas.height = 0;
    webgpuTileScratchCache = null;
}

function getOrCreateWebGpuTileScratchCache() {
    if (webgpuTileScratchCache) {
        return webgpuTileScratchCache;
    }

    const sourceCanvas = document.createElement('canvas');
    const sourceCtx = sourceCanvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!sourceCtx) {
        throw new Error('WebGPU tiled mode failed to create reusable tile source context');
    }

    sourceCtx.imageSmoothingEnabled = false;

    webgpuTileScratchCache = {
        sourceCanvas,
        sourceCtx,
        outputCanvas: document.createElement('canvas')
    };

    return webgpuTileScratchCache;
}

function resizeCanvasIfNeeded(canvas, width, height) {
    if (canvas.width !== width) {
        canvas.width = width;
    }
    if (canvas.height !== height) {
        canvas.height = height;
    }
}

function roundProfileDuration(value) {
    return Math.round(value * 1000) / 1000;
}

function getProfileNow() {
    if (typeof performance?.now === 'function') {
        return performance.now();
    }
    return Date.now();
}

function formatWebGpuProfileDurations(rawDurations = {}) {
    const result = {};
    Object.entries(rawDurations).forEach(([key, value]) => {
        if (Number.isFinite(value)) {
            result[key] = roundProfileDuration(value);
        }
    });
    return result;
}

function createWebGpuProfileCollector(enabled, baseFields = {}) {
    if (!enabled) return null;

    return {
        enabled,
        baseFields,
        emit(label, data = {}) {
            runtimeProfileLog(label, {
                ...baseFields,
                ...data
            });
        }
    };
}

function createWebGpuProcessingCacheKey({ kind, modelName, nativeWidth, nativeHeight, targetWidth, targetHeight }) {
    return `${kind}|${modelName}|${nativeWidth}x${nativeHeight}|${targetWidth}x${targetHeight}`;
}

function getOrCreateWebGpuProcessingPipeline({
    lib,
    device,
    settings,
    nativeWidth,
    nativeHeight,
    targetWidth,
    targetHeight
}) {
    const presetCtor = typeof lib.Anime4K === 'function' ? null : getWebGpuPresetCtor(lib, settings);
    if (!lib.Anime4K && !presetCtor) {
        throw new Error('No compatible anime4k-webgpu preset class is exported');
    }

    const kind = typeof lib.Anime4K === 'function' ? 'anime4k' : 'preset';
    const modelName = kind === 'anime4k'
        ? 'Anime4K'
        : (presetCtor.name || settings.selectedWebGpuModel);
    const cacheKey = createWebGpuProcessingCacheKey({
        kind,
        modelName,
        nativeWidth,
        nativeHeight,
        targetWidth,
        targetHeight
    });

    if (
        webgpuProcessingCache &&
        webgpuProcessingCache.device === device &&
        webgpuProcessingCache.key === cacheKey
    ) {
        return { ...webgpuProcessingCache, cacheHit: true };
    }

    disposeWebGpuProcessingCache();

    const inputTexture = device.createTexture({
        size: [nativeWidth, nativeHeight, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
    });

    if (kind === 'anime4k') {
        const outputTexture = device.createTexture({
            size: [targetWidth, targetHeight, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
        });

        const anime = new lib.Anime4K(device, inputTexture);
        webgpuProcessingCache = {
            key: cacheKey,
            device,
            kind,
            modelUsed: modelName,
            inputTexture,
            outputTexture,
            anime,
            pipeline: null
        };
    } else {
        const pipeline = new presetCtor({
            device,
            inputTexture,
            nativeDimensions: { width: nativeWidth, height: nativeHeight },
            targetDimensions: { width: targetWidth, height: targetHeight }
        });

        if (typeof pipeline.pass !== 'function' || typeof pipeline.getOutputTexture !== 'function') {
            throw new Error('Invalid anime4k-webgpu pipeline interface');
        }

        const outputTexture = pipeline.getOutputTexture();
        webgpuProcessingCache = {
            key: cacheKey,
            device,
            kind,
            modelUsed: modelName,
            inputTexture,
            outputTexture,
            anime: null,
            pipeline
        };
    }

    return { ...webgpuProcessingCache, cacheHit: false };
}

function getWebGpuRenderResources(device) {
    if (!webgpuRenderBindGroupLayout) {
        webgpuRenderBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} }
            ]
        });
    }

    if (!webgpuSampler) {
        webgpuSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    }

    if (!webgpuVertexModule) {
        webgpuVertexModule = device.createShaderModule({
            code: `
@vertex
fn main(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );
    let p = positions[vertexIndex];
    return vec4<f32>(p, 0.0, 1.0);
}
`
        });
    }

    if (!webgpuFragmentModule) {
        webgpuFragmentModule = device.createShaderModule({
            code: `
@group(0) @binding(0) var linearSampler : sampler;
@group(0) @binding(1) var sourceTex : texture_2d<f32>;

@fragment
fn main(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(sourceTex, 0));
    let uv = pos.xy / dims;
    return textureSample(sourceTex, linearSampler, uv);
}
`
        });
    }

    if (!webgpuRenderPipelineLayout) {
        webgpuRenderPipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [webgpuRenderBindGroupLayout]
        });
    }

    return {
        bindGroupLayout: webgpuRenderBindGroupLayout,
        sampler: webgpuSampler,
        vertexModule: webgpuVertexModule,
        fragmentModule: webgpuFragmentModule,
        pipelineLayout: webgpuRenderPipelineLayout
    };
}

function getOrCreateWebGpuRenderPipeline(device, format) {
    if (webgpuRenderPipeline && webgpuRenderPipelineFormat === format) {
        return webgpuRenderPipeline;
    }

    const shader = getWebGpuRenderResources(device);

    webgpuRenderPipeline = device.createRenderPipeline({
        layout: shader.pipelineLayout,
        vertex: { module: shader.vertexModule, entryPoint: 'main' },
        fragment: { module: shader.fragmentModule, entryPoint: 'main', targets: [{ format }] },
        primitive: { topology: 'triangle-list' }
    });
    webgpuRenderPipelineFormat = format;
    return webgpuRenderPipeline;
}

function getOrCreateWebGpuCanvasContext(canvas, device, format) {
    let cached = webgpuCanvasContextCache.get(canvas);
    if (!cached) {
        const context = canvas.getContext('webgpu');
        if (!context) {
            throw new Error('Failed to acquire WebGPU canvas context');
        }
        cached = { context, configuredDevice: null, configuredFormat: null };
        webgpuCanvasContextCache.set(canvas, cached);
    }

    if (cached.configuredDevice !== device || cached.configuredFormat !== format) {
        cached.context.configure({ device, format, alphaMode: 'premultiplied' });
        cached.configuredDevice = device;
        cached.configuredFormat = format;
    }

    return cached.context;
}

async function runAnime4KWebGpuSingle(sourceImage, canvas, settings, device, maxTextureDimension2D) {
    const nativeWidth = sourceImage.naturalWidth || sourceImage.width;
    const nativeHeight = sourceImage.naturalHeight || sourceImage.height;
    if (
        !Number.isFinite(nativeWidth) ||
        !Number.isFinite(nativeHeight) ||
        nativeWidth <= 0 ||
        nativeHeight <= 0
    ) {
        throw new Error('Invalid source image dimensions for WebGPU');
    }

    const requestedScale = settings.selectedWebGpuScale;
    const targetWidth = Math.max(1, Math.round(nativeWidth * requestedScale));
    const targetHeight = Math.max(1, Math.round(nativeHeight * requestedScale));
    const profiling = createWebGpuProfileCollector(isRuntimeProfilingEnabled(), {
        backend: 'webgpu',
        runKind: 'single',
        model: settings.selectedWebGpuModel,
        scale: requestedScale,
        nativeWidth,
        nativeHeight,
        targetWidth,
        targetHeight
    });
    const timings = profiling ? { totalStart: getProfileNow() } : null;

    if (
        nativeWidth > maxTextureDimension2D ||
        nativeHeight > maxTextureDimension2D ||
        targetWidth > maxTextureDimension2D ||
        targetHeight > maxTextureDimension2D
    ) {
        throw new Error(
            `WebGPU texture limits exceeded: input=${nativeWidth}x${nativeHeight}, target=${targetWidth}x${targetHeight}, max=${maxTextureDimension2D}`
        );
    }

    const lib = getWebGpuLibrary();
    if (!lib) {
        throw new Error('anime4k-webgpu runtime is not loaded on window');
    }

    const processing = getOrCreateWebGpuProcessingPipeline({
        lib,
        device,
        settings,
        nativeWidth,
        nativeHeight,
        targetWidth,
        targetHeight
    });

    if (timings) {
        timings.pipelineReadyAt = getProfileNow();
    }

    device.queue.copyExternalImageToTexture(
        { source: sourceImage },
        { texture: processing.inputTexture },
        [nativeWidth, nativeHeight]
    );

    if (timings) {
        timings.copyCompleteAt = getProfileNow();
    }

    const encoder = device.createCommandEncoder();
    let outputTexture = null;
    let modelUsed = processing.modelUsed;

    if (processing.kind === 'anime4k') {
        outputTexture = processing.outputTexture;
        processing.anime.render(outputTexture, encoder);
    } else {
        outputTexture = processing.outputTexture;
        processing.pipeline.pass(encoder);
    }

    if (!outputTexture) {
        throw new Error('anime4k-webgpu did not produce an output texture');
    }

    if (timings) {
        timings.processPassEncodedAt = getProfileNow();
    }

    if (canvas.width !== outputTexture.width) {
        canvas.width = outputTexture.width;
    }
    if (canvas.height !== outputTexture.height) {
        canvas.height = outputTexture.height;
    }

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    const context = getOrCreateWebGpuCanvasContext(canvas, device, canvasFormat);

    const renderPipeline = getOrCreateWebGpuRenderPipeline(device, canvasFormat);
    const shader = getWebGpuRenderResources(device);
    const bindGroup = device.createBindGroup({
        layout: shader.bindGroupLayout,
        entries: [
            { binding: 0, resource: shader.sampler },
            { binding: 1, resource: outputTexture.createView() }
        ]
    });

    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store'
        }]
    });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();

    if (timings) {
        timings.renderPassEncodedAt = getProfileNow();
    }

    device.queue.submit([encoder.finish()]);

    if (timings) {
        timings.submitAt = getProfileNow();
    }

    if (WEBGPU_WAIT_FOR_SUBMISSION) {
        await device.queue.onSubmittedWorkDone();
    }

    if (profiling?.enabled) {
        await device.queue.onSubmittedWorkDone();
        const submittedWorkDoneAt = getProfileNow();
        profiling.emit('webgpu-single', {
            modelUsed,
            cacheHit: !!processing.cacheHit,
            processingKind: processing.kind,
            durationsMs: formatWebGpuProfileDurations({
                total: submittedWorkDoneAt - timings.totalStart,
                pipelineSetup: timings.pipelineReadyAt - timings.totalStart,
                textureUpload: timings.copyCompleteAt - timings.pipelineReadyAt,
                processEncode: timings.processPassEncodedAt - timings.copyCompleteAt,
                presentEncode: timings.renderPassEncodedAt - timings.processPassEncodedAt,
                submit: timings.submitAt - timings.renderPassEncodedAt,
                gpuExecution: submittedWorkDoneAt - timings.submitAt
            })
        });
    }

    return { modelUsed, width: canvas.width, height: canvas.height };
}

async function runAnime4KWebGpuTiled(tempImg, canvas, settings, device, maxTextureDimension2D) {
    const nativeWidth = tempImg.naturalWidth || tempImg.width;
    const nativeHeight = tempImg.naturalHeight || tempImg.height;
    const scale = settings.selectedWebGpuScale;

    const maxSourceTileWidth = Math.max(1, Math.floor(maxTextureDimension2D / scale));
    const maxSourceTileHeight = Math.max(1, Math.floor(maxTextureDimension2D / scale));
    const sourceTileWidth = Math.min(nativeWidth, maxSourceTileWidth);
    const sourceTileHeight = Math.min(nativeHeight, maxSourceTileHeight);

    if (sourceTileWidth <= 0 || sourceTileHeight <= 0) {
        throw new Error(`WebGPU tiled mode failed to compute tile size for max=${maxTextureDimension2D}, scale=${scale}`);
    }

    const finalTargetWidth = Math.max(1, Math.round(nativeWidth * scale));
    const finalTargetHeight = Math.max(1, Math.round(nativeHeight * scale));
    const profiling = createWebGpuProfileCollector(isRuntimeProfilingEnabled(), {
        backend: 'webgpu',
        runKind: 'tiled',
        model: settings.selectedWebGpuModel,
        scale,
        nativeWidth,
        nativeHeight,
        targetWidth: finalTargetWidth,
        targetHeight: finalTargetHeight,
        maxTextureDimension2D
    });
    const tiledTimingStart = profiling ? getProfileNow() : 0;
    canvas.width = finalTargetWidth;
    canvas.height = finalTargetHeight;

    if (canvas.width !== finalTargetWidth || canvas.height !== finalTargetHeight) {
        throw new Error(`WebGPU tiled output canvas rejected size: ${finalTargetWidth}x${finalTargetHeight}`);
    }

    const composeCtx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!composeCtx) {
        throw new Error('WebGPU tiled mode failed to acquire 2D canvas context for composition');
    }

    composeCtx.imageSmoothingEnabled = false;
    composeCtx.clearRect(0, 0, finalTargetWidth, finalTargetHeight);

    const tileScratch = getOrCreateWebGpuTileScratchCache();
    const tileSourceCanvas = tileScratch.sourceCanvas;
    const tileSourceCtx = tileScratch.sourceCtx;
    const tileOutputCanvas = tileScratch.outputCanvas;

    let modelUsed = settings.selectedWebGpuModel;
    let tileCount = 0;
    let tileExtractionMs = 0;
    let tileSingleRunMs = 0;
    let tileComposeMs = 0;

    for (let sourceY = 0; sourceY < nativeHeight; sourceY += sourceTileHeight) {
        const tileSourceHeight = Math.min(sourceTileHeight, nativeHeight - sourceY);

        for (let sourceX = 0; sourceX < nativeWidth; sourceX += sourceTileWidth) {
            const tileSourceWidth = Math.min(sourceTileWidth, nativeWidth - sourceX);

            const tileExtractionStart = profiling ? getProfileNow() : 0;
            resizeCanvasIfNeeded(tileSourceCanvas, tileSourceWidth, tileSourceHeight);
            tileSourceCtx.clearRect(0, 0, tileSourceWidth, tileSourceHeight);
            tileSourceCtx.drawImage(
                tempImg,
                sourceX,
                sourceY,
                tileSourceWidth,
                tileSourceHeight,
                0,
                0,
                tileSourceWidth,
                tileSourceHeight
            );
            if (profiling) {
                tileExtractionMs += getProfileNow() - tileExtractionStart;
            }

            const tileSingleRunStart = profiling ? getProfileNow() : 0;
            const tileResult = await runAnime4KWebGpuSingle(
                tileSourceCanvas,
                tileOutputCanvas,
                settings,
                device,
                maxTextureDimension2D
            );
            if (profiling) {
                tileSingleRunMs += getProfileNow() - tileSingleRunStart;
            }
            modelUsed = tileResult.modelUsed;

            const targetX = Math.round(sourceX * scale);
            const targetY = Math.round(sourceY * scale);
            const targetWidth = tileResult.width;
            const targetHeight = tileResult.height;

            const tileComposeStart = profiling ? getProfileNow() : 0;
            composeCtx.drawImage(
                tileOutputCanvas,
                0,
                0,
                targetWidth,
                targetHeight,
                targetX,
                targetY,
                targetWidth,
                targetHeight
            );
            if (profiling) {
                tileComposeMs += getProfileNow() - tileComposeStart;
            }

            tileCount++;
        }
    }

    runtimeLog('webgpu:tiled-run', {
        sourceWidth: nativeWidth,
        sourceHeight: nativeHeight,
        targetWidth: finalTargetWidth,
        targetHeight: finalTargetHeight,
        scale,
        maxTextureDimension2D,
        sourceTileWidth,
        sourceTileHeight,
        tileCount
    });

    if (profiling) {
        profiling.emit('webgpu-tiled', {
            modelUsed,
            sourceTileWidth,
            sourceTileHeight,
            tileCount,
            durationsMs: formatWebGpuProfileDurations({
                total: getProfileNow() - tiledTimingStart,
                tileExtraction: tileExtractionMs,
                tileSingleRun: tileSingleRunMs,
                tileCompose: tileComposeMs
            })
        });
    }

    return modelUsed;
}

async function runAnime4KWebGpu(tempImg, canvas, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const profilingEnabled = isRuntimeProfilingEnabled();
    const upscaleStart = profilingEnabled ? getProfileNow() : 0;
    const lib = getWebGpuLibrary();
    if (!lib) {
        throw new Error('anime4k-webgpu runtime is not loaded on window');
    }

    const device = await getWebGpuDevice();
    const nativeWidth = tempImg.naturalWidth || tempImg.width;
    const nativeHeight = tempImg.naturalHeight || tempImg.height;
    if (
        !Number.isFinite(nativeWidth) ||
        !Number.isFinite(nativeHeight) ||
        nativeWidth <= 0 ||
        nativeHeight <= 0
    ) {
        throw new Error('Invalid source image dimensions for WebGPU');
    }

    const requestedScale = settings.selectedWebGpuScale;
    const targetWidth = Math.max(1, Math.round(nativeWidth * requestedScale));
    const targetHeight = Math.max(1, Math.round(nativeHeight * requestedScale));
    const maxTextureDimension2D = device?.limits?.maxTextureDimension2D || 8192;

    if (
        nativeWidth > maxTextureDimension2D ||
        nativeHeight > maxTextureDimension2D ||
        targetWidth > maxTextureDimension2D ||
        targetHeight > maxTextureDimension2D
    ) {
        const tiledModel = await runAnime4KWebGpuTiled(tempImg, canvas, settings, device, maxTextureDimension2D);
        if (profilingEnabled) {
            runtimeProfileLog('webgpu-upscale', {
                backend: 'webgpu',
                runMode: 'tiled',
                model: tiledModel,
                requestedModel: settings.selectedWebGpuModel,
                scale: settings.selectedWebGpuScale,
                nativeWidth,
                nativeHeight,
                targetWidth,
                targetHeight,
                durationsMs: formatWebGpuProfileDurations({
                    total: getProfileNow() - upscaleStart
                })
            });
        }
        return { model: tiledModel, runMode: 'tiled' };
    }

    const singleRun = await runAnime4KWebGpuSingle(
        tempImg,
        canvas,
        settings,
        device,
        maxTextureDimension2D
    );
    if (profilingEnabled) {
        runtimeProfileLog('webgpu-upscale', {
            backend: 'webgpu',
            runMode: 'single',
            model: singleRun.modelUsed,
            requestedModel: settings.selectedWebGpuModel,
            scale: settings.selectedWebGpuScale,
            nativeWidth,
            nativeHeight,
            targetWidth,
            targetHeight,
            durationsMs: formatWebGpuProfileDurations({
                total: getProfileNow() - upscaleStart
            })
        });
    }
    return { model: singleRun.modelUsed, runMode: 'single' };
}

async function prewarmWebGpu() {
    await getWebGpuDevice();
}

window.WebGPUAdapter = createLibraryBackedEngineAdapter({
    getLibrary: getWebGpuLibraryForAdapter,
    isLibrarySupported: isWebGpuLibrarySupported,
    upscale: runAnime4KWebGpu,
    ensureReady: prewarmWebGpu,
    resetState: resetWebGpuAdapterState,
    isInitialized: () => !!webgpuDevicePromise
});
