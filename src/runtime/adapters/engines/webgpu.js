// WebGPU adapter using anime4k-webgpu library

let webgpuDevicePromise = null;
let webgpuRenderPipeline = null;
let webgpuRenderPipelineFormat = null;
let webgpuRenderBindGroupLayout = null;
let webgpuSampler = null;
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
}

function getWebGpuRenderShaderModules(device) {
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

    return {
        bindGroupLayout: webgpuRenderBindGroupLayout,
        sampler: webgpuSampler,
        vertexModule: device.createShaderModule({
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
        }),
        fragmentModule: device.createShaderModule({
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
        })
    };
}

function getOrCreateWebGpuRenderPipeline(device, format) {
    if (webgpuRenderPipeline && webgpuRenderPipelineFormat === format) {
        return webgpuRenderPipeline;
    }

    const shader = getWebGpuRenderShaderModules(device);
    const layout = device.createPipelineLayout({ bindGroupLayouts: [shader.bindGroupLayout] });

    webgpuRenderPipeline = device.createRenderPipeline({
        layout,
        vertex: { module: shader.vertexModule, entryPoint: 'main' },
        fragment: { module: shader.fragmentModule, entryPoint: 'main', targets: [{ format }] },
        primitive: { topology: 'triangle-list' }
    });
    webgpuRenderPipelineFormat = format;
    return webgpuRenderPipeline;
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

    const inputTexture = device.createTexture({
        size: [nativeWidth, nativeHeight, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
    });

    device.queue.copyExternalImageToTexture(
        { source: sourceImage },
        { texture: inputTexture },
        [nativeWidth, nativeHeight]
    );

    const encoder = device.createCommandEncoder();
    let outputTexture = null;
    let modelUsed = settings.selectedWebGpuModel;
    const lib = getWebGpuLibrary();
    if (!lib) {
        throw new Error('anime4k-webgpu runtime is not loaded on window');
    }

    if (typeof lib.Anime4K === 'function') {
        const anime = new lib.Anime4K(device, inputTexture);
        modelUsed = 'Anime4K';
        outputTexture = device.createTexture({
            size: [targetWidth, targetHeight, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
        });
        anime.render(outputTexture, encoder);
    } else {
        const presetCtor = getWebGpuPresetCtor(lib, settings);
        if (!presetCtor) {
            throw new Error('No compatible anime4k-webgpu preset class is exported');
        }
        modelUsed = presetCtor.name || settings.selectedWebGpuModel;
        const pipeline = new presetCtor({
            device,
            inputTexture,
            nativeDimensions: { width: nativeWidth, height: nativeHeight },
            targetDimensions: { width: targetWidth, height: targetHeight }
        });

        if (typeof pipeline.pass !== 'function' || typeof pipeline.getOutputTexture !== 'function') {
            throw new Error('Invalid anime4k-webgpu pipeline interface');
        }

        pipeline.pass(encoder);
        outputTexture = pipeline.getOutputTexture();
    }

    if (!outputTexture) {
        throw new Error('anime4k-webgpu did not produce an output texture');
    }

    canvas.width = outputTexture.width;
    canvas.height = outputTexture.height;

    const context = canvas.getContext('webgpu');
    if (!context) {
        throw new Error('Failed to acquire WebGPU canvas context');
    }

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: canvasFormat, alphaMode: 'premultiplied' });

    const renderPipeline = getOrCreateWebGpuRenderPipeline(device, canvasFormat);
    const shader = getWebGpuRenderShaderModules(device);
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

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

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
    canvas.width = finalTargetWidth;
    canvas.height = finalTargetHeight;

    if (canvas.width !== finalTargetWidth || canvas.height !== finalTargetHeight) {
        throw new Error(`WebGPU tiled output canvas rejected size: ${finalTargetWidth}x${finalTargetHeight}`);
    }

    const composeCtx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!composeCtx) {
        throw new Error('WebGPU tiled mode failed to acquire 2D canvas context for composition');
    }

    composeCtx.imageSmoothingEnabled = true;
    composeCtx.imageSmoothingQuality = 'high';
    composeCtx.clearRect(0, 0, finalTargetWidth, finalTargetHeight);

    let modelUsed = settings.selectedWebGpuModel;
    let tileCount = 0;

    for (let sourceY = 0; sourceY < nativeHeight; sourceY += sourceTileHeight) {
        const tileSourceHeight = Math.min(sourceTileHeight, nativeHeight - sourceY);

        for (let sourceX = 0; sourceX < nativeWidth; sourceX += sourceTileWidth) {
            const tileSourceWidth = Math.min(sourceTileWidth, nativeWidth - sourceX);

            const tileSourceCanvas = document.createElement('canvas');
            tileSourceCanvas.width = tileSourceWidth;
            tileSourceCanvas.height = tileSourceHeight;
            const tileSourceCtx = tileSourceCanvas.getContext('2d', { alpha: true, desynchronized: true });
            if (!tileSourceCtx) {
                throw new Error('WebGPU tiled mode failed to create tile source context');
            }
            tileSourceCtx.imageSmoothingEnabled = true;
            tileSourceCtx.imageSmoothingQuality = 'high';
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

            const tileOutputCanvas = document.createElement('canvas');
            const tileResult = await runAnime4KWebGpuSingle(
                tileSourceCanvas,
                tileOutputCanvas,
                settings,
                device,
                maxTextureDimension2D
            );
            modelUsed = tileResult.modelUsed;

            const targetX = Math.round(sourceX * scale);
            const targetY = Math.round(sourceY * scale);
            const targetWidth = tileResult.width;
            const targetHeight = tileResult.height;

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

    return modelUsed;
}

async function runAnime4KWebGpu(tempImg, canvas, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
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
        return { model: tiledModel, runMode: 'tiled' };
    }

    const singleRun = await runAnime4KWebGpuSingle(
        tempImg,
        canvas,
        settings,
        device,
        maxTextureDimension2D
    );
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
