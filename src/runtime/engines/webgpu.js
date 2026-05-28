// WebGPU backend orchestration; shader factories live under runtime/factories/shaders.

let webgpuDevicePromise = null;
let webgpuRenderPipeline = null;
let webgpuRenderPipelineFormat = null;
let webgpuRenderBindGroupLayout = null;
let webgpuSampler = null;
let webgpuVertexModule = null;
let webgpuFragmentModule = null;
let webgpuRenderPipelineLayout = null;
const webgpuCanvasContextCache = new WeakMap();
const webgpuCanvasSizeCache = new WeakMap();
let webgpuPreferredCanvasFormat = null;
const WEBGPU_PROCESSING_CACHE_MAX_ENTRIES = 6;
const webgpuProcessingCacheByKey = new Map();
let webgpuProcessingCacheTick = 0;
const webgpuPreparedImageBitmapCache = new Map();
let webgpuTiledComposeCache = null;
let webgpuTiledInputScratchCache = null;
const WEBGPU_TILE_INPUT_SCRATCH_SIZE = 1024;

function destroyWebGpuTexture(texture) {
    try {
        texture?.destroy?.();
    } catch {}
}

function closeWebGpuImageBitmap(bitmap) {
    try {
        bitmap?.close?.();
    } catch {}
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

function resetWebGpuAdapterState() {
    webgpuDevicePromise = null;
    webgpuRenderPipeline = null;
    webgpuRenderPipelineFormat = null;
    webgpuRenderBindGroupLayout = null;
    webgpuSampler = null;
    webgpuVertexModule = null;
    webgpuFragmentModule = null;
    webgpuRenderPipelineLayout = null;
    webgpuPreferredCanvasFormat = null;
    disposeWebGpuProcessingCache();
    disposeWebGpuPreparedImageBitmapCache();
    disposeWebGpuTiledComposeCache();
    disposeWebGpuTiledInputScratchCache();
}

function getWebGpuPreferredCanvasFormat() {
    if (webgpuPreferredCanvasFormat) {
        return webgpuPreferredCanvasFormat;
    }

    webgpuPreferredCanvasFormat = navigator.gpu.getPreferredCanvasFormat();
    return webgpuPreferredCanvasFormat;
}

function disposeWebGpuProcessingCache() {
    if (webgpuProcessingCacheByKey.size <= 0) return;

    for (const entry of webgpuProcessingCacheByKey.values()) {
        disposeWebGpuProcessingEntry(entry);
    }

    webgpuProcessingCacheByKey.clear();
}

function disposeWebGpuProcessingEntry(entry) {
    if (!entry) return;

    try {
        entry.disposeProcessing?.();
    } catch {}

    if (entry.ownsInputTexture !== false) {
        destroyWebGpuTexture(entry.inputTexture);
    }

    destroyWebGpuTexture(entry.outputTexture);
}

function touchWebGpuProcessingEntry(entry) {
    entry.lastUsedTick = ++webgpuProcessingCacheTick;
}

function evictWebGpuProcessingCacheEntries() {
    if (webgpuProcessingCacheByKey.size <= WEBGPU_PROCESSING_CACHE_MAX_ENTRIES) {
        return;
    }

    const entries = [...webgpuProcessingCacheByKey.entries()];
    entries.sort((a, b) => {
        const aTick = Number.isFinite(a[1]?.lastUsedTick) ? a[1].lastUsedTick : 0;
        const bTick = Number.isFinite(b[1]?.lastUsedTick) ? b[1].lastUsedTick : 0;
        return aTick - bTick;
    });

    const removeCount = webgpuProcessingCacheByKey.size - WEBGPU_PROCESSING_CACHE_MAX_ENTRIES;
    for (let i = 0; i < removeCount; i++) {
        const [key, entry] = entries[i] || [];
        if (!key) continue;
        webgpuProcessingCacheByKey.delete(key);
        disposeWebGpuProcessingEntry(entry);
    }
}

function getWebGpuProcessingCacheEntry(cacheKey, device) {
    const entry = webgpuProcessingCacheByKey.get(cacheKey);
    if (!entry) {
        return null;
    }

    if (entry.device !== device) {
        webgpuProcessingCacheByKey.delete(cacheKey);
        disposeWebGpuProcessingEntry(entry);
        return null;
    }

    entry.cacheHit = true;
    touchWebGpuProcessingEntry(entry);
    return entry;
}

function setWebGpuProcessingCacheEntry(cacheKey, entry) {
    touchWebGpuProcessingEntry(entry);
    webgpuProcessingCacheByKey.set(cacheKey, entry);
    evictWebGpuProcessingCacheEntries();
    return entry;
}

function disposeWebGpuTiledComposeCache() {
    if (!webgpuTiledComposeCache) return;

    destroyWebGpuTexture(webgpuTiledComposeCache.texture);

    webgpuTiledComposeCache = null;
}

function disposeWebGpuTiledInputScratchCache() {
    if (!webgpuTiledInputScratchCache) return;

    destroyWebGpuTexture(webgpuTiledInputScratchCache.texture);

    webgpuTiledInputScratchCache = null;
}

function getOrCreateWebGpuTiledInputScratch(device, maxTextureDimension2D) {
    const size = Math.max(1, Math.min(WEBGPU_TILE_INPUT_SCRATCH_SIZE, maxTextureDimension2D || WEBGPU_TILE_INPUT_SCRATCH_SIZE));

    if (
        webgpuTiledInputScratchCache &&
        webgpuTiledInputScratchCache.device === device &&
        webgpuTiledInputScratchCache.size === size
    ) {
        return webgpuTiledInputScratchCache;
    }

    disposeWebGpuTiledInputScratchCache();

    const texture = device.createTexture({
        size: [size, size, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
    });

    webgpuTiledInputScratchCache = {
        device,
        size,
        texture
    };

    return webgpuTiledInputScratchCache;
}

function getOrCreateWebGpuTiledComposeCache(device, width, height) {
    const format = 'rgba16float';
    if (
        webgpuTiledComposeCache &&
        webgpuTiledComposeCache.device === device &&
        webgpuTiledComposeCache.width === width &&
        webgpuTiledComposeCache.height === height &&
        webgpuTiledComposeCache.format === format
    ) {
        return webgpuTiledComposeCache;
    }

    disposeWebGpuTiledComposeCache();

    const texture = device.createTexture({
        size: [width, height, 1],
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    });

    webgpuTiledComposeCache = {
        device,
        width,
        height,
        format,
        texture,
        textureView: texture.createView(),
        presentCarrier: {
            outputTexture: texture,
            outputTextureView: null,
            presentBindGroup: null,
            presentBindGroupDevice: null
        }
    };

    return webgpuTiledComposeCache;
}

function disposeWebGpuPreparedImageBitmapCache() {
    if (webgpuPreparedImageBitmapCache.size <= 0) return;

    for (const cached of webgpuPreparedImageBitmapCache.values()) {
        closeWebGpuImageBitmap(cached.bitmap);
    }

    webgpuPreparedImageBitmapCache.clear();
}

function getWebGpuImageSourceCacheKey(sourceImage) {
    const src = sourceImage.currentSrc || sourceImage.src || '';
    const width = sourceImage.naturalWidth || sourceImage.width || 0;
    const height = sourceImage.naturalHeight || sourceImage.height || 0;
    return `${src}|${width}x${height}`;
}

async function createWebGpuUploadSource(sourceImage) {
    if (typeof ImageBitmap !== 'undefined' && sourceImage instanceof ImageBitmap) {
        return { source: sourceImage, dispose: null, sourceType: 'direct-imagebitmap' };
    }

    if (typeof VideoFrame !== 'undefined' && sourceImage instanceof VideoFrame) {
        return { source: sourceImage, dispose: null, sourceType: 'direct-videoframe' };
    }

    if (typeof HTMLVideoElement !== 'undefined' && sourceImage instanceof HTMLVideoElement) {
        return { source: sourceImage, dispose: null, sourceType: 'direct-video' };
    }

    if (typeof HTMLCanvasElement !== 'undefined' && sourceImage instanceof HTMLCanvasElement) {
        return { source: sourceImage, dispose: null, sourceType: 'direct-canvas' };
    }

    if (typeof OffscreenCanvas !== 'undefined' && sourceImage instanceof OffscreenCanvas) {
        return { source: sourceImage, dispose: null, sourceType: 'direct-offscreen-canvas' };
    }

    if (typeof HTMLImageElement !== 'undefined' && sourceImage instanceof HTMLImageElement) {
        if (typeof createImageBitmap !== 'function') {
            return { source: sourceImage, dispose: null, sourceType: 'direct-image' };
        }

        const cacheKey = getWebGpuImageSourceCacheKey(sourceImage);
        const cached = webgpuPreparedImageBitmapCache.get(sourceImage);
        if (cached && cached.cacheKey === cacheKey && cached.bitmap) {
            return { source: cached.bitmap, dispose: null, sourceType: 'imageBitmap-cached' };
        }

        try {
            const bitmap = await createImageBitmap(sourceImage, {
                premultiplyAlpha: 'none',
                colorSpaceConversion: 'none'
            });

            if (cached?.bitmap && cached.bitmap !== bitmap) {
                closeWebGpuImageBitmap(cached.bitmap);
            }

            webgpuPreparedImageBitmapCache.set(sourceImage, { cacheKey, bitmap });
            return { source: bitmap, sourceType: 'imageBitmap-prepared', dispose: null };
        } catch {
            return { source: sourceImage, dispose: null, sourceType: 'direct-image-fallback' };
        }
    }

    if (typeof createImageBitmap !== 'function') {
        return { source: sourceImage, dispose: null, sourceType: 'direct' };
    }

    try {
        const bitmap = await createImageBitmap(sourceImage, {
            premultiplyAlpha: 'none',
            colorSpaceConversion: 'none'
        });
        return {
            source: bitmap,
            sourceType: 'imageBitmap',
            dispose() {
                closeWebGpuImageBitmap(bitmap);
            }
        };
    } catch {
        return { source: sourceImage, dispose: null, sourceType: 'direct' };
    }
}

function encodeWebGpuBlitPass({
    encoder,
    targetTextureView,
    renderPipeline,
    bindGroup,
    loadOp,
    clearValue,
    viewport
}) {
    const colorAttachment = {
        view: targetTextureView,
        loadOp,
        storeOp: 'store'
    };

    if (loadOp === 'clear' && clearValue) {
        colorAttachment.clearValue = clearValue;
    }

    const pass = encoder.beginRenderPass({ colorAttachments: [colorAttachment] });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, bindGroup);

    if (viewport) {
        pass.setViewport(viewport.x, viewport.y, viewport.width, viewport.height, 0, 1);
        pass.setScissorRect(viewport.x, viewport.y, viewport.width, viewport.height);
    }

    pass.draw(6);
    pass.end();
}

function encodeWebGpuProcessingPass(processing, encoder) {
    const outputTexture = processing.outputTexture;

    if (typeof processing.runProcessingPass !== 'function') {
        throw new Error(`WebGPU processing entry is missing run callback for provider ${processing.providerName || 'unknown'}`);
    }
    processing.runProcessingPass(encoder);

    if (!outputTexture) {
        throw new Error(`WebGPU model provider ${processing.providerName || 'unknown'} did not produce an output texture`);
    }

    return outputTexture;
}

function copyExternalImageToInputTexture({
    device,
    uploadSource,
    sourceOriginX,
    sourceOriginY,
    inputTexture,
    width,
    height,
    timingTarget
}) {
    const uploadStartAt = timingTarget ? getProfileNow() : 0;
    const externalSource = { source: uploadSource };
    if (sourceOriginX !== 0 || sourceOriginY !== 0) {
        externalSource.origin = [sourceOriginX, sourceOriginY];
    }
    const uploadConfiguredAt = timingTarget ? getProfileNow() : 0;

    device.queue.copyExternalImageToTexture(
        externalSource,
        { texture: inputTexture },
        [width, height]
    );

    const uploadCopiedAt = timingTarget ? getProfileNow() : 0;

    if (timingTarget) {
        timingTarget.textureUploadStartAt = uploadStartAt;
        timingTarget.externalSourceReadyAt = uploadConfiguredAt;
        timingTarget.copyExternalImageCompleteAt = uploadCopiedAt;
        timingTarget.copyCompleteAt = uploadCopiedAt;
    }

    return {
        uploadStartAt,
        uploadConfiguredAt,
        uploadCopiedAt
    };
}

function buildWebGpuTextureUploadProfile({
    nativeWidth,
    nativeHeight,
    sourceOriginX,
    sourceOriginY,
    uploadSourceType,
    timings
}) {
    const uploadPixelCount = nativeWidth * nativeHeight;
    const estimatedUploadBytes = estimateRgba16FloatUploadBytes(nativeWidth, nativeHeight);

    return {
        sourceType: uploadSourceType,
        subRegionUpload: sourceOriginX !== 0 || sourceOriginY !== 0,
        sourceOriginX,
        sourceOriginY,
        uploadWidth: nativeWidth,
        uploadHeight: nativeHeight,
        uploadPixelCount,
        estimatedRgba16FloatUploadBytes: estimatedUploadBytes,
        estimatedRgba16FloatUploadMiB: roundProfileDuration(estimatedUploadBytes / (1024 * 1024)),
        durationsMs: formatWebGpuProfileDurations({
            total: timings.copyCompleteAt - timings.textureUploadStartAt,
            externalSourceConfig: timings.externalSourceReadyAt - timings.textureUploadStartAt,
            copyExternalImageCall: timings.copyExternalImageCompleteAt - timings.externalSourceReadyAt
        })
    };
}

function buildWebGpuSingleDurationsMs(timings) {
    return formatWebGpuProfileDurations({
        total: timings.submitAt - timings.totalStart,
        pipelineSetup: timings.pipelineReadyAt - timings.totalStart,
        textureUpload: timings.copyCompleteAt - timings.pipelineReadyAt,
        textureUploadExternalSourceConfig: timings.externalSourceReadyAt - timings.textureUploadStartAt,
        textureUploadCopyExternalImageCall: timings.copyExternalImageCompleteAt - timings.externalSourceReadyAt,
        processEncode: timings.processPassEncodedAt - timings.copyCompleteAt,
        presentEncode: timings.renderPassEncodedAt - timings.processPassEncodedAt,
        submit: timings.submitAt - timings.renderPassEncodedAt
    });
}

function buildWebGpuTiledDurationsMs({
    tiledTimingStart,
    tileExtractionMs,
    tileSingleRunMs,
    tileComposeMs,
    tileUploadTotalMs,
    tileUploadConfigMs,
    tileUploadCopyCallMs,
    tileCount
}) {
    return formatWebGpuProfileDurations({
        total: getProfileNow() - tiledTimingStart,
        tileExtraction: tileExtractionMs,
        tileSingleRun: tileSingleRunMs,
        tileCompose: tileComposeMs,
        tileUploadTotal: tileUploadTotalMs,
        tileUploadExternalSourceConfig: tileUploadConfigMs,
        tileUploadCopyExternalImageCall: tileUploadCopyCallMs,
        tileUploadTotalAvg: tileCount > 0 ? (tileUploadTotalMs / tileCount) : 0,
        tileUploadCopyExternalImageCallAvg: tileCount > 0 ? (tileUploadCopyCallMs / tileCount) : 0
    });
}

function buildWebGpuUpscaleDurationsMs({
    upscaleStart,
    uploadSourcePrepareMs,
    uploadDurations
}) {
    return formatWebGpuProfileDurations({
        total: getProfileNow() - upscaleStart,
        uploadSourcePrepare: uploadSourcePrepareMs,
        textureUpload: uploadDurations.total,
        textureUploadExternalSourceConfig: uploadDurations.externalSourceConfig,
        textureUploadCopyExternalImageCall: uploadDurations.copyExternalImageCall
    });
}

function ensureWebGpuCanvasTargetSize(canvas, width, height) {
    const cached = webgpuCanvasSizeCache.get(canvas);
    if (cached && cached.width === width && cached.height === height) {
        return;
    }

    canvas.width = width;
    canvas.height = height;
    webgpuCanvasSizeCache.set(canvas, { width, height });
}

function roundProfileDuration(value) {
    return Math.round(value * 1000) / 1000;
}

function estimateRgba16FloatUploadBytes(width, height) {
    const w = Number(width);
    const h = Number(height);
    if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
        return 0;
    }

    const bytesPerPixel = 8;
    return Math.round(w * h * bytesPerPixel);
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

function createWebGpuProcessingCacheKey({ providerName, kind, modelName, nativeWidth, nativeHeight, targetWidth, targetHeight }) {
    return `${providerName}|${kind}|${modelName}|${nativeWidth}x${nativeHeight}|${targetWidth}x${targetHeight}`;
}

function getOrCreateWebGpuProcessingPipeline({
    processingFactory,
    device,
    nativeWidth,
    nativeHeight,
    targetWidth,
    targetHeight,
    inputTextureOverride = null,
    inputTextureOverrideKey = ''
}) {
    if (!processingFactory || typeof processingFactory.createResources !== 'function') {
        throw new Error('WebGPU processing factory is not configured correctly');
    }

    const kind = processingFactory.kind || 'unknown';
    const modelName = processingFactory.modelName || 'default';
    const cacheKey = createWebGpuProcessingCacheKey({
        providerName: processingFactory.providerName || 'provider',
        kind,
        modelName,
        nativeWidth,
        nativeHeight,
        targetWidth,
        targetHeight
    }) + (inputTextureOverrideKey ? `|in:${inputTextureOverrideKey}` : '');

    if (
        webgpuProcessingCacheByKey.size > 0
    ) {
        const cached = getWebGpuProcessingCacheEntry(cacheKey, device);
        if (cached) {
            return cached;
        }
    }

    const inputTexture = inputTextureOverride || device.createTexture({
        size: [nativeWidth, nativeHeight, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
    });
    const ownsInputTexture = !inputTextureOverride;

    const created = processingFactory.createResources({
        device,
        inputTexture,
        nativeWidth,
        nativeHeight,
        targetWidth,
        targetHeight
    });
    if (!created || typeof created.runPass !== 'function' || !created.outputTexture) {
        throw new Error(`WebGPU model provider ${processingFactory.providerName || 'unknown'} returned an invalid processing pipeline`);
    }

    const entry = {
        key: cacheKey,
        device,
        providerName: processingFactory.providerName,
        kind,
        modelUsed: created.modelUsed || modelName,
        inputTexture,
        ownsInputTexture,
        outputTexture: created.outputTexture,
        runProcessingPass: created.runPass,
        disposeProcessing: created.dispose,
        cacheHit: false,
        outputTextureView: null,
        presentBindGroup: null,
        presentBindGroupDevice: null,
        lastUsedTick: 0
    };
    return setWebGpuProcessingCacheEntry(cacheKey, entry);
}

function getOrCreateWebGpuPresentBindGroup(processing, device) {
    if (
        processing.presentBindGroup &&
        processing.presentBindGroupDevice === device
    ) {
        return processing.presentBindGroup;
    }

    const shader = getWebGpuRenderResources(device);
    processing.outputTextureView = processing.outputTexture.createView();
    processing.presentBindGroup = device.createBindGroup({
        layout: shader.bindGroupLayout,
        entries: [
            { binding: 0, resource: shader.sampler },
            { binding: 1, resource: processing.outputTextureView }
        ]
    });
    processing.presentBindGroupDevice = device;

    return processing.presentBindGroup;
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
struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>
};

@vertex
fn main(@builtin(vertex_index) vertexIndex : u32) -> VSOut {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0)
    );

    var out : VSOut;
    let p = positions[vertexIndex];
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = uvs[vertexIndex];
    return out;
}
`
        });
    }

    if (!webgpuFragmentModule) {
        webgpuFragmentModule = device.createShaderModule({
            code: `
@group(0) @binding(0) var linearSampler : sampler;
@group(0) @binding(1) var sourceTex : texture_2d<f32>;

struct FSIn {
    @location(0) uv : vec2<f32>
};

@fragment
fn main(input : FSIn) -> @location(0) vec4<f32> {
    return textureSample(sourceTex, linearSampler, input.uv);
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
        cached = { context, configuredDevice: null, configuredFormat: null, configured: false };
        webgpuCanvasContextCache.set(canvas, cached);
    }

    const needsConfigure =
        !cached.configured ||
        cached.configuredDevice !== device ||
        cached.configuredFormat !== format;

    if (needsConfigure) {
        cached.context.configure({ device, format, alphaMode: 'premultiplied' });
        cached.configuredDevice = device;
        cached.configuredFormat = format;
        cached.configured = true;
    }

    return cached.context;
}

async function runAnime4KWebGpuSingle(sourceImage, canvas, settings, device, maxTextureDimension2D, uploadOptions = {}) {
    const sourceWidth = uploadOptions.sourceWidth || sourceImage.naturalWidth || sourceImage.width;
    const sourceHeight = uploadOptions.sourceHeight || sourceImage.naturalHeight || sourceImage.height;
    const uploadSource = uploadOptions.uploadSource || sourceImage;
    const uploadSourceType = uploadOptions.uploadSourceType || 'unknown';
    const sourceOriginX = uploadOptions.sourceOriginX || 0;
    const sourceOriginY = uploadOptions.sourceOriginY || 0;

    const nativeWidth = sourceWidth;
    const nativeHeight = sourceHeight;
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

    const lib = getAnime4kWebGpuLibrary();
    if (!lib) {
        throw new Error('anime4k-webgpu runtime is not loaded on window');
    }
    const processingFactory = getAnime4kWebGpuProcessingFactory(lib, settings);

    const processing = getOrCreateWebGpuProcessingPipeline({
        processingFactory,
        device,
        nativeWidth,
        nativeHeight,
        targetWidth,
        targetHeight
    });

    if (timings) {
        timings.pipelineReadyAt = getProfileNow();
    }

    copyExternalImageToInputTexture({
        device,
        uploadSource,
        sourceOriginX,
        sourceOriginY,
        inputTexture: processing.inputTexture,
        width: nativeWidth,
        height: nativeHeight,
        timingTarget: timings
    });

    const encoder = device.createCommandEncoder();
    let modelUsed = processing.modelUsed;
    const outputTexture = encodeWebGpuProcessingPass(processing, encoder);

    if (timings) {
        timings.processPassEncodedAt = getProfileNow();
    }

    if (canvas.width !== outputTexture.width || canvas.height !== outputTexture.height) {
        throw new Error(
            `WebGPU output canvas size mismatch: canvas=${canvas.width}x${canvas.height}, output=${outputTexture.width}x${outputTexture.height}`
        );
    }

    const canvasFormat = getWebGpuPreferredCanvasFormat();
    const context = getOrCreateWebGpuCanvasContext(canvas, device, canvasFormat);

    const renderPipeline = getOrCreateWebGpuRenderPipeline(device, canvasFormat);
    const bindGroup = getOrCreateWebGpuPresentBindGroup(processing, device);

    encodeWebGpuBlitPass({
        encoder,
        targetTextureView: context.getCurrentTexture().createView(),
        renderPipeline,
        bindGroup,
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 }
    });

    if (timings) {
        timings.renderPassEncodedAt = getProfileNow();
    }

    device.queue.submit([encoder.finish()]);

    if (timings) {
        timings.submitAt = getProfileNow();
    }

    if (profiling?.enabled) {
        const submittedWorkDoneAt = getProfileNow();
        const textureUploadDetails = buildWebGpuTextureUploadProfile({
            nativeWidth,
            nativeHeight,
            sourceOriginX,
            sourceOriginY,
            uploadSourceType,
            timings
        });

        profiling.emit('webgpu-single', {
            modelUsed,
            cacheHit: !!processing.cacheHit,
            provider: processing.providerName,
            processingKind: processing.kind,
            durationsMs: buildWebGpuSingleDurationsMs(timings),
            textureUploadDetails,
            gpuExecutionOmitted: true,
            profileTimestamp: roundProfileDuration(submittedWorkDoneAt)
        });

        return {
            modelUsed,
            width: canvas.width,
            height: canvas.height,
            uploadProfile: textureUploadDetails
        };
    }

    return { modelUsed, width: canvas.width, height: canvas.height };
}

async function runAnime4KWebGpuTiled(tempImg, canvas, settings, device, maxTextureDimension2D) {
    const nativeWidth = tempImg.naturalWidth || tempImg.width;
    const nativeHeight = tempImg.naturalHeight || tempImg.height;
    const scale = settings.selectedWebGpuScale;
    const tiledInputScratch = getOrCreateWebGpuTiledInputScratch(device, maxTextureDimension2D);

    const maxSourceTileWidth = Math.max(1, Math.floor(maxTextureDimension2D / scale));
    const maxSourceTileHeight = Math.max(1, Math.floor(maxTextureDimension2D / scale));
    const sourceTileWidth = Math.min(nativeWidth, maxSourceTileWidth, tiledInputScratch.size);
    const sourceTileHeight = Math.min(nativeHeight, maxSourceTileHeight, tiledInputScratch.size);

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
    ensureWebGpuCanvasTargetSize(canvas, finalTargetWidth, finalTargetHeight);

    if (canvas.width !== finalTargetWidth || canvas.height !== finalTargetHeight) {
        throw new Error(`WebGPU tiled output canvas rejected size: ${finalTargetWidth}x${finalTargetHeight}`);
    }

    const lib = getAnime4kWebGpuLibrary();
    if (!lib) {
        throw new Error('anime4k-webgpu runtime is not loaded on window');
    }
    const processingFactory = getAnime4kWebGpuProcessingFactory(lib, settings);

    const canvasFormat = getWebGpuPreferredCanvasFormat();
    const canvasContext = getOrCreateWebGpuCanvasContext(canvas, device, canvasFormat);
    const composeCache = getOrCreateWebGpuTiledComposeCache(device, finalTargetWidth, finalTargetHeight);
    const composePipeline = getOrCreateWebGpuRenderPipeline(device, composeCache.format);
    const composePresentBindGroup = getOrCreateWebGpuPresentBindGroup(composeCache.presentCarrier, device);

    const uploadSourceHandle = await createWebGpuUploadSource(tempImg);
    const uploadSource = uploadSourceHandle.source;

    let modelUsed = settings.selectedWebGpuModel;
    let tileCount = 0;
    let tileExtractionMs = 0;
    let tileSingleRunMs = 0;
    let tileComposeMs = 0;
    let tileUploadConfigMs = 0;
    let tileUploadCopyCallMs = 0;
    let tileUploadTotalMs = 0;

    try {
        const encoder = device.createCommandEncoder();

        const clearComposePass = encoder.beginRenderPass({
            colorAttachments: [{
                view: composeCache.textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });
        clearComposePass.end();

        for (let sourceY = 0; sourceY < nativeHeight; sourceY += sourceTileHeight) {
            const tileSourceHeight = Math.min(sourceTileHeight, nativeHeight - sourceY);

            for (let sourceX = 0; sourceX < nativeWidth; sourceX += sourceTileWidth) {
                const tileSourceWidth = Math.min(sourceTileWidth, nativeWidth - sourceX);
                const tileTargetWidth = Math.max(1, Math.round(tileSourceWidth * scale));
                const tileTargetHeight = Math.max(1, Math.round(tileSourceHeight * scale));
                const targetX = Math.round(sourceX * scale);
                const targetY = Math.round(sourceY * scale);

                const tileExtractionStart = profiling ? getProfileNow() : 0;
                const tileUploadOptions = {
                    uploadSource,
                    uploadSourceType: uploadSourceHandle.sourceType,
                    sourceWidth: tileSourceWidth,
                    sourceHeight: tileSourceHeight,
                    sourceOriginX: sourceX,
                    sourceOriginY: sourceY
                };
                if (profiling) {
                    tileExtractionMs += getProfileNow() - tileExtractionStart;
                }

                const tileSingleRunStart = profiling ? getProfileNow() : 0;
                const processing = getOrCreateWebGpuProcessingPipeline({
                    processingFactory,
                    device,
                    nativeWidth: tileSourceWidth,
                    nativeHeight: tileSourceHeight,
                    targetWidth: tileTargetWidth,
                    targetHeight: tileTargetHeight,
                    inputTextureOverride: tiledInputScratch.texture,
                    inputTextureOverrideKey: `tile-scratch-${tiledInputScratch.size}`
                });

                const uploadTiming = copyExternalImageToInputTexture({
                    device,
                    uploadSource,
                    sourceOriginX: tileUploadOptions.sourceOriginX,
                    sourceOriginY: tileUploadOptions.sourceOriginY,
                    inputTexture: processing.inputTexture,
                    width: tileSourceWidth,
                    height: tileSourceHeight,
                    timingTarget: profiling ? {} : null
                });

                encodeWebGpuProcessingPass(processing, encoder);

                const tileBindGroup = getOrCreateWebGpuPresentBindGroup(processing, device);
                const tileComposeStart = profiling ? getProfileNow() : 0;
                encodeWebGpuBlitPass({
                    encoder,
                    targetTextureView: composeCache.textureView,
                    renderPipeline: composePipeline,
                    bindGroup: tileBindGroup,
                    loadOp: 'load',
                    viewport: {
                        x: targetX,
                        y: targetY,
                        width: tileTargetWidth,
                        height: tileTargetHeight
                    }
                });
                if (profiling) {
                    tileComposeMs += getProfileNow() - tileComposeStart;
                    tileSingleRunMs += getProfileNow() - tileSingleRunStart;
                    tileUploadConfigMs += uploadTiming.uploadConfiguredAt - uploadTiming.uploadStartAt;
                    tileUploadCopyCallMs += uploadTiming.uploadCopiedAt - uploadTiming.uploadConfiguredAt;
                    tileUploadTotalMs += uploadTiming.uploadCopiedAt - uploadTiming.uploadStartAt;
                }

                modelUsed = processing.modelUsed;

                tileCount++;
            }
        }

        const presentPipeline = getOrCreateWebGpuRenderPipeline(device, canvasFormat);
        encodeWebGpuBlitPass({
            encoder,
            targetTextureView: canvasContext.getCurrentTexture().createView(),
            renderPipeline: presentPipeline,
            bindGroup: composePresentBindGroup,
            loadOp: 'clear',
            clearValue: { r: 0, g: 0, b: 0, a: 1 }
        });

        device.queue.submit([encoder.finish()]);
    } finally {
        uploadSourceHandle.dispose?.();
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
            provider: processingFactory.providerName,
            uploadSourceType: uploadSourceHandle.sourceType,
            sourceTileWidth,
            sourceTileHeight,
            tileCount,
            durationsMs: buildWebGpuTiledDurationsMs({
                tiledTimingStart,
                tileExtractionMs,
                tileSingleRunMs,
                tileComposeMs,
                tileUploadTotalMs,
                tileUploadConfigMs,
                tileUploadCopyCallMs,
                tileCount
            })
        });
    }

    return modelUsed;
}

async function runAnime4KWebGpu(tempImg, canvas, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const profilingEnabled = isRuntimeProfilingEnabled();
    const upscaleStart = profilingEnabled ? getProfileNow() : 0;

    const lib = getAnime4kWebGpuLibrary();
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
        const errorMessage = `WebGPU single-pass limits exceeded and tiled fallback is disabled for performance: input=${nativeWidth}x${nativeHeight}, target=${targetWidth}x${targetHeight}, max=${maxTextureDimension2D}`;
        runtimeLog('webgpu:tiled-disabled-fallback', {
            nativeWidth,
            nativeHeight,
            targetWidth,
            targetHeight,
            maxTextureDimension2D
        });
        throw new Error(errorMessage);
    }

    ensureWebGpuCanvasTargetSize(canvas, targetWidth, targetHeight);
    const uploadSourcePrepareStart = profilingEnabled ? getProfileNow() : 0;
    const uploadSourceHandle = await createWebGpuUploadSource(tempImg);
    const uploadSourcePrepareMs = profilingEnabled ? (getProfileNow() - uploadSourcePrepareStart) : 0;

    let singleRun;
    try {
        singleRun = await runAnime4KWebGpuSingle(
            tempImg,
            canvas,
            settings,
            device,
            maxTextureDimension2D,
            {
                uploadSource: uploadSourceHandle.source,
                uploadSourceType: uploadSourceHandle.sourceType
            }
        );
    } finally {
        uploadSourceHandle.dispose?.();
    }
    if (profilingEnabled) {
        const uploadDurations = singleRun.uploadProfile?.durationsMs || {};
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
            uploadSourceType: uploadSourceHandle.sourceType,
            durationsMs: buildWebGpuUpscaleDurationsMs({
                upscaleStart,
                uploadSourcePrepareMs,
                uploadDurations
            }),
            textureUploadDetails: singleRun.uploadProfile || null
        });
    }
    return { model: singleRun.modelUsed, runMode: 'single' };
}

async function prewarmWebGpu() {
    await getWebGpuDevice();
}

window.WebGPUAdapter = createLibraryBackedEngineAdapter({
    getLibrary: getAnime4kWebGpuLibraryForAdapter,
    isLibrarySupported: isAnime4kWebGpuLibrarySupported,
    upscale: runAnime4KWebGpu,
    ensureReady: prewarmWebGpu,
    resetState: resetWebGpuAdapterState,
    isInitialized: () => !!webgpuDevicePromise
});
