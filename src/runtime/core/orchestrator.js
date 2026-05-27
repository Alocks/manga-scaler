// Main orchestrator: listeners, observers, and foreground image processing

const SAFETY_INTERVAL_MS = 1000;
const BACKGROUND_DISCOVERY_DEBOUNCE_MS = 150;
const NH_SCALER_HOOK_MARK = '__nhScalerHooked__';
const NH_SCALER_IMAGE_PROXY_MARK = '__nhScalerImageProxy__';
const BOOT_DIAGNOSTICS_PHASE_INITIAL = 'initial';
const BOOT_DIAGNOSTICS_PHASE_READY = 'ready';
const OBSERVED_IMAGE_ATTRIBUTES = ['src', 'srcset', 'data-src', 'data-srcset'];

let jobCounter = 0;
const CLEAR_CACHE_MESSAGE_TYPE = 'manga-scaler:clear-cache';
const GET_DIAGNOSTICS_MESSAGE_TYPE = 'manga-scaler:get-diagnostics';
let runtimeMutationSuppressed = false;
let runtimeSettingsFlushTimeoutId = null;
const onnxForegroundSourceLocks = new Set();

const log = runtimeLog;

function getAdapterMethodStatus(adapterName) {
    const adapter = window[adapterName];
    return {
        exists: !!adapter,
        isSupported: typeof adapter?.isSupported === 'function',
        upscale: typeof adapter?.upscale === 'function',
        prewarm: typeof adapter?.prewarm === 'function',
        reset: typeof adapter?.reset === 'function'
    };
}

function getRuntimeHookStatus() {
    const imageSrcDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'src');
    return {
        fetch: !!window.fetch?.[NH_SCALER_HOOK_MARK],
        imageConstructor: !!window.Image?.[NH_SCALER_IMAGE_PROXY_MARK],
        imageSrc: !!imageSrcDescriptor?.set?.[NH_SCALER_HOOK_MARK]
    };
}

function getAdapterDiagnosticsStatus(adapterName, capabilityCheck) {
    const adapter = window[adapterName];
    if (typeof adapter?.getDiagnosticsStatus === 'function') {
        return {
            exists: true,
            ...adapter.getDiagnosticsStatus()
        };
    }

    return {
        exists: !!adapter,
        capable: !!capabilityCheck(),
        initialized: false,
        isSupported: typeof adapter?.isSupported === 'function' ? !!adapter.isSupported() : false
    };
}

function runBootDiagnostics(phase) {
    const webGlAdapterStatus = getAdapterMethodStatus('WebGLAdapter');
    const webGpuAdapterStatus = getAdapterMethodStatus('WebGPUAdapter');

    const diagnostics = {
        phase,
        url: window.location.href,
        readyState: document.readyState,
        hasBody: !!document.body,
        readerRoute: isReaderPageUrl(window.location.href),
        hooks: getRuntimeHookStatus(),
        adapters: {
            webgl: webGlAdapterStatus,
            webgpu: webGpuAdapterStatus
        }
    };

    log('boot:diagnostics', diagnostics);

    const missing = [];
    if (!diagnostics.hasBody) missing.push('document.body');
    if (!diagnostics.hooks.fetch) missing.push('fetch-hook');
    if (!diagnostics.hooks.imageConstructor) missing.push('image-constructor-hook');
    if (!diagnostics.hooks.imageSrc) missing.push('image-src-hook');
    if (!webGlAdapterStatus.exists || !webGlAdapterStatus.upscale) missing.push('WebGLAdapter.upscale');
    if (!webGpuAdapterStatus.exists || !webGpuAdapterStatus.upscale) missing.push('WebGPUAdapter.upscale');

    if (missing.length > 0) {
        log('boot:missing-dependencies', { phase, missing });
    }
}

function isForegroundTab() {
    return document.visibilityState === 'visible' && !document.hidden;
}

function isRuntimeMutationSuppressed() {
    return runtimeMutationSuppressed;
}

function getRequestedScaleForBackend(runtimeSettings) {
    const backend = getEffectiveBackend(runtimeSettings);
    const rawScale = backend === 'webgpu' ? runtimeSettings?.selectedWebGpuScale : 2;
    const scale = Number(rawScale);
    return Number.isFinite(scale) && scale > 0 ? scale : 1;
}

function shouldSkipSourceAfterError(error) {
    const message = String(error || '');
    return (
        message.includes('Canvas output is empty after upscale') ||
        message.includes('WebGPU texture limits exceeded') ||
        message.includes('Failed to acquire WebGPU canvas context') ||
        message.includes('Invalid source image dimensions for WebGPU')
    );
}

function resetProcessedRuntimeState() {
    if (typeof resetProcessedMemoryCache === 'function') {
        resetProcessedMemoryCache();
    }

    if (typeof resetBackgroundQueueState === 'function') {
        resetBackgroundQueueState();
    }
}

function restoreAllProcessedImages() {
    document.querySelectorAll('img[data-ai-processed-src]').forEach((img) => {
        restoreOriginalImage(img);
        delete img.dataset.aiJobId;
    });
}

function resetRuntimeAfterSettingsChange() {
    resetBackendRuntimeState();
    resetProcessedRuntimeState();
    restoreAllProcessedImages();
}

function resetRuntimeAfterCacheClear() {
    if (typeof resetBackgroundQueueState === 'function') {
        resetBackgroundQueueState();
    }
}

function getQueueDebugData(sourceUrl) {
    return {
        sourceUrl,
        page: getSourcePageNumber(sourceUrl),
        pageKey: getSourcePageKey(sourceUrl),
        queueSize: backgroundQueue.length,
        processedCount: processedPageKeys.size,
        inFlightCount: inFlightPageKeys.size,
    };
}

function logQueueEvent(label, sourceUrl, extra = {}) {
    log(label, {
        ...getQueueDebugData(sourceUrl),
        foreground: isForegroundTab(),
        backend: backendPreferenceLoaded ? getEffectiveBackend() : 'pending',
        ...extra,
    });
}

function isStaleForegroundJob(img, jobId, sourceUrl, parent) {
    const latestSrc = getImageSourceUrl(img);
    return (
        !isForegroundTab() ||
        !img.isConnected ||
        img.parentElement !== parent ||
        img.dataset.aiJobId !== jobId ||
        latestSrc !== sourceUrl
    );
}

function getStaleForegroundJobReason(img, jobId, sourceUrl, parent) {
    if (!isForegroundTab()) return 'hidden-tab';
    if (!img.isConnected) return 'img-disconnected';
    if (img.parentElement !== parent) return 'parent-changed';
    if (img.dataset.aiJobId !== jobId) return 'jobid-changed';
    const latestSrc = getImageSourceUrl(img);
    if (latestSrc !== sourceUrl) return 'src-changed';
    return null;
}

function getRuntimeDiagnosticsSnapshot() {
    const preferences = getRuntimePreferenceSnapshot();
    let effectiveBackend = 'unknown';
    try {
        effectiveBackend = getEffectiveBackend(preferences);
    } catch (error) {
        effectiveBackend = `error:${String(error)}`;
    }

    const webGlDiagnostics = getAdapterDiagnosticsStatus('WebGLAdapter', () => !!window.Anime4KJS || !!window.Anime4K);
    const webGpuDiagnostics = getAdapterDiagnosticsStatus('WebGPUAdapter', () => !!navigator?.gpu);

    return {
        generatedAt: Date.now(),
        pageUrl: window.location.href,
        preferences,
        effectiveBackend,
        readerRoute: isReaderPageUrl(window.location.href),
        foreground: isForegroundTab(),
        backendPreferenceLoaded,
        hooks: getRuntimeHookStatus(),
        adapters: {
            webgl: webGlDiagnostics,
            webgpu: webGpuDiagnostics
        },
        queue: {
            size: backgroundQueue.length,
            processing: !!backgroundQueueRunPromise,
            processedCount: processedPageKeys.size,
            inFlightCount: inFlightPageKeys.size
        }
    };
}

if (typeof window.fetch === 'function' && !window.fetch[NH_SCALER_HOOK_MARK]) {
    const originalFetch = window.fetch;
    const wrappedFetch = function(...args) {
        const url = args[0];
        const urlString = typeof url === 'string' ? url : url?.url;

        queueBackgroundIfEligible(urlString, 'fetch');

        return originalFetch.apply(this, args);
    };
    wrappedFetch[NH_SCALER_HOOK_MARK] = true;
    wrappedFetch.originalFetch = originalFetch;
    window.fetch = wrappedFetch;
}

const originalImageProto = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'src');
if (originalImageProto && typeof originalImageProto.set === 'function' && !originalImageProto.set[NH_SCALER_HOOK_MARK]) {
    const originalSet = originalImageProto.set;
    const wrappedSet = function(value) {
        queueBackgroundIfEligible(value, 'image-src');
        originalSet.call(this, value);
    };
    wrappedSet[NH_SCALER_HOOK_MARK] = true;

    Object.defineProperty(HTMLImageElement.prototype, 'src', {
        set: wrappedSet,
        get: originalImageProto.get,
        configurable: originalImageProto.configurable,
        enumerable: originalImageProto.enumerable
    });
}

if (typeof window.Image === 'function' && !window.Image[NH_SCALER_IMAGE_PROXY_MARK]) {
    const OriginalImage = window.Image;

    function ProxyImage(...args) {
        const img = new OriginalImage(...args);
        const srcDescriptor = Object.getOwnPropertyDescriptor(img, 'src') || Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'src');
        if (srcDescriptor && srcDescriptor.configurable && typeof srcDescriptor.set === 'function') {
            Object.defineProperty(img, 'src', {
                set(value) {
                    queueBackgroundIfEligible(value, 'image-constructor');
                    srcDescriptor.set.call(this, value);
                },
                get() {
                    return srcDescriptor.get.call(this);
                },
                configurable: true,
                enumerable: srcDescriptor.enumerable
            });
        }
        return img;
    }

    ProxyImage.prototype = OriginalImage.prototype;
    ProxyImage[NH_SCALER_IMAGE_PROXY_MARK] = true;
    ProxyImage.originalImageConstructor = OriginalImage;
    window.Image = ProxyImage;
}

async function processCurrentImage(container) {
    if (!isForegroundTab()) return;

    await Promise.all([backendReadyPromise, webgpuModelReadyPromise, webgpuScaleReadyPromise, onnxModelReadyPromise]);

    const img = selectForegroundImage(container);
    if (!img) return;

    const sourceUrl = getImageSourceUrl(img);
    if (!sourceUrl) return;
    if (img.dataset.aiSkipSource === sourceUrl) return;

    const runtimeSettings = getRuntimePreferenceSnapshot();

    if (getEffectiveBackend(runtimeSettings) === 'off') {
        disableUpscalingForContainer(container, img);
        return;
    }

    const parent = img.parentElement;
    if (!parent) return;

    const sourceWidth = img.naturalWidth || img.width;
    const sourceHeight = img.naturalHeight || img.height;
    const requestedScale = getRequestedScaleForBackend(runtimeSettings);

    if (
        Number.isFinite(sourceWidth) &&
        Number.isFinite(sourceHeight) &&
        sourceWidth > 0 &&
        sourceHeight > 0 &&
        !canCanvasSupportDimensions(sourceWidth, sourceHeight)
    ) {
        img.dataset.aiSkipSource = sourceUrl;
        img.dataset.aiProcessed = 'false';
        disableUpscalingForContainer(container);
        log('process:skip-oversize', {
            sourceUrl,
            page: getSourcePageNumber(sourceUrl),
            sourceWidth,
            sourceHeight,
            targetWidth: Math.max(1, Math.round(sourceWidth * requestedScale)),
            targetHeight: Math.max(1, Math.round(sourceHeight * requestedScale)),
            maxCanvasDimension: getMaxCanvasDimension()
        });
        return;
    }

    if (img.dataset.aiProcessedSrc === sourceUrl && img.dataset.aiBlobUrl) {
        return;
    }

    const cachedBlob = await getProcessedCacheBlob(sourceUrl, runtimeSettings);
    if (cachedBlob) {
        applyProcessedBlobToImage(img, sourceUrl, cachedBlob);
        return;
    }

    // If the background worker is mid-tile on this same page, wait for it instead of
    // launching a duplicate GPU pass. Once it finishes, serve the result from cache.
    const pageKeyForInFlight = getSourcePageKey(sourceUrl);
    if (pageKeyForInFlight && inFlightPageKeys.has(pageKeyForInFlight)) {
        log('process:wait-for-background', { sourceUrl, pageKey: pageKeyForInFlight });
        await waitForInFlightPageKey(pageKeyForInFlight);
        if (!img.isConnected || getImageSourceUrl(img) !== sourceUrl) return;
        const bgResult = await getProcessedCacheBlob(sourceUrl, runtimeSettings);
        if (bgResult) {
            applyProcessedBlobToImage(img, sourceUrl, bgResult);
        }
        return;
    }

    if (img.dataset.aiProcessingSrc === sourceUrl) return;

    const effectiveBackend = getEffectiveBackend(runtimeSettings);
    const shouldSerializeOnnxSource = effectiveBackend === 'onnx';
    if (shouldSerializeOnnxSource && onnxForegroundSourceLocks.has(sourceUrl)) {
        log('process:skip-inflight-onnx', { sourceUrl, page: getSourcePageNumber(sourceUrl) });
        console.warn('[NH] process-skipped (already inflight):', sourceUrl.split('/').slice(-2).join('/'));
        return;
    }

    const jobId = String(++jobCounter);
    img.dataset.aiJobId = jobId;
    img.dataset.aiProcessingSrc = sourceUrl;
    delete img.dataset.aiSkipSource;
    img.dataset.aiProcessed = 'true';
    markForegroundQueueActive();
    if (shouldSerializeOnnxSource) {
        onnxForegroundSourceLocks.add(sourceUrl);
    }
    const page = getSourcePageNumber(sourceUrl);
    if (page == null) {
        log('process:page-missing', { sourceUrl, pageKey: getSourcePageKey(sourceUrl), jobId });
    }
    console.log(`[NH] process-start job:${jobId} page:${page}`, sourceUrl.split('/').slice(-2).join('/'));
    log('process:start', { sourceUrl, page, jobId, backend: effectiveBackend });

    const canvas = document.createElement('canvas');
    canvas.width = 0;
    canvas.height = 0;

    try {
        const tempImg = await loadSourceImage(sourceUrl);
        const loadedWidth = tempImg.naturalWidth || tempImg.width;
        const loadedHeight = tempImg.naturalHeight || tempImg.height;
        const loadedTargetWidth = Math.max(1, Math.round(loadedWidth * requestedScale));
        const loadedTargetHeight = Math.max(1, Math.round(loadedHeight * requestedScale));

        if (!canCanvasSupportDimensions(loadedWidth, loadedHeight)) {
            img.dataset.aiSkipSource = sourceUrl;
            img.dataset.aiProcessed = 'false';
            delete img.dataset.aiProcessingSrc;
            disableUpscalingForContainer(container);
            log('process:skip-oversize', {
                sourceUrl,
                page,
                sourceWidth: loadedWidth,
                sourceHeight: loadedHeight,
                targetWidth: loadedTargetWidth,
                targetHeight: loadedTargetHeight,
                maxCanvasDimension: getMaxCanvasDimension(),
                phase: 'post-load'
            });
            return;
        }

        const latestSrc = getImageSourceUrl(img);
        const staleBeforeUpscaleReason = getStaleForegroundJobReason(img, jobId, sourceUrl, parent);
        if (staleBeforeUpscaleReason) {
            log('process:abort-stale', { sourceUrl, latestSrc, jobId, activeJobId: img.dataset.aiJobId, phase: 'before-upscale' });
            delete img.dataset.aiProcessingSrc;
            return;
        }

        const t3 = performance.now();
        const runInfo = await upscaleWithSelectedBackend(tempImg, canvas, runtimeSettings, { page, sourceUrl });
        const t4 = performance.now();
        log('process:upscale-time', {
            sourceUrl,
            page,
            duration: (t4 - t3).toFixed(2) + 'ms',
            backend: runInfo.backend,
            runMode: runInfo.runMode,
            model: runInfo.model,
        });

        const latestAfterUpscale = getImageSourceUrl(img);
        const staleAfterUpscaleReason = getStaleForegroundJobReason(img, jobId, sourceUrl, parent);
        if (staleAfterUpscaleReason && staleAfterUpscaleReason !== 'hidden-tab') {
            log('process:abort-stale', {
                sourceUrl,
                latestSrc: latestAfterUpscale,
                jobId,
                activeJobId: img.dataset.aiJobId,
                phase: 'after-upscale',
                reason: staleAfterUpscaleReason
            });
            delete img.dataset.aiProcessingSrc;
            return;
        }

        if (canvas.width <= 0 || canvas.height <= 0) {
            throw new Error('Canvas output is empty after upscale');
        }

        const processedBlob = await canvasToBlob(canvas);

        const staleBeforeCacheWriteReason = getStaleForegroundJobReason(img, jobId, sourceUrl, parent);
        if (staleBeforeCacheWriteReason && staleBeforeCacheWriteReason !== 'hidden-tab') {
            const latestAfterBlob = getImageSourceUrl(img);
            log('process:abort-stale', {
                sourceUrl,
                latestSrc: latestAfterBlob,
                jobId,
                activeJobId: img.dataset.aiJobId,
                phase: 'before-cache-write',
                reason: staleBeforeCacheWriteReason
            });
            delete img.dataset.aiProcessingSrc;
            return;
        }

        const cacheWriteOk = await setProcessedCacheBlob(sourceUrl, processedBlob, runtimeSettings);
        log('process:cache-write', {
            sourceUrl,
            page,
            jobId,
            ok: !!cacheWriteOk,
            blobBytes: processedBlob.size
        });

        const staleBeforeApplyReason = getStaleForegroundJobReason(img, jobId, sourceUrl, parent);
        if (staleBeforeApplyReason) {
            log('process:skip-apply', {
                sourceUrl,
                page,
                jobId,
                reason: staleBeforeApplyReason,
                blobBytes: processedBlob.size,
                cacheWriteOk: !!cacheWriteOk
            });
            delete img.dataset.aiProcessingSrc;
            return;
        }

        applyProcessedBlobToImage(img, sourceUrl, processedBlob);
    } catch (err) {
        if (shouldSkipSourceAfterError(err)) {
            img.dataset.aiSkipSource = sourceUrl;
        }
        img.dataset.aiProcessed = 'false';
        delete img.dataset.aiProcessingSrc;
        restoreOriginalImage(img);
        log('process:error', { sourceUrl, page, jobId, error: String(err) });
        console.error(`[NH] process-error job:${jobId} page:${page} skipped:${!!img.dataset.aiSkipSource}`, String(err));
    } finally {
        markForegroundQueueIdle();
        if (shouldSerializeOnnxSource) {
            onnxForegroundSourceLocks.delete(sourceUrl);
        }
    }
}

function isImageNodeOrContainer(node) {
    return node instanceof HTMLImageElement || (node instanceof Element && !!node.querySelector('img'));
}

let observedContainer = null;
let containerObserver = null;
let scheduled = false;
let backgroundDiscoveryTimeoutId = null;

const scheduleProcess = (reason) => {
    if (isRuntimeMutationSuppressed()) return;
    if (scheduled) return;
    scheduled = true;
    queueMicrotask(() => {
        scheduled = false;
        const container = getActiveContainer();
        if (!container) {
            log('observer:run:waiting-for-container', { reason });
            return;
        }

        processCurrentImage(container);
    });
};

function scheduleBackgroundDiscovery(reason) {
    if (isRuntimeMutationSuppressed()) return;
    if (backgroundDiscoveryTimeoutId !== null) {
        clearTimeout(backgroundDiscoveryTimeoutId);
    }

    backgroundDiscoveryTimeoutId = window.setTimeout(() => {
        backgroundDiscoveryTimeoutId = null;
        findAndProcessBackgroundImages();
        scanPerformanceResources();
        log('bg-discovery:run', { reason });
    }, BACKGROUND_DISCOVERY_DEBOUNCE_MS);
}

function attachContainerObserver() {
    if (isRuntimeMutationSuppressed()) return;
    const container = getActiveContainer();
    if (!container) {
        if (containerObserver) {
            containerObserver.disconnect();
            containerObserver = null;
            observedContainer = null;
        }
        return;
    }

    if (container === observedContainer && containerObserver) return;

    if (containerObserver) containerObserver.disconnect();

    observedContainer = container;
    containerObserver = new MutationObserver((mutations) => {
        let shouldProcess = false;
        let reason = 'unknown';

        for (const mutation of mutations) {
            if (mutation.type === 'childList') {
                const imgChange =
                    Array.from(mutation.addedNodes).some((n) => isImageNodeOrContainer(n)) ||
                    Array.from(mutation.removedNodes).some((n) => isImageNodeOrContainer(n));

                if (imgChange) {
                    shouldProcess = true;
                    reason = 'childList:img-change';
                    break;
                }
            }

            if (
                mutation.type === 'attributes' &&
                OBSERVED_IMAGE_ATTRIBUTES.includes(mutation.attributeName || '')
            ) {
                shouldProcess = true;
                reason = `attributes:${mutation.attributeName}`;
                break;
            }
        }

        if (shouldProcess) {
            scheduleProcess(reason);
            scheduleBackgroundDiscovery(reason);
        }
    });

    containerObserver.observe(container, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: OBSERVED_IMAGE_ATTRIBUTES
    });

    log('observer:attached-container', {
        tag: container.tagName,
        className: container.className
    });
    scheduleProcess('container-attached');
    scheduleBackgroundDiscovery('container-attached');
}

const rootObserver = new MutationObserver((mutations) => {
    if (isRuntimeMutationSuppressed()) return;
    attachContainerObserver();
    scheduleBackgroundDiscovery('root-mutation');

    for (const mutation of mutations) {
        if (mutation.type === 'childList') {
            for (const node of mutation.addedNodes) {
                if (node instanceof HTMLImageElement) {
                    const srcUrl = getImageSourceUrl(node);
                    queueBackgroundIfEligible(srcUrl, 'dom-image');
                } else if (node instanceof Element) {
                    const imgs = node.querySelectorAll('img[src], img[data-src]');
                    for (const img of imgs) {
                        const srcUrl = getImageSourceUrl(img);
                        queueBackgroundIfEligible(srcUrl, 'dom-scan');
                    }
                }
            }
        }
    }
});

function startRootObserver() {
    if (!document.body) {
        log('boot:waiting-for-body');
        return;
    }
    rootObserver.observe(document.body, { childList: true, subtree: true });
}

startRootObserver();
if (!document.body) {
    document.addEventListener('DOMContentLoaded', startRootObserver, { once: true });
}

document.addEventListener('visibilitychange', () => {
    if (!isForegroundTab()) {
        return;
    }

    if (isRuntimeMutationSuppressed()) return;

    attachContainerObserver();
    scheduleProcess('visibilitychange');
    scheduleBackgroundDiscovery('visibilitychange');
    runQueueTaskSafely(processBackgroundQueue(), 'visibilitychange-process-loop');
});

if (chrome?.storage?.onChanged) {
    chrome.storage.onChanged.addListener((changes, areaName) => {
        if (areaName !== 'sync') return;

        const didChange = applyRuntimePreferenceStorageChanges(changes);
        if (!didChange) return;

        runtimeMutationSuppressed = true;

        if (runtimeSettingsFlushTimeoutId !== null) {
            clearTimeout(runtimeSettingsFlushTimeoutId);
        }

        runtimeSettingsFlushTimeoutId = window.setTimeout(() => {
            runtimeSettingsFlushTimeoutId = null;

            resetRuntimeAfterSettingsChange();
            const nextSettings = getRuntimePreferenceSnapshot();

            log('settings:changed', nextSettings);

            runtimeMutationSuppressed = false;

            if (nextSettings.selectedEngineBackend === 'off') {
                return;
            }

            scheduleProcess('preset-changed');
            scheduleBackgroundDiscovery('preset-changed');
        }, 50);
    });
}

if (chrome?.runtime?.onMessage) {
    chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
        if (message?.type === GET_DIAGNOSTICS_MESSAGE_TYPE) {
            sendResponse({ ok: true, diagnostics: getRuntimeDiagnosticsSnapshot() });
            return false;
        }

        if (message?.type !== CLEAR_CACHE_MESSAGE_TYPE) return;

        (async () => {
            try {
                const cleared = await clearProcessedCache();
                resetRuntimeAfterCacheClear();
                log('cache:cleared', { cleared });
                sendResponse({ ok: cleared });
            } catch (error) {
                sendResponse({ ok: false, error: String(error) });
            }
        })();

        return true;
    });
}

Promise.all([backendReadyPromise, onnxModelReadyPromise])
    .then(() => {
        return prewarmSelectedBackend();
    })
    .catch(err => {
        log('engine:prewarm-failed', { error: String(err) });
    });

Promise.allSettled([backendReadyPromise, webgpuModelReadyPromise, webgpuScaleReadyPromise, onnxModelReadyPromise, presetReadyPromise])
    .then((results) => {
        runBootDiagnostics(BOOT_DIAGNOSTICS_PHASE_READY);
        const failed = results
            .map((result, index) => ({ result, index }))
            .filter((entry) => entry.result.status === 'rejected')
            .map((entry) => ({
                promiseIndex: entry.index,
                reason: String(entry.result.reason)
            }));

        if (failed.length > 0) {
            log('boot:promise-rejections', { failed });
        }
    });

setInterval(() => {
    if (!isForegroundTab()) return;
    if (isRuntimeMutationSuppressed()) return;

    const hasObserverForConnectedContainer = !!containerObserver && !!observedContainer && observedContainer.isConnected;
    if (hasObserverForConnectedContainer) return;

    attachContainerObserver();
    scheduleProcess('interval');
    scheduleBackgroundDiscovery('interval');
}, SAFETY_INTERVAL_MS);

attachContainerObserver();
scheduleProcess('initial');
scheduleBackgroundDiscovery('initial');
runBootDiagnostics(BOOT_DIAGNOSTICS_PHASE_INITIAL);
