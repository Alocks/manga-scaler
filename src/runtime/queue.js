// Background queue management for batch image upscaling

const processedPageKeys = new Set();
const inFlightPageKeys = new Set();
let backgroundQueue = [];
let backgroundProcessing = false;
let backgroundQueueRunPromise = null;
const seenPerformanceResourceUrls = new Set();

function getNextBackgroundQueueIndex() {
    if (backgroundQueue.length === 0) return -1;

    let bestIndex = 0;
    let bestPage = getSourcePageNumber(backgroundQueue[0]);
    let bestRank = bestPage == null ? Number.POSITIVE_INFINITY : bestPage;

    for (let i = 1; i < backgroundQueue.length; i++) {
        const page = getSourcePageNumber(backgroundQueue[i]);
        const rank = page == null ? Number.POSITIVE_INFINITY : page;
        if (rank < bestRank) {
            bestRank = rank;
            bestIndex = i;
            bestPage = page;
        }
    }

    return bestIndex;
}

function queueBackgroundIfEligible(url, source) {
    if (isRuntimeMutationSuppressed()) return false;
    if (!isReaderPageUrl(window.location.href)) return false;
    if (!isSourceImageUrl(url)) return false;

    const runtimeSettings = getRuntimePreferenceSnapshot();

    if (!backendPreferenceLoaded) {
        logQueueEvent('bg-queue:skip', url, { source, reason: 'backend-pending' });
        return false;
    }

    if (getEffectiveBackend(runtimeSettings) === 'off') {
        logQueueEvent('bg-queue:skip', url, { source, reason: 'backend-off' });
        return false;
    }

    if (!isForegroundTab()) {
        logQueueEvent('bg-queue:skip', url, { source, reason: 'tab-hidden' });
        return false;
    }

    const activeContainer = getActiveContainer();
    const activeImg = activeContainer ? selectForegroundImage(activeContainer) : null;
    const activeUrl = activeImg?.isConnected ? getImageSourceUrl(activeImg) : null;
    if (url === activeUrl) {
        logQueueEvent('bg-queue:skip', url, { source, reason: 'active-image' });
        return false;
    }
    const key = getSourcePageKey(url);
    if (key && processedPageKeys.has(key)) {
        logQueueEvent('bg-queue:skip', url, { source, reason: 'page-already-processed' });
        return false;
    }
    if (key && inFlightPageKeys.has(key)) {
        logQueueEvent('bg-queue:skip', url, { source, reason: 'page-in-flight' });
        return false;
    }
    if (key && backgroundQueue.some(u => getSourcePageKey(u) === key)) {
        logQueueEvent('bg-queue:skip', url, { source, reason: 'page-already-queued' });
        return false;
    }
    if (hasProcessedCacheEntry(url, runtimeSettings)) {
        logQueueEvent('bg-queue:skip', url, { source, reason: 'memory-cache-hit' });
        return false;
    }
    if (backgroundQueue.includes(url)) {
        logQueueEvent('bg-queue:skip', url, { source, reason: 'url-already-queued' });
        return false;
    }

    logQueueEvent('bg-queue:candidate', url, { source });
    preprocessBackgroundImage(url);
    return true;
}

async function preprocessBackgroundImage(sourceUrl) {
    if (!isReaderPageUrl(window.location.href)) return;
    const runtimeSettings = getRuntimePreferenceSnapshot();
    if (!isForegroundTab()) {
        logQueueEvent('bg-queue:skip', sourceUrl, { reason: 'tab-hidden-before-enqueue' });
        return;
    }

    const persistedCachedBlob = await getProcessedCacheBlob(sourceUrl, runtimeSettings);
    if (persistedCachedBlob) {
        logQueueEvent('bg-process:skip-cached', sourceUrl, {
            cache: 'persistent',
            persistedSizeBytes: persistedCachedBlob.size
        });
        return;
    }

    if (backgroundQueue.includes(sourceUrl)) {
        logQueueEvent('bg-queue:skip', sourceUrl, { reason: 'url-already-queued-late' });
        return;
    }

    const activeContainer = getActiveContainer();
    const activeImg = activeContainer ? selectForegroundImage(activeContainer) : null;
    const activeUrl = activeImg?.isConnected ? getImageSourceUrl(activeImg) : null;
    if (sourceUrl === activeUrl) {
        logQueueEvent('bg-process:skip-foreground', sourceUrl);
        return;
    }

    if (!backgroundQueue.includes(sourceUrl)) {
        backgroundQueue.push(sourceUrl);
        logQueueEvent('bg-queue:enqueued', sourceUrl);
    }

    if (!backgroundProcessing) {
        log('bg-queue:kickoff', { queueSize: backgroundQueue.length });
        processBackgroundQueue();
    }
}

async function processBackgroundQueue() {
    if (backgroundQueueRunPromise) {
        return backgroundQueueRunPromise;
    }

    backgroundQueueRunPromise = (async () => {
    if (!isReaderPageUrl(window.location.href)) {
        backgroundQueue = [];
        backgroundProcessing = false;
        return;
    }
    if (!isForegroundTab()) {
        log('bg-queue:paused', { reason: 'tab-hidden', queueSize: backgroundQueue.length });
        return;
    }

    if (backgroundProcessing) return;

    backgroundProcessing = true;

    await Promise.all([backendReadyPromise, webgpuModelReadyPromise, webgpuScaleReadyPromise]);

    if (backgroundQueue.length === 0) return;

    const queueRuntimeSettings = getRuntimePreferenceSnapshot();
    if (getEffectiveBackend(queueRuntimeSettings) === 'off') {
        log('bg-queue:cleared', { reason: 'backend-off', queueSize: backgroundQueue.length });
        backgroundQueue = [];
        return;
    }

    log('bg-queue:start', { queueSize: backgroundQueue.length });

    while (backgroundQueue.length > 0) {
        if (!isForegroundTab()) {
            log('bg-queue:paused', { reason: 'tab-hidden-mid-run', queueSize: backgroundQueue.length });
            break;
        }

        const nextIndex = getNextBackgroundQueueIndex();
        if (nextIndex < 0) break;
        const [sourceUrl] = backgroundQueue.splice(nextIndex, 1);
        logQueueEvent('bg-queue:dequeued', sourceUrl, { nextIndex });

        const itemRuntimeSettings = getRuntimePreferenceSnapshot();

        const persistedCachedBlob = await getProcessedCacheBlob(sourceUrl, itemRuntimeSettings);
        if (persistedCachedBlob) {
            logQueueEvent('bg-process:skip-cached', sourceUrl, {
                cache: 'persistent-after-dequeue',
                persistedSizeBytes: persistedCachedBlob.size
            });
            continue;
        }

        const activeImg = getActiveContainer()?.querySelector('img');
        const activeUrl = activeImg?.isConnected ? getImageSourceUrl(activeImg) : null;
        if (sourceUrl === activeUrl) {
            logQueueEvent('bg-process:skip-now-foreground', sourceUrl);
            continue;
        }

        const page = getSourcePageNumber(sourceUrl);
        const pageKey = getSourcePageKey(sourceUrl);
        if (page == null) {
            logQueueEvent('bg-process:page-missing', sourceUrl);
        }
        if (pageKey) inFlightPageKeys.add(pageKey);

        try {
            const tempImg = await loadSourceImage(sourceUrl);

            const sourceWidth = tempImg.naturalWidth || tempImg.width;
            const sourceHeight = tempImg.naturalHeight || tempImg.height;
            const backend = getEffectiveBackend(itemRuntimeSettings);
            const rawScale = backend === 'webgpu' ? itemRuntimeSettings?.selectedWebGpuScale : 2;
            const requestedScale = Number(rawScale);
            const effectiveScale = Number.isFinite(requestedScale) && requestedScale > 0 ? requestedScale : 1;
            const targetWidth = Math.max(1, Math.round(sourceWidth * effectiveScale));
            const targetHeight = Math.max(1, Math.round(sourceHeight * effectiveScale));

            if (!canCanvasSupportDimensions(sourceWidth, sourceHeight)) {
                if (pageKey) processedPageKeys.add(pageKey);
                logQueueEvent('bg-process:skip-oversize', sourceUrl, {
                    sourceWidth,
                    sourceHeight,
                    targetWidth,
                    targetHeight,
                    maxCanvasDimension: getMaxCanvasDimension(),
                    backend
                });
                continue;
            }

            if (!isForegroundTab()) {
                logQueueEvent('bg-process:skip-hidden-after-load', sourceUrl);
                continue;
            }

            const bgCanvas = document.createElement('canvas');
            const t3 = performance.now();
            const runInfo = await upscaleWithSelectedBackend(tempImg, bgCanvas, itemRuntimeSettings);
            const t4 = performance.now();
            log('bg-process:upscale-time', {
                sourceUrl,
                page,
                duration: (t4 - t3).toFixed(2) + 'ms',
                backend: runInfo.backend,
                runMode: runInfo.runMode,
                model: runInfo.model
            });

            if (pageKey) processedPageKeys.add(pageKey);

            if (bgCanvas.width <= 0 || bgCanvas.height <= 0) {
                throw new Error('Canvas output is empty after upscale');
            }

            const processedBlob = await canvasToBlob(bgCanvas);

            if (!isForegroundTab()) {
                logQueueEvent('bg-process:skip-hidden-before-cache', sourceUrl);
                continue;
            }

            await setProcessedCacheBlob(sourceUrl, processedBlob, itemRuntimeSettings);
            logQueueEvent('bg-process:cached', sourceUrl, {
                width: bgCanvas.width,
                height: bgCanvas.height,
            });
        } catch (err) {
            log('bg-process:error', { sourceUrl, page, error: String(err) });
        } finally {
            if (pageKey) inFlightPageKeys.delete(pageKey);
        }

        await new Promise(resolve => setTimeout(resolve, 10));
    }

    })();

    try {
        return await backgroundQueueRunPromise;
    } finally {
        backgroundProcessing = false;
        backgroundQueueRunPromise = null;
        log('bg-queue:idle', { queueSize: backgroundQueue.length });
    }
}

function findAndProcessBackgroundImages() {
    if (!isReaderPageUrl(window.location.href)) return;
    if (!isForegroundTab()) return;

    const allImages = Array.from(document.querySelectorAll('img[src], img[data-src]'));
    const activeContainer = getActiveContainer();
    const activeImg = activeContainer ? selectForegroundImage(activeContainer) : null;
    const activeUrl = activeImg?.isConnected ? getImageSourceUrl(activeImg) : null;

    let scannedSourceImages = 0;
    let queued = 0;
    const maxQueuedPerScan = 3;

    for (const img of allImages) {
        const srcUrl = getImageSourceUrl(img);
        if (!srcUrl || srcUrl === activeUrl) continue;

        if (isSourceImageUrl(srcUrl)) {
            scannedSourceImages++;
            const enqueued = queueBackgroundIfEligible(srcUrl, 'scan');
            if (enqueued) {
                queued++;
            }

            if (queued >= maxQueuedPerScan) break;
        }
    }

    if (scannedSourceImages > 0) {
        log('bg-process:found', {
            scannedSourceImages,
            queued,
            queueSize: backgroundQueue.length
        });
    }
}

function scanPerformanceResources() {
    if (!performance?.getEntriesByType) return;

    const activeAdapter = getActiveSourceAdapter(window.location.href);
    const deferSeenUntilBackendReady = !backendPreferenceLoaded && activeAdapter?.id === 'mangadex';

    const resources = performance.getEntriesByType('resource');
    for (const entry of resources) {
        const url = entry?.name;
        if (typeof url !== 'string' || !url) continue;
        if (seenPerformanceResourceUrls.has(url)) continue;

        const sourceImage = isSourceImageUrl(url);
        if (deferSeenUntilBackendReady && sourceImage) {
            queueBackgroundIfEligible(url, `perf:${entry.initiatorType || 'unknown'}`);
            continue;
        }

        seenPerformanceResourceUrls.add(url);
        if (sourceImage) {
            queueBackgroundIfEligible(url, `perf:${entry.initiatorType || 'unknown'}`);
        }
    }
}
