// Background queue management for batch image upscaling

const processedPageKeys = new Set();
const inFlightPageKeys = new Set();
let backgroundQueue = [];
let backgroundQueueRunPromise = null;
const queuedSourceUrls = new Set();
const queuedPageKeyCounts = new Map();
const seenPerformanceResourceUrls = new Set();
const BACKGROUND_QUEUE_YIELD_MS = 10;
const BACKGROUND_QUEUE_MAX_CONCURRENCY = 1;
const BACKGROUND_QUEUE_KICKOFF_DELAY_MS = 40;
let backgroundQueueResetVersion = 0;
let backgroundQueueKickoffTimerId = null;
let foregroundQueueActiveCount = 0;
const foregroundQueueIdleWaiters = [];

function isBackgroundQueueRunning() {
    return !!backgroundQueueRunPromise;
}

function runQueueTaskSafely(promise, task, extra = {}) {
    if (!promise || typeof promise.catch !== 'function') return;
    promise.catch((error) => {
        log('bg-queue:task-error', {
            task,
            error: String(error),
            ...extra
        });
    });
}

function waitForInFlightPageKey(pageKey) {
    return new Promise((resolve) => {
        function check() {
            if (!inFlightPageKeys.has(pageKey)) { resolve(); return; }
            window.setTimeout(check, 50);
        }
        check();
    });
}

function isForegroundQueueActive() {
    return foregroundQueueActiveCount > 0;
}

function markForegroundQueueActive() {
    foregroundQueueActiveCount += 1;
}

function markForegroundQueueIdle() {
    if (foregroundQueueActiveCount > 0) {
        foregroundQueueActiveCount -= 1;
    }

    if (foregroundQueueActiveCount === 0) {
        while (foregroundQueueIdleWaiters.length > 0) {
            const resolve = foregroundQueueIdleWaiters.shift();
            if (typeof resolve === 'function') {
                resolve();
            }
        }
    }
}

function waitForForegroundQueueIdle() {
    if (!isForegroundQueueActive()) {
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        foregroundQueueIdleWaiters.push(resolve);
    });
}

function scheduleBackgroundQueueKickoff() {
    if (isBackgroundQueueRunning()) return;
    if (backgroundQueueKickoffTimerId !== null) return;
    backgroundQueueKickoffTimerId = window.setTimeout(() => {
        backgroundQueueKickoffTimerId = null;
        if (!isBackgroundQueueRunning()) {
            log('bg-queue:kickoff', { queueSize: backgroundQueue.length });
            runQueueTaskSafely(processBackgroundQueue(), 'process-loop');
        }
    }, BACKGROUND_QUEUE_KICKOFF_DELAY_MS);
}

function resetBackgroundQueueState() {
    backgroundQueueResetVersion++;
    const hasActiveRun = isBackgroundQueueRunning();

    if (backgroundQueueKickoffTimerId !== null) {
        clearTimeout(backgroundQueueKickoffTimerId);
        backgroundQueueKickoffTimerId = null;
    }

    foregroundQueueActiveCount = 0;
    while (foregroundQueueIdleWaiters.length > 0) {
        const resolve = foregroundQueueIdleWaiters.shift();
        if (typeof resolve === 'function') {
            resolve();
        }
    }

    processedPageKeys.clear();
    inFlightPageKeys.clear();
    clearBackgroundQueue();
    seenPerformanceResourceUrls.clear();

    if (!hasActiveRun) {
        backgroundQueueRunPromise = null;
    }
}

function clearBackgroundQueue() {
    backgroundQueue = [];
    queuedSourceUrls.clear();
    queuedPageKeyCounts.clear();
}

function enqueueBackgroundQueueUrl(sourceUrl) {
    backgroundQueue.push(sourceUrl);
    queuedSourceUrls.add(sourceUrl);

    const pageKey = getSourcePageKey(sourceUrl);
    if (!pageKey) return;
    queuedPageKeyCounts.set(pageKey, (queuedPageKeyCounts.get(pageKey) || 0) + 1);
}

function dequeueBackgroundQueueUrlAt(index) {
    if (index < 0 || index >= backgroundQueue.length) return null;

    const [sourceUrl] = backgroundQueue.splice(index, 1);
    queuedSourceUrls.delete(sourceUrl);

    const pageKey = getSourcePageKey(sourceUrl);
    if (pageKey) {
        const nextCount = (queuedPageKeyCounts.get(pageKey) || 0) - 1;
        if (nextCount > 0) {
            queuedPageKeyCounts.set(pageKey, nextCount);
        } else {
            queuedPageKeyCounts.delete(pageKey);
        }
    }

    return sourceUrl;
}

function getNextBackgroundQueueIndex() {
    if (backgroundQueue.length === 0) return -1;

    let bestIndex = 0;
    const firstPage = getSourcePageNumber(backgroundQueue[0]);
    let bestRank = firstPage == null ? Number.POSITIVE_INFINITY : firstPage;

    for (let i = 1; i < backgroundQueue.length; i++) {
        const page = getSourcePageNumber(backgroundQueue[i]);
        const rank = page == null ? Number.POSITIVE_INFINITY : page;
        if (rank < bestRank) {
            bestRank = rank;
            bestIndex = i;
        }
    }

    return bestIndex;
}

function getActiveForegroundSourceUrl() {
    const activeContainer = getActiveContainer();
    const activeImg = activeContainer ? selectForegroundImage(activeContainer) : null;
    return activeImg?.isConnected ? getImageSourceUrl(activeImg) : null;
}

function getQueueConflictReason(url, runtimeSettings) {
    const activeUrl = getActiveForegroundSourceUrl();
    if (url === activeUrl) return 'active-image';

    const key = getSourcePageKey(url);
    if (key && processedPageKeys.has(key)) return 'page-already-processed';
    if (key && inFlightPageKeys.has(key)) return 'page-in-flight';
    if (key && queuedPageKeyCounts.has(key)) return 'page-already-queued';
    if (hasProcessedCacheEntry(url, runtimeSettings)) return 'memory-cache-hit';
    if (queuedSourceUrls.has(url)) return 'url-already-queued';

    return null;
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

    const conflictReason = getQueueConflictReason(url, runtimeSettings);
    if (conflictReason) {
        logQueueEvent('bg-queue:skip', url, { source, reason: conflictReason });
        return false;
    }

    logQueueEvent('bg-queue:candidate', url, { source });
    runQueueTaskSafely(preprocessBackgroundImage(url), 'preprocess', { sourceUrl: url, source });
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

    const conflictReason = getQueueConflictReason(sourceUrl, runtimeSettings);
    if (conflictReason) {
        logQueueEvent('bg-queue:skip', sourceUrl, { reason: `${conflictReason}-late` });
        return;
    }

    enqueueBackgroundQueueUrl(sourceUrl);
    logQueueEvent('bg-queue:enqueued', sourceUrl);

    scheduleBackgroundQueueKickoff();
}

function getBackgroundExecutionLane(slotIndex) {
    const normalizedSlot = Number.isFinite(slotIndex) ? Math.max(0, Math.floor(slotIndex)) : 0;
    if (normalizedSlot <= 0) {
        return 'background';
    }

    return `background-${normalizedSlot}`;
}

async function processBackgroundQueueItem(sourceUrl, runResetVersion, executionLane) {
    const itemRuntimeSettings = getRuntimePreferenceSnapshot();

    const persistedCachedBlob = await getProcessedCacheBlob(sourceUrl, itemRuntimeSettings);
    if (persistedCachedBlob) {
        logQueueEvent('bg-process:skip-cached', sourceUrl, {
            cache: 'persistent-after-dequeue',
            persistedSizeBytes: persistedCachedBlob.size
        });
        return;
    }

    const activeUrl = getActiveForegroundSourceUrl();
    if (sourceUrl === activeUrl) {
        logQueueEvent('bg-process:skip-now-foreground', sourceUrl);
        return;
    }

    const page = getSourcePageNumber(sourceUrl);
    const pageKey = getSourcePageKey(sourceUrl);
    if (page == null) {
        logQueueEvent('bg-process:page-missing', sourceUrl);
    }
    if (pageKey) inFlightPageKeys.add(pageKey);

    try {
        const tempImg = await loadSourceImage(sourceUrl);

        if (runResetVersion !== backgroundQueueResetVersion) {
            logQueueEvent('bg-process:cancelled-after-load', sourceUrl);
            return;
        }

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
            return;
        }

        if (!isForegroundTab()) {
            logQueueEvent('bg-process:skip-hidden-after-load', sourceUrl);
            return;
        }

        const bgCanvas = document.createElement('canvas');
        const t3 = performance.now();
        const runInfo = await upscaleWithSelectedBackend(tempImg, bgCanvas, itemRuntimeSettings, {
            lane: executionLane
        });
        const t4 = performance.now();
        log('bg-process:upscale-time', {
            sourceUrl,
            page,
            lane: executionLane,
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

        if (runResetVersion !== backgroundQueueResetVersion) {
            logQueueEvent('bg-process:cancelled-before-cache', sourceUrl);
            return;
        }

        if (!isForegroundTab()) {
            logQueueEvent('bg-process:skip-hidden-before-cache', sourceUrl);
            return;
        }

        await setProcessedCacheBlob(sourceUrl, processedBlob, itemRuntimeSettings);
        logQueueEvent('bg-process:cached', sourceUrl, {
            width: bgCanvas.width,
            height: bgCanvas.height,
        });
    } catch (err) {
        log('bg-process:error', { sourceUrl, page, lane: executionLane, error: String(err) });
    } finally {
        if (pageKey) inFlightPageKeys.delete(pageKey);
    }
}

async function processBackgroundQueue() {
    if (isBackgroundQueueRunning()) {
        return backgroundQueueRunPromise;
    }

    backgroundQueueRunPromise = (async () => {
    const runResetVersion = backgroundQueueResetVersion;

    if (!isReaderPageUrl(window.location.href)) {
        clearBackgroundQueue();
        return;
    }
    if (!isForegroundTab()) {
        log('bg-queue:paused', { reason: 'tab-hidden', queueSize: backgroundQueue.length });
        return;
    }

    await Promise.all([backendReadyPromise, webgpuModelReadyPromise, webgpuScaleReadyPromise, onnxModelReadyPromise]);

    if (runResetVersion !== backgroundQueueResetVersion) {
        log('bg-queue:cancelled', { reason: 'reset-before-start' });
        return;
    }

    if (backgroundQueue.length === 0) return;

    const queueRuntimeSettings = getRuntimePreferenceSnapshot();
    if (getEffectiveBackend(queueRuntimeSettings) === 'off') {
        log('bg-queue:cleared', { reason: 'backend-off', queueSize: backgroundQueue.length });
        clearBackgroundQueue();
        return;
    }

    log('bg-queue:start', { queueSize: backgroundQueue.length });

    const maxConcurrency = Math.max(1, Math.floor(BACKGROUND_QUEUE_MAX_CONCURRENCY));
    const activeTasks = new Set();
    let nextLaneSlot = 0;
    let pausedForHiddenTab = false;

    while (backgroundQueue.length > 0 || activeTasks.size > 0) {
        if (runResetVersion !== backgroundQueueResetVersion) {
            log('bg-queue:cancelled', {
                reason: 'reset-mid-run',
                queueSize: backgroundQueue.length,
                activeTasks: activeTasks.size
            });
            break;
        }

        if (!isForegroundTab()) {
            pausedForHiddenTab = true;
        }

        if (!pausedForHiddenTab && activeTasks.size === 0 && backgroundQueue.length > 0 && isForegroundQueueActive()) {
            log('bg-queue:waiting-for-foreground', {
                queueSize: backgroundQueue.length
            });
            await waitForForegroundQueueIdle();
            continue;
        }

        while (!pausedForHiddenTab && activeTasks.size < maxConcurrency && backgroundQueue.length > 0) {
            const nextIndex = getNextBackgroundQueueIndex();
            if (nextIndex < 0) {
                break;
            }

            const sourceUrl = dequeueBackgroundQueueUrlAt(nextIndex);
            if (!sourceUrl) {
                continue;
            }

            const executionLane = getBackgroundExecutionLane(nextLaneSlot % maxConcurrency);
            nextLaneSlot += 1;

            logQueueEvent('bg-queue:dequeued', sourceUrl, {
                nextIndex,
                executionLane,
                activeTasks: activeTasks.size
            });

            let taskPromise;
            taskPromise = processBackgroundQueueItem(sourceUrl, runResetVersion, executionLane)
                .finally(() => {
                    activeTasks.delete(taskPromise);
                });

            activeTasks.add(taskPromise);
        }

        if (activeTasks.size === 0) {
            break;
        }

        await Promise.race(activeTasks);

        if (BACKGROUND_QUEUE_YIELD_MS > 0) {
            await new Promise((resolve) => setTimeout(resolve, BACKGROUND_QUEUE_YIELD_MS));
        }
    }

    if (pausedForHiddenTab && backgroundQueue.length > 0) {
        log('bg-queue:paused', {
            reason: 'tab-hidden-mid-run',
            queueSize: backgroundQueue.length
        });
    }

    })();

    // After all background jobs are done, dispose ONNX background worker/session to free VRAM
    try {
        if (typeof window.resetOnnxWorkerState === 'function') {
            window.resetOnnxWorkerState('background');
        }
    } catch (err) {
        log('bg-queue:onnx-dispose-error', { error: String(err) });
    }

    try {
        return await backgroundQueueRunPromise;
    } finally {
        backgroundQueueRunPromise = null;
        log('bg-queue:idle', { queueSize: backgroundQueue.length });
    }
}

function findAndProcessBackgroundImages() {
    if (!isReaderPageUrl(window.location.href)) return;
    if (!isForegroundTab()) return;

    const allImages = Array.from(document.querySelectorAll('img[src], img[data-src]'));
    const activeUrl = getActiveForegroundSourceUrl();

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
