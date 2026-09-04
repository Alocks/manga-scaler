// DOM manipulation for image rendering and canvas management

const IMAGE_LOAD_TIMEOUT_MS = 10000;
const HARD_MAX_CANVAS_DIMENSION = 16384;
let cachedMaxCanvasDimension = null;

function getMaxCanvasDimension() {
    if (cachedMaxCanvasDimension !== null) return cachedMaxCanvasDimension;

    // Keep a conservative cap. Some browsers accept larger width/height values
    // but still fail to render large surfaces reliably on draw operations.
    cachedMaxCanvasDimension = HARD_MAX_CANVAS_DIMENSION;
    return cachedMaxCanvasDimension;
}

function canCanvasSupportDimensions(width, height) {
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        return false;
    }

    const max = getMaxCanvasDimension();
    return width <= max && height <= max;
}

function getImageSourceUrl(img) {
    if (!(img instanceof HTMLImageElement)) return null;
    if (img.dataset.aiBlobUrl && img.dataset.aiProcessedSrc) {
        return img.dataset.aiProcessedSrc;
    }
    return img.currentSrc || img.src || img.dataset.src || null;
}

function rememberOriginalImageState(img, sourceUrl) {
    if (!(img instanceof HTMLImageElement)) return;
    if (!img.dataset.aiOriginalSrc && sourceUrl) {
        img.dataset.aiOriginalSrc = sourceUrl;
    }
    if (!Object.prototype.hasOwnProperty.call(img.dataset, 'aiOriginalSrcset')) {
        img.dataset.aiOriginalSrcset = img.getAttribute('srcset') || '';
    }
    if (!Object.prototype.hasOwnProperty.call(img.dataset, 'aiOriginalSizes')) {
        img.dataset.aiOriginalSizes = img.getAttribute('sizes') || '';
    }
    if (!Object.prototype.hasOwnProperty.call(img.dataset, 'aiOriginalImageRendering')) {
        img.dataset.aiOriginalImageRendering = img.style.getPropertyValue('image-rendering') || '';
    }
    if (!Object.prototype.hasOwnProperty.call(img.dataset, 'aiOriginalImageRenderingPriority')) {
        img.dataset.aiOriginalImageRenderingPriority = img.style.getPropertyPriority('image-rendering') || '';
    }
}

function revokeProcessedBlobUrl(img) {
    if (!(img instanceof HTMLImageElement)) return;
    const blobUrl = img.dataset.aiBlobUrl;
    if (!blobUrl) return;

    try {
        URL.revokeObjectURL(blobUrl);
    } catch {
        // Ignore revocation failures.
    }

    delete img.dataset.aiBlobUrl;
}

function restoreOriginalImage(img) {
    if (!(img instanceof HTMLImageElement)) return;

    revokeProcessedBlobUrl(img);

    const originalSrc = img.dataset.aiOriginalSrc;
    if (originalSrc) {
        if (img.dataset.aiOriginalSrcset) {
            img.setAttribute('srcset', img.dataset.aiOriginalSrcset);
        } else {
            img.removeAttribute('srcset');
        }

        if (img.dataset.aiOriginalSizes) {
            img.setAttribute('sizes', img.dataset.aiOriginalSizes);
        } else {
            img.removeAttribute('sizes');
        }

        img.src = originalSrc;
    }

    const originalImageRendering = img.dataset.aiOriginalImageRendering || '';
    const originalImageRenderingPriority = img.dataset.aiOriginalImageRenderingPriority || '';
    if (originalImageRendering) {
        img.style.setProperty('image-rendering', originalImageRendering, originalImageRenderingPriority);
    } else {
        img.style.removeProperty('image-rendering');
    }

    img.dataset.aiProcessed = 'false';
    delete img.dataset.aiProcessedSrc;
    delete img.dataset.aiProcessingSrc;
    delete img.dataset.aiJobId;
}

async function applyProcessedBlobToImage(img, sourceUrl, processedBlob) {
    if (!(img instanceof HTMLImageElement)) return null;
    if (!(processedBlob instanceof Blob)) return null;

    rememberOriginalImageState(img, sourceUrl);

    let objectUrl = img.dataset.aiBlobUrl || null;
    if (!objectUrl) {
        objectUrl = URL.createObjectURL(processedBlob);
        img.dataset.aiBlobUrl = objectUrl;
    }

    img.dataset.aiProcessed = 'true';
    img.dataset.aiProcessedSrc = sourceUrl;
    delete img.dataset.aiProcessingSrc;
    delete img.dataset.aiJobId;

    img.removeAttribute('srcset');
    img.removeAttribute('sizes');
    img.style.setProperty('image-rendering', '-webkit-optimize-contrast');
    if (img.currentSrc !== objectUrl && img.src !== objectUrl) {
        img.src = objectUrl;
    }

    return objectUrl;
}

// Synchronous apply path used from the img.src setter hook.
// When the memory cache already has the processed blob, this injects it
// directly so the original image URL is never loaded — eliminating the
// flash of the original image during page navigation.
function applyCachedBlobFromSrcHook(img, sourceUrl, cachedBlob) {
    let objectUrl = img.dataset.aiBlobUrl || null;
    if (!objectUrl) {
        objectUrl = URL.createObjectURL(cachedBlob);
        img.dataset.aiBlobUrl = objectUrl;
    }

    // Force-update the original-src reference to the new page URL.
    img.dataset.aiOriginalSrc = sourceUrl;
    if (!Object.prototype.hasOwnProperty.call(img.dataset, 'aiOriginalSrcset')) {
        img.dataset.aiOriginalSrcset = img.getAttribute('srcset') || '';
    }
    if (!Object.prototype.hasOwnProperty.call(img.dataset, 'aiOriginalSizes')) {
        img.dataset.aiOriginalSizes = img.getAttribute('sizes') || '';
    }
    if (!Object.prototype.hasOwnProperty.call(img.dataset, 'aiOriginalImageRendering')) {
        img.dataset.aiOriginalImageRendering = img.style.getPropertyValue('image-rendering') || '';
    }
    if (!Object.prototype.hasOwnProperty.call(img.dataset, 'aiOriginalImageRenderingPriority')) {
        img.dataset.aiOriginalImageRenderingPriority = img.style.getPropertyPriority('image-rendering') || '';
    }

    img.dataset.aiProcessed = 'true';
    img.dataset.aiProcessedSrc = sourceUrl;
    delete img.dataset.aiProcessingSrc;
    delete img.dataset.aiJobId;

    img.removeAttribute('srcset');
    img.removeAttribute('sizes');
    img.style.setProperty('image-rendering', '-webkit-optimize-contrast');

    return objectUrl;
}

function disableUpscalingForContainer(container, activeImg = null) {
    if (activeImg instanceof HTMLImageElement) {
        if (activeImg.dataset.aiBlobUrl || activeImg.dataset.aiProcessedSrc || activeImg.dataset.aiProcessingSrc) {
            restoreOriginalImage(activeImg);
        }
        return;
    }

    const imgs = container.querySelectorAll('img');
    for (const img of imgs) {
        restoreOriginalImage(img);
    }
}

function getSourceLoadCandidates(sourceUrl) {
    if (typeof sourceUrl !== 'string' || !sourceUrl) return [];

    const candidates = [sourceUrl];

    const sourceLookup = typeof getSourceAdapterForImageUrl === 'function'
        ? getSourceAdapterForImageUrl(sourceUrl)
        : null;
    const adapter = sourceLookup?.adapter || (typeof getActiveSourceAdapter === 'function' ? getActiveSourceAdapter() : null);

    if (adapter && typeof adapter.getImageLoadCandidates === 'function') {
        const adapterCandidates = adapter.getImageLoadCandidates(sourceUrl, sourceLookup?.parsed);
        if (Array.isArray(adapterCandidates)) {
            candidates.push(...adapterCandidates);
        }
    }

    return [...new Set(candidates.filter((candidate) => typeof candidate === 'string' && !!candidate))];
}

function loadSourceImage(sourceUrl) {
    return new Promise((resolve, reject) => {
        if (typeof sourceUrl !== 'string' || !sourceUrl) {
            reject(new Error(`Invalid source image URL: ${String(sourceUrl)}`));
            return;
        }

        const loadCandidates = getSourceLoadCandidates(sourceUrl);
        if (loadCandidates.length === 0) {
            reject(new Error(`No load candidates resolved for source image URL: ${sourceUrl}`));
            return;
        }

        const tempImg = new Image();
        let settled = false;
        let timeoutId = null;
        let attemptIndex = 0;

        const cleanup = () => {
            tempImg.onload = null;
            tempImg.onerror = null;
            if (timeoutId !== null) {
                clearTimeout(timeoutId);
                timeoutId = null;
            }
        };

        const settleResolve = () => {
            if (settled) return;
            settled = true;
            cleanup();
            resolve(tempImg);
        };

        const settleReject = (error) => {
            if (settled) return;
            settled = true;
            cleanup();
            reject(error);
        };

        const attemptNextCandidate = () => {
            if (settled) return;

            if (timeoutId !== null) {
                clearTimeout(timeoutId);
                timeoutId = null;
            }

            if (attemptIndex >= loadCandidates.length) {
                settleReject(new Error(`Failed to load source image: ${sourceUrl} (tried ${loadCandidates.join(', ')})`));
                return;
            }

            const candidateUrl = loadCandidates[attemptIndex++];

            tempImg.crossOrigin = 'anonymous';
            tempImg.onload = () => settleResolve();
            tempImg.onerror = () => {
                attemptNextCandidate();
            };

            timeoutId = window.setTimeout(() => {
                tempImg.src = '';
                attemptNextCandidate();
            }, IMAGE_LOAD_TIMEOUT_MS);

            tempImg.src = candidateUrl;
        };

        attemptNextCandidate();
    });
}

function canvasToBlob(canvas, type = 'image/webp', quality = 0.98) {
    return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) {
                resolve(blob);
                return;
            }
            reject(new Error('Canvas toBlob returned null'));
        }, type, quality);
    });
}
