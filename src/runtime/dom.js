// DOM manipulation for image rendering and canvas management

const IMAGE_LOAD_TIMEOUT_MS = 10000;
const HARD_MAX_CANVAS_DIMENSION = 16384;
let cachedMaxCanvasDimension = null;
const NH_SCALER_CANVAS_MARK = '__nhScalerCanvas__';

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

function hideOriginal(img) {
    img.style.setProperty('display', 'none', 'important');
    img.style.setProperty('visibility', 'hidden', 'important');
}

function showOriginal(img) {
    img.style.removeProperty('display');
    img.style.removeProperty('visibility');
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

    img.dataset.aiProcessed = 'false';
    delete img.dataset.aiProcessedSrc;
    delete img.dataset.aiProcessingSrc;
    delete img.dataset.aiJobId;
}

function applyProcessedBlobToImage(img, sourceUrl, processedBlob) {
    if (!(img instanceof HTMLImageElement)) return null;
    if (!(processedBlob instanceof Blob)) return null;

    rememberOriginalImageState(img, sourceUrl);
    revokeProcessedBlobUrl(img);

    const objectUrl = URL.createObjectURL(processedBlob);
    img.dataset.aiBlobUrl = objectUrl;
    img.dataset.aiProcessed = 'true';
    img.dataset.aiProcessedSrc = sourceUrl;
    delete img.dataset.aiProcessingSrc;
    delete img.dataset.aiJobId;

    img.removeAttribute('srcset');
    img.removeAttribute('sizes');
    img.src = objectUrl;

    return objectUrl;
}

function isInjectedCanvas(node) {
    return node instanceof HTMLCanvasElement && node[NH_SCALER_CANVAS_MARK] === true;
}

function getInjectedCanvases(root) {
    if (!(root instanceof Element)) return [];
    return Array.from(root.querySelectorAll('canvas')).filter((canvas) => isInjectedCanvas(canvas));
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

    const canvases = getInjectedCanvases(container);
    for (const canvas of canvases) {
        canvas.remove();
    }
}

function syncCanvasPresentation(canvas, img) {
    if (!(canvas instanceof HTMLCanvasElement) || !(img instanceof HTMLImageElement)) return;

    const activeAdapter = typeof getActiveSourceAdapter === 'function' ? getActiveSourceAdapter() : null;
    const mirrorSourcePresentation = activeAdapter?.mirrorSourceImagePresentation !== false;

    canvas[NH_SCALER_CANVAS_MARK] = true;

    if (mirrorSourcePresentation) {
        canvas.style.width = img.style.width || '';
        canvas.style.height = img.style.height || '';
        canvas.style.maxWidth = img.style.maxWidth || '';
        canvas.style.maxHeight = img.style.maxHeight || '';
        canvas.style.minWidth = img.style.minWidth || '';
        canvas.style.minHeight = img.style.minHeight || '';
        canvas.style.objectFit = img.style.objectFit || '';
        canvas.style.objectPosition = img.style.objectPosition || '';
    } else {
        canvas.style.removeProperty('width');
        canvas.style.removeProperty('height');
        canvas.style.removeProperty('max-width');
        canvas.style.removeProperty('max-height');
        canvas.style.removeProperty('min-width');
        canvas.style.removeProperty('min-height');
        canvas.style.removeProperty('object-fit');
        canvas.style.removeProperty('object-position');
    }
}

function getCanvasForImage(img) {
    if (!(img instanceof HTMLImageElement)) return null;
    const nextSibling = img.nextElementSibling;
    if (isInjectedCanvas(nextSibling)) {
        return nextSibling;
    }

    const previousSibling = img.previousElementSibling;
    if (isInjectedCanvas(previousSibling)) {
        return previousSibling;
    }

    const parent = img.parentElement;
    if (!parent) return null;

    const canvases = getInjectedCanvases(parent);
    return canvases.length === 1 ? canvases[0] : null;
}

function ensureCanvas(parent, sourceImg) {
    const canvas = document.createElement('canvas');
    canvas.width = 0;
    canvas.height = 0;
    canvas[NH_SCALER_CANVAS_MARK] = true;
    if (sourceImg instanceof HTMLImageElement) {
        syncCanvasPresentation(canvas, sourceImg);
    }

    return canvas;
}

function hasRenderedCanvasForSource(img, canvas, sourceUrl) {
    return (
        !!canvas &&
        canvas.dataset.aiSourceUrl === sourceUrl &&
        canvas.width > 0 &&
        canvas.height > 0 &&
        img.dataset.aiProcessedSrc === sourceUrl
    );
}

function reconcile(container) {
    if (getEffectiveBackend() === 'off') {
        const activeImg = selectForegroundImage(container);
        if (activeImg && (activeImg.dataset.aiBlobUrl || activeImg.dataset.aiProcessedSrc || activeImg.dataset.aiProcessingSrc)) {
            disableUpscalingForContainer(container, activeImg);
        }
        return;
    }

    const activeImg = selectForegroundImage(container);
    const imgs = container.querySelectorAll('img');
    for (const img of imgs) {
        const sourceUrl = getImageSourceUrl(img);
        if (!sourceUrl) continue;

        const parent = img.parentElement;
        if (!parent) continue;

        const canvas = getCanvasForImage(img);
        if (hasRenderedCanvasForSource(img, canvas, sourceUrl)) {
            if (img === activeImg) {
                syncCanvasPresentation(canvas, img);
                canvas.style.display = 'block';
                canvas.style.visibility = 'visible';
                hideOriginal(img);
            } else {
                canvas.style.display = 'none';
                canvas.style.visibility = 'hidden';
            }
        }
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

function canvasToBlob(canvas, type = 'image/png', quality) {
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
