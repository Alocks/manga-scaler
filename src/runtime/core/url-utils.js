// Source adapter dispatch helpers

function getRegisteredSourceAdapters() {
    return Array.isArray(window.NHScalerSourceAdapters) ? window.NHScalerSourceAdapters : [];
}

function getActiveSourceAdapter(pageUrl = window.location.href) {
    const adapters = getRegisteredSourceAdapters();
    return adapters.find((adapter) => typeof adapter?.isReaderPageUrl === 'function' && adapter.isReaderPageUrl(pageUrl)) || null;
}

function getSourceAdapterForImageUrl(url) {
    const activeAdapter = getActiveSourceAdapter(window.location.href);
    if (activeAdapter && typeof activeAdapter?.parseImageUrl === 'function') {
        const parsed = activeAdapter.parseImageUrl(url);
        if (parsed) return { adapter: activeAdapter, parsed };
    }

    const adapters = getRegisteredSourceAdapters();
    for (const adapter of adapters) {
        if (adapter === activeAdapter || typeof adapter?.parseImageUrl !== 'function') continue;
        const parsed = adapter.parseImageUrl(url);
        if (parsed) return { adapter, parsed };
    }

    return null;
}

function isReaderPageUrl(url) {
    return !!getActiveSourceAdapter(url);
}

function isSourceImageUrl(url) {
    return !!getSourceAdapterForImageUrl(url);
}

function getSourcePageKey(url) {
    return getSourceAdapterForImageUrl(url)?.parsed?.pageKey || null;
}

function getSourcePageNumber(url) {
    return getSourceAdapterForImageUrl(url)?.parsed?.page || null;
}

function getReaderImageCandidates() {
    const images = Array.from(document.querySelectorAll('img[src], img[data-src]'));
    return images.filter((img) => {
        const sourceUrl = getImageSourceUrl(img);
        return !!sourceUrl && isSourceImageUrl(sourceUrl) && !img.closest('header, nav, footer');
    });
}

function selectVisibleOrUnprocessedImageCandidate(candidates) {
    if (!Array.isArray(candidates) || candidates.length === 0) return null;

    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;

    const visibleCandidate = candidates.find((img) => {
        const sourceUrl = getImageSourceUrl(img);
        const computedStyle = window.getComputedStyle(img);
        const rect = img.getBoundingClientRect();
        return (
            !!sourceUrl &&
            computedStyle.display !== 'none' &&
            computedStyle.visibility !== 'hidden' &&
            rect.width > 0 &&
            rect.height > 0 &&
            rect.bottom >= 0 &&
            rect.top <= viewportHeight
        );
    });
    if (visibleCandidate) return visibleCandidate;

    const unprocessedCandidate = candidates.find((img) => {
        const sourceUrl = getImageSourceUrl(img);
        return !!sourceUrl && img.dataset.aiProcessedSrc !== sourceUrl;
    });
    if (unprocessedCandidate) return unprocessedCandidate;

    return candidates[0];
}

function selectReaderForegroundImage() {
    const candidates = getReaderImageCandidates();
    return selectVisibleOrUnprocessedImageCandidate(candidates);
}

function isLikelyReaderContainerElement(element) {
    if (!(element instanceof Element)) return false;
    if (element === document.body || element === document.documentElement) return false;

    if (element.id === 'image-container') return true;

    const className = typeof element.className === 'string' ? element.className : '';
    if (/\b(md--reader-pages|md--page|rpage-main__inner|rpage-page)\b/i.test(className)) return true;

    const idName = typeof element.id === 'string' ? element.id : '';
    return /(?:reader|page|image|viewer)/i.test(idName) && /(?:reader|page|image|viewer)/i.test(className);
}

function getReaderContainerFromImage(img) {
    if (!(img instanceof HTMLImageElement)) return null;

    let current = img.parentElement;
    let fallback = current;

    while (current && current !== document.body) {
        if (isLikelyReaderContainerElement(current)) {
            return current;
        }

        const parent = current.parentElement;
        if (!parent) break;

        if (current.tagName === 'A' || current.tagName === 'PICTURE' || current.tagName === 'FIGURE') {
            current = parent;
            continue;
        }

        if (current.querySelectorAll('img').length === 1 && current.contains(img)) {
            fallback = current;
            if (parent.children.length > 1) {
                return current;
            }
        }

        current = parent;
    }

    return fallback || img.parentElement || img;
}

function getGenericActiveContainer() {
    const foregroundImage = selectReaderForegroundImage();
    if (!foregroundImage) return null;

    const container = getReaderContainerFromImage(foregroundImage);
    return container instanceof Element ? container : null;
}

function getActiveContainer(pageUrl = window.location.href) {
    const activeAdapter = getActiveSourceAdapter(pageUrl);
    if (activeAdapter && typeof activeAdapter?.getActiveContainer === 'function') {
        const container = activeAdapter.getActiveContainer(pageUrl);
        if (container instanceof Element) return container;
    }

    const genericContainer = getGenericActiveContainer();
    if (genericContainer) return genericContainer;

    return document.querySelector('#image-container');
}

function selectForegroundImage(container, pageUrl = window.location.href) {
    if (!(container instanceof Element)) return null;

    const activeAdapter = getActiveSourceAdapter(pageUrl);
    if (activeAdapter && typeof activeAdapter?.selectForegroundImage === 'function') {
        const selectedImage = activeAdapter.selectForegroundImage(container, pageUrl);
        if (selectedImage instanceof HTMLImageElement) return selectedImage;
    }

    return container.querySelector('img');
}
