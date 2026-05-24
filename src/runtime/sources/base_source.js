function parseSourceUrlSafely(url) {
    if (typeof url !== 'string' || !url) return null;
    try {
        return new URL(url, window.location.href);
    } catch {
        return null;
    }
}

function parseSourceBlobInnerUrl(url, parseUrlFn = parseSourceUrlSafely) {
    if (typeof url !== 'string' || !url || !url.startsWith('blob:')) return null;
    return parseUrlFn(url.slice(5));
}

function getBlobAwareParsedUrl(url, parseUrlFn = parseSourceUrlSafely) {
    const parsed = parseUrlFn(url);
    if (!parsed) return null;
    if (parsed.protocol !== 'blob:') return parsed;
    return parseSourceBlobInnerUrl(url, parseUrlFn);
}

function logSourceParseIssue(issueStore, sourceId, kind, url, extra = {}) {
    if (!(issueStore instanceof Set)) return;
    if (typeof sourceId !== 'string' || !sourceId) return;
    if (typeof url !== 'string' || !url) return;

    const issueKey = `${kind}|${url}`;
    if (issueStore.has(issueKey)) return;
    issueStore.add(issueKey);

    if (typeof window.MangaScalerLog === 'function') {
        window.MangaScalerLog(`url:${kind}`, { source: sourceId, url, ...extra });
    }
}

function getSourceImagesByParser(container, parseImageUrlFn) {
    if (!(container instanceof Element) || typeof parseImageUrlFn !== 'function') return [];

    const imgs = Array.from(container.querySelectorAll('img[src], img[data-src]'));
    return imgs.filter((img) => {
        const sourceUrl = getImageSourceUrl(img);
        return !!sourceUrl && !!parseImageUrlFn(sourceUrl);
    });
}

function parseStructuredImageUrl({
    url,
    parseUrlFn = parseSourceUrlSafely,
    isImageHost,
    parsePathname,
    issueStore,
    sourceId,
    invalidPageIssue = 'page-number-invalid'
}) {
    const parsed = getBlobAwareParsedUrl(url, parseUrlFn);
    if (!parsed) return null;

    if (typeof isImageHost === 'function' && !isImageHost(parsed.hostname)) {
        return null;
    }

    if (typeof parsePathname !== 'function') return null;
    const pathInfo = parsePathname(parsed.pathname);
    if (!pathInfo) return null;

    const page = Number(pathInfo.pageRaw);
    if (!Number.isFinite(page)) {
        logSourceParseIssue(issueStore, sourceId, invalidPageIssue, url, { rawPage: pathInfo.pageRaw });
        return null;
    }

    return { parsed, pathInfo, page };
}

function createStructuredSourceAdapter({
    id,
    supportsHost,
    isReaderPath,
    parsePathname,
    buildParsedImageMeta,
    isImageHost,
    issueStore,
    parseUrlFn = parseSourceUrlSafely,
    getActiveContainer = getGenericActiveContainer,
    selectForegroundImage,
    parseImageUrlOverride,
    getImageLoadCandidates,
    isReaderPageUrlOverride,
    invalidPageIssue
}) {
    if (typeof id !== 'string' || !id) return null;

    const adapter = {
        id,
        isReaderPageUrl(url) {
            if (typeof isReaderPageUrlOverride === 'function') return !!isReaderPageUrlOverride(url);
            const parsed = parseUrlFn(url);
            if (!parsed || typeof supportsHost !== 'function' || !supportsHost(parsed.hostname)) return false;
            return typeof isReaderPath === 'function' ? !!isReaderPath(parsed.pathname, parsed) : false;
        },
        parseImageUrl(url) {
            if (typeof parseImageUrlOverride === 'function') return parseImageUrlOverride(url, adapter);

            const parsedResult = parseStructuredImageUrl({
                url,
                parseUrlFn,
                isImageHost,
                parsePathname,
                issueStore,
                sourceId: id,
                invalidPageIssue
            });
            if (!parsedResult) return null;

            if (typeof buildParsedImageMeta !== 'function') return null;
            return buildParsedImageMeta(parsedResult, url, adapter);
        },
        getImageLoadCandidates(sourceUrl, parsedImageInfo = null) {
            if (typeof getImageLoadCandidates === 'function') {
                return getImageLoadCandidates(sourceUrl, parsedImageInfo, adapter);
            }
            return [sourceUrl];
        },
        getActiveContainer(pageUrl) {
            if (typeof getActiveContainer === 'function') return getActiveContainer(pageUrl, adapter);
            return getGenericActiveContainer();
        },
        selectForegroundImage(container, pageUrl = window.location.href) {
            if (typeof selectForegroundImage === 'function') {
                return selectForegroundImage(container, pageUrl, adapter);
            }

            const sourceImgs = getSourceImagesByParser(container, (sourceUrl) => adapter.parseImageUrl(sourceUrl));
            return selectVisibleOrUnprocessedImageCandidate(sourceImgs);
        }
    };

    return adapter;
}

function registerSourceAdapter(adapter) {
    if (!adapter || typeof adapter !== 'object') return;

    if (!Array.isArray(window.NHScalerSourceAdapters)) {
        window.NHScalerSourceAdapters = [];
    }

    const existingIndex = window.NHScalerSourceAdapters.findIndex((existingAdapter) => existingAdapter?.id === adapter.id);
    if (existingIndex >= 0) {
        window.NHScalerSourceAdapters[existingIndex] = adapter;
    } else {
        window.NHScalerSourceAdapters.push(adapter);
    }
}
