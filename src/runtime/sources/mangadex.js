// Source adapter for MangaDex chapter URL and reader/image detection

const loggedMangaDexParseIssues = new Set();

function isMangaDexHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)mangadex\.org$/i.test(hostname);
}

function isMangaDexImageHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)(uploads\.mangadex\.org|mangadex\.network)$/i.test(hostname);
}

function isMangaDexRasterImagePath(pathname) {
    return typeof pathname === 'string' && /\.(avif|webp|jpe?g|png)$/i.test(pathname);
}

function toPositiveInt(value) {
    const n = Number(value);
    return Number.isInteger(n) && n > 0 ? n : null;
}

function getMangaDexChapterId(pageUrl = window.location.href) {
    const parsed = parseSourceUrlSafely(pageUrl);
    if (!parsed || !isMangaDexHost(parsed.hostname)) return null;

    const match = parsed.pathname.match(/^\/chapter\/([0-9a-f-]{32,36})(?:\/\d+)?\/?$/i);
    return match ? match[1].toLowerCase() : null;
}

function getMangaDexReaderImages() {
    const selectors = [
        '.md--reader-pages img[src], .md--reader-pages img[data-src]',
        '[class*="reader-pages"] img[src], [class*="reader-pages"] img[data-src]',
        'img[src*="mangadex.network/data/"], img[data-src*="mangadex.network/data/"]'
    ];

    for (const selector of selectors) {
        const images = Array.from(document.querySelectorAll(selector));
        if (images.length > 0) return images;
    }

    return [];
}

function getMangaDexActiveContainer() {
    const explicitContainer = document.querySelector('.md--reader-pages, [class*="reader-pages"]');
    if (explicitContainer instanceof Element) return explicitContainer;

    const readerImages = getMangaDexReaderImages();
    if (readerImages.length > 0) {
        return getReaderContainerFromImage(readerImages[0]);
    }

    return getGenericActiveContainer();
}

function getMangaDexPageNumberFromDom(imageUrl) {
    if (typeof imageUrl !== 'string' || !imageUrl) return null;

    const imgs = getMangaDexReaderImages();
    for (let i = 0; i < imgs.length; i++) {
        const img = imgs[i];
        const srcUrl = getImageSourceUrl(img);
        if (!srcUrl) continue;
        if (srcUrl === imageUrl) return i + 1;
    }

    return null;
}

function getMangaDexCurrentPageFromUrl(pageUrl = window.location.href) {
    const parsed = parseSourceUrlSafely(pageUrl);
    if (!parsed || !isMangaDexHost(parsed.hostname)) return null;

    const match = parsed.pathname.match(/^\/chapter\/[0-9a-f-]{32,36}\/(\d+)\/?$/i);
    return match ? toPositiveInt(match[1]) : null;
}

function getMangaDexCurrentPageFromProgressUi() {
    const currentProgressLabel = document.querySelector('.reader-progress-wrap .prog-divider.current .prog-divider-label');
    if (currentProgressLabel) {
        const current = toPositiveInt(currentProgressLabel.textContent?.trim());
        if (current != null) return current;
    }

    const pageMeta = document.querySelector('.reader--meta.page');
    if (pageMeta) {
        const match = String(pageMeta.textContent || '').match(/Pg\.\s*(\d+)\s*\//i);
        const current = match ? toPositiveInt(match[1]) : null;
        if (current != null) return current;
    }

    const pageSelect = document.querySelector('label + div span');
    if (pageSelect) {
        const current = toPositiveInt(pageSelect.textContent?.trim());
        if (current != null) return current;
    }

    return null;
}

function getMangaDexCurrentPageNumber(pageUrl = window.location.href) {
    return getMangaDexCurrentPageFromUrl(pageUrl) || getMangaDexCurrentPageFromProgressUi();
}

const mangadexSourceAdapter = createStructuredSourceAdapter({
    id: 'mangadex',
    supportsHost: isMangaDexHost,
    isReaderPath: (pathname) => /^\/chapter\/[0-9a-f-]{32,36}(?:\/\d+)?\/?$/i.test(pathname),
    getActiveContainer() {
        return getMangaDexActiveContainer();
    },
    parseImageUrlOverride(url, adapter) {
        const parsed = getBlobAwareParsedUrl(url, parseSourceUrlSafely);
        if (!parsed || !isMangaDexImageHost(parsed.hostname) || !isMangaDexRasterImagePath(parsed.pathname)) return null;

        const chapterId = getMangaDexChapterId() || 'unknown';
        const page = getMangaDexPageNumberFromDom(url);
        if (page == null) {
            logSourceParseIssue(loggedMangaDexParseIssues, adapter.id, 'page-number-missing', url, {
                chapterId
            });
        }

        return {
            parsedUrl: parsed,
            chapterId,
            page,
            pageKey: page != null ? `${chapterId}/${page}` : `${chapterId}/${url}`
        };
    },
    selectForegroundImage(container, pageUrl = window.location.href, adapter) {
        const sourceImgs = getSourceImagesByParser(container, (sourceUrl) => adapter.parseImageUrl(sourceUrl));
        if (sourceImgs.length === 0) return null;

        const currentPage = getMangaDexCurrentPageNumber(pageUrl);
        if (currentPage != null) {
            const pageByIndex = sourceImgs[currentPage - 1];
            if (pageByIndex) return pageByIndex;

            const pageByAltPrefix = sourceImgs.find((img) => {
                const alt = String(img.alt || '').trim();
                return new RegExp(`^${currentPage}[-_.\\s]`).test(alt) || alt === String(currentPage);
            });
            if (pageByAltPrefix) return pageByAltPrefix;
        }

        return selectVisibleOrUnprocessedImageCandidate(sourceImgs);
    }
});

registerSourceAdapter(mangadexSourceAdapter);
