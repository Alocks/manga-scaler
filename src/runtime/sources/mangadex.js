// Source adapter for MangaDex chapter URL and reader/image detection

const MANGADEX_SOURCE_ID = 'mangadex';
const loggedMangaDexParseIssues = new Set();

function parseMangaDexUrlSafely(url) {
    if (typeof url !== 'string' || !url) return null;
    try {
        return new URL(url, window.location.href);
    } catch {
        return null;
    }
}

function parseBlobInnerUrl(url) {
    if (typeof url !== 'string' || !url || !url.startsWith('blob:')) return null;
    return parseMangaDexUrlSafely(url.slice(5));
}

function isMangaDexHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)mangadex\.org$/i.test(hostname);
}

function isMangaDexImageHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)uploads\.mangadex\.org$/i.test(hostname);
}

function isMangaDexImageUrl(url) {
    const parsed = parseMangaDexUrlSafely(url);
    if (!parsed) return false;

    if (parsed.protocol === 'blob:') {
        const inner = parseBlobInnerUrl(url);
        return !!inner && isMangaDexHost(inner.hostname);
    }

    return isMangaDexImageHost(parsed.hostname);
}

function getMangaDexChapterId(pageUrl = window.location.href) {
    const parsed = parseMangaDexUrlSafely(pageUrl);
    if (!parsed || !isMangaDexHost(parsed.hostname)) return null;

    const match = parsed.pathname.match(/^\/chapter\/([0-9a-f-]{32,36})(?:\/\d+)?\/?$/i);
    return match ? match[1].toLowerCase() : null;
}

function getMangaDexReaderImages() {
    return Array.from(document.querySelectorAll('.md--reader-pages img[src], .md--reader-pages img[data-src]'));
}

function getMangaDexPageNumberFromDom(imageUrl) {
    if (typeof imageUrl !== 'string' || !imageUrl) return null;

    const imgs = getMangaDexReaderImages();
    for (let i = 0; i < imgs.length; i++) {
        const img = imgs[i];
        const srcUrl = getImageSourceUrl(img);
        if (!srcUrl) continue;
        if (srcUrl === imageUrl) {
            return i + 1;
        }
    }

    return null;
}

function toPositiveInt(value) {
    const n = Number(value);
    return Number.isInteger(n) && n > 0 ? n : null;
}

function getMangaDexCurrentPageFromUrl(pageUrl = window.location.href) {
    const parsed = parseMangaDexUrlSafely(pageUrl);
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

function logMangaDexParseIssue(kind, url, extra = {}) {
    if (typeof url !== 'string' || !url) return;

    const issueKey = `${kind}|${url}`;
    if (loggedMangaDexParseIssues.has(issueKey)) return;
    loggedMangaDexParseIssues.add(issueKey);

    if (typeof window.NHScalerLog === 'function') {
        window.NHScalerLog(`url:${kind}`, { source: MANGADEX_SOURCE_ID, url, ...extra });
    }
}

const mangadexSourceAdapter = {
    id: MANGADEX_SOURCE_ID,
    mirrorSourceImagePresentation: false,
    supportsUrl(url) {
        const parsed = parseMangaDexUrlSafely(url);
        return !!parsed && isMangaDexHost(parsed.hostname);
    },
    isReaderPageUrl(url) {
        const parsed = parseMangaDexUrlSafely(url);
        if (!parsed || !isMangaDexHost(parsed.hostname)) return false;
        return /^\/chapter\/[0-9a-f-]{32,36}(?:\/\d+)?\/?$/i.test(parsed.pathname);
    },
    parseImageUrl(url) {
        if (!isMangaDexImageUrl(url)) return null;

        const parsed = parseMangaDexUrlSafely(url);
        if (!parsed) return null;

        const chapterId = getMangaDexChapterId() || 'unknown';
        const page = getMangaDexPageNumberFromDom(url);
        if (page == null) {
            logMangaDexParseIssue('page-number-missing', url, { chapterId });
        }

        return {
            parsedUrl: parsed,
            chapterId,
            page,
            pageKey: page != null ? `${chapterId}/${page}` : `${chapterId}/${url}`
        };
    },
    getActiveContainer() {
        return getGenericActiveContainer();
    },
    selectForegroundImage(container, pageUrl = window.location.href) {
        const imgs = Array.from(container.querySelectorAll('img[src], img[data-src]'));
        const sourceImgs = imgs.filter((img) => {
            const sourceUrl = getImageSourceUrl(img);
            return !!sourceUrl && isMangaDexImageUrl(sourceUrl);
        });
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

        const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;

        const visibleCandidate = sourceImgs.find((img) => {
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

        const unprocessedCandidate = sourceImgs.find((img) => {
            const sourceUrl = getImageSourceUrl(img);
            return sourceUrl && img.dataset.aiProcessedSrc !== sourceUrl;
        });
        if (unprocessedCandidate) return unprocessedCandidate;

        return sourceImgs[0];
    }
};

registerSourceAdapter(mangadexSourceAdapter);
