// Source adapter for comix.to URL and reader/image detection

const COMIX_SOURCE_ID = 'comix';
const loggedComixParseIssues = new Set();

function parseComixUrlSafely(url) {
    if (typeof url !== 'string' || !url) return null;
    try {
        return new URL(url, window.location.href);
    } catch {
        return null;
    }
}

function isComixHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)comix\.to$/i.test(hostname);
}

function isComixImageHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)wowpic\d+\.store$/i.test(hostname);
}

function parseComixBlobInnerUrl(url) {
    if (typeof url !== 'string' || !url || !url.startsWith('blob:')) return null;
    return parseComixUrlSafely(url.slice(5));
}

function getComixImageParsedUrl(url) {
    const parsed = parseComixUrlSafely(url);
    if (!parsed) return null;
    if (parsed.protocol !== 'blob:') return parsed;
    return parseComixBlobInnerUrl(url);
}

function parseComixImagePath(pathname) {
    if (typeof pathname !== 'string' || !pathname) return null;

    // Primary comix reader CDN shape.
    const iiMatch = pathname.match(/^\/ii\/([^/]+)\/(\d+)\.(webp|jpe?g|png|avif)$/i);
    if (iiMatch) {
        return {
            hash: iiMatch[1],
            pageRaw: iiMatch[2],
            extension: iiMatch[3].toLowerCase()
        };
    }

    // Fallback for alternate CDN layouts where page is still in filename.
    const fileMatch = pathname.match(/\/([^/]+)\/(\d+)\.(webp|jpe?g|png|avif)$/i);
    if (fileMatch) {
        return {
            hash: fileMatch[1],
            pageRaw: fileMatch[2],
            extension: fileMatch[3].toLowerCase()
        };
    }

    return null;
}

function logComixParseIssue(kind, url, extra = {}) {
    if (typeof url !== 'string' || !url) return;

    const issueKey = `${kind}|${url}`;
    if (loggedComixParseIssues.has(issueKey)) return;
    loggedComixParseIssues.add(issueKey);

    if (typeof window.NHScalerLog === 'function') {
        window.NHScalerLog(`url:${kind}`, { source: COMIX_SOURCE_ID, url, ...extra });
    }
}

const comixSourceAdapter = {
    id: COMIX_SOURCE_ID,
    mirrorSourceImagePresentation: false,
    supportsUrl(url) {
        const parsed = parseComixUrlSafely(url);
        return !!parsed && isComixHost(parsed.hostname);
    },
    isReaderPageUrl(url) {
        const parsed = parseComixUrlSafely(url);
        if (!parsed || !isComixHost(parsed.hostname)) return false;
        // Matches /title/{manga_title}/{chapter}
        return /^\/title\/[^/]+\/[^/]+\/?$/i.test(parsed.pathname);
    },
    parseImageUrl(url) {
        const parsed = getComixImageParsedUrl(url);
        if (!parsed) return null;

        const isKnownImageHost = isComixImageHost(parsed.hostname) || isComixHost(parsed.hostname);
        if (!isKnownImageHost) return null;

        const pathInfo = parseComixImagePath(parsed.pathname);
        if (!pathInfo) return null;

        const hash = pathInfo.hash;
        const page = Number(pathInfo.pageRaw);
        if (!Number.isFinite(page)) {
            logComixParseIssue('page-number-invalid', url, { rawPage: pathInfo.pageRaw });
            return null;
        }

        return {
            parsedUrl: parsed,
            hash,
            page,
            extension: pathInfo.extension,
            pageKey: `${hash}/${page}`
        };
    },
    getActiveContainer() {
        return getGenericActiveContainer();
    },
    selectForegroundImage(container) {
        const imgs = Array.from(container.querySelectorAll('img[src], img[data-src]'));
        const sourceImgs = imgs.filter((img) => {
            const sourceUrl = getImageSourceUrl(img);
            return !!sourceUrl && !!comixSourceAdapter.parseImageUrl(sourceUrl);
        });
        if (sourceImgs.length === 0) return null;

        const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;

        const visibleCandidate = sourceImgs.find((img) => {
            const rect = img.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 && rect.bottom >= 0 && rect.top <= viewportHeight;
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

registerSourceAdapter(comixSourceAdapter);
