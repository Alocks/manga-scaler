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
        const parsed = parseComixUrlSafely(url);
        if (!parsed || !isComixImageHost(parsed.hostname)) return null;

        // Matches /ii/{hash}/{page}.{ext}
        const match = parsed.pathname.match(/^\/ii\/([^/]+)\/(\d+)\.(webp|jpe?g|png)$/i);
        if (!match) return null;

        const hash = match[1];
        const page = Number(match[2]);
        if (!Number.isFinite(page)) {
            logComixParseIssue('page-number-invalid', url, { rawPage: match[2] });
            return null;
        }

        return {
            parsedUrl: parsed,
            hash,
            page,
            extension: match[3].toLowerCase(),
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
            const parsed = parseComixUrlSafely(sourceUrl);
            return !!parsed && isComixImageHost(parsed.hostname);
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
