// Source adapter for nhentai URL and reader/image detection

const NHENTAI_SOURCE_ID = 'nhentai';
const loggedNhentaiParseIssues = new Set();

function parseNhentaiUrlSafely(url) {
    if (typeof url !== 'string' || !url) return null;
    try {
        return new URL(url, window.location.href);
    } catch {
        return null;
    }
}

function isNhentaiHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)nhentai\.net$/i.test(hostname);
}

function logNhentaiParseIssue(kind, url, extra = {}) {
    if (typeof url !== 'string' || !url) return;

    const issueKey = `${kind}|${url}`;
    if (loggedNhentaiParseIssues.has(issueKey)) return;
    loggedNhentaiParseIssues.add(issueKey);

    if (typeof window.NHScalerLog === 'function') {
        window.NHScalerLog(`url:${kind}`, { source: NHENTAI_SOURCE_ID, url, ...extra });
    }
}

const nhentaiSourceAdapter = {
    id: NHENTAI_SOURCE_ID,
    mirrorSourceImagePresentation: false,
    supportsUrl(url) {
        const parsed = parseNhentaiUrlSafely(url);
        return !!parsed && isNhentaiHost(parsed.hostname);
    },
    isReaderPageUrl(url) {
        const parsed = parseNhentaiUrlSafely(url);
        if (!parsed || !isNhentaiHost(parsed.hostname)) return false;
        return /^\/g\/\d+\/\d+\/?$/i.test(parsed.pathname);
    },
    parseImageUrl(url) {
        const parsed = parseNhentaiUrlSafely(url);
        if (!parsed || !isNhentaiHost(parsed.hostname)) return null;

        const match = parsed.pathname.match(/^\/galleries\/(\d+)\/(\d+)\.(webp|jpe?g|png)$/i);
        if (!match) return null;

        const galleryId = match[1];
        const page = Number(match[2]);
        if (!Number.isFinite(page)) {
            logNhentaiParseIssue('page-number-invalid', url, { rawPage: match[2] });
            return null;
        }

        return {
            parsedUrl: parsed,
            galleryId,
            page,
            extension: match[3].toLowerCase(),
            pageKey: `${galleryId}/${page}`
        };
    },
    getActiveContainer() {
        return getGenericActiveContainer();
    },
    selectForegroundImage(container) {
        return container.querySelector('img');
    }
};

registerSourceAdapter(nhentaiSourceAdapter);
