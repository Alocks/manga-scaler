// Source adapter for mangakakalot.gg chapter URL and reader/image detection

const loggedMangakakalotParseIssues = new Set();

function isMangakakalotHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)(mangakakalot|manganato)\.gg$/i.test(hostname);
}

function getMangakakalotReaderImages() {
    const selectors = [
        '.container-chapter-reader img[src], .container-chapter-reader img[data-src]',
        '.reading-content img[src], .reading-content img[data-src]',
        '.chapter-content img[src], .chapter-content img[data-src]',
        '#vungdoc img[src], #vungdoc img[data-src]'
    ];

    for (const selector of selectors) {
        const images = Array.from(document.querySelectorAll(selector));
        if (images.length > 0) return images;
    }

    return [];
}

function getMangakakalotPageNumberFromDom(imageUrl) {
    if (typeof imageUrl !== 'string' || !imageUrl) return null;

    const readerImages = getMangakakalotReaderImages();
    for (let i = 0; i < readerImages.length; i++) {
        const candidateUrl = getImageSourceUrl(readerImages[i]);
        if (candidateUrl && candidateUrl === imageUrl) {
            return i + 1;
        }
    }

    return null;
}

function getMangakakalotChapterId(pageUrl = window.location.href) {
    const parsed = parseSourceUrlSafely(pageUrl);
    if (!parsed || !isMangakakalotHost(parsed.hostname)) return 'unknown';

    const chapterMatch = parsed.pathname.match(/\/chapter\/([^/?#]+)/i);
    if (chapterMatch) return chapterMatch[1].toLowerCase();

    const normalizedPath = parsed.pathname.replace(/\/+$/, '').toLowerCase();
    return normalizedPath || 'unknown';
}

function getMangakakalotOnErrorFallbackSourceUrl(sourceUrl) {
    if (typeof sourceUrl !== 'string' || !sourceUrl) return null;

    const images = document.querySelectorAll('.container-chapter-reader img[onerror], .reading-content img[onerror], .chapter-content img[onerror], #vungdoc img[onerror]');
    for (const img of images) {
        const imgSource = getImageSourceUrl(img);
        const originalSource = img.dataset?.aiOriginalSrc || null;
        if (imgSource !== sourceUrl && originalSource !== sourceUrl) continue;

        const onErrorValue = img.getAttribute('onerror') || '';
        const match = onErrorValue.match(/this\.src\s*=\s*['\"]([^'\"]+)['\"]/i);
        if (!match) continue;

        const fallbackRaw = match[1].trim();
        if (!fallbackRaw) continue;

        const fallbackParsed = parseSourceUrlSafely(fallbackRaw);
        if (fallbackParsed) return fallbackParsed.href;
    }

    return null;
}

function getMangakakalotImageLoadCandidates(sourceUrl) {
    if (typeof sourceUrl !== 'string' || !sourceUrl) return [];

    const candidates = [sourceUrl];
    const parsed = parseSourceUrlSafely(sourceUrl);
    if (parsed) {
        const imgRMatch = parsed.hostname.match(/^img-r(\d+)\.2xstorage\.com$/i);
        const imgsMatch = parsed.hostname.match(/^imgs-(\d+)\.2xstorage\.com$/i);

        if (imgRMatch) {
            const fallback = new URL(parsed.href);
            fallback.hostname = `imgs-${imgRMatch[1]}.2xstorage.com`;
            candidates.push(fallback.href);
        } else if (imgsMatch) {
            const fallback = new URL(parsed.href);
            fallback.hostname = `img-r${imgsMatch[1]}.2xstorage.com`;
            candidates.push(fallback.href);
        }
    }

    const onErrorFallback = getMangakakalotOnErrorFallbackSourceUrl(sourceUrl);
    if (onErrorFallback) {
        candidates.push(onErrorFallback);
    }

    return [...new Set(candidates.filter((candidate) => typeof candidate === 'string' && !!candidate))];
}

const mangakakalotSourceAdapter = createStructuredSourceAdapter({
    id: 'mangakakalot',
    supportsHost: isMangakakalotHost,
    isReaderPath: (pathname) => /\/(?:chapter(?:[-/]|$)|read(?:[-/]|$))/i.test(pathname),
    parseImageUrlOverride(url, adapter) {
        const parsed = getBlobAwareParsedUrl(url, parseSourceUrlSafely);
        if (!parsed) return null;

        const page = getMangakakalotPageNumberFromDom(url);
        const chapterId = getMangakakalotChapterId();
        if (page == null) {
            logSourceParseIssue(loggedMangakakalotParseIssues, adapter.id, 'page-number-missing', url, {
                chapterId
            });
            return null;
        }

        return {
            parsedUrl: parsed,
            chapterId,
            page,
            pageKey: `${chapterId}/${page}`
        };
    },
    getActiveContainer() {
        return document.querySelector('.container-chapter-reader, .reading-content, .chapter-content, #vungdoc') || getGenericActiveContainer();
    },
    getImageLoadCandidates(sourceUrl) {
        return getMangakakalotImageLoadCandidates(sourceUrl);
    }
});

registerSourceAdapter(mangakakalotSourceAdapter);
