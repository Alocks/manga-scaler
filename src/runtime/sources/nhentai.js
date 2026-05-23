// Source adapter for nhentai URL and reader/image detection

const loggedNhentaiParseIssues = new Set();

function isNhentaiHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)nhentai\.net$/i.test(hostname);
}

function isNhentaiImageHost(hostname) {
    return isNhentaiHost(hostname);
}

function parseNhentaiImagePath(pathname) {
    if (typeof pathname !== 'string' || !pathname) return null;

    const match = pathname.match(/^\/galleries\/(\d+)\/(\d+)\.(webp|jpe?g|png|avif)$/i);
    if (!match) return null;

    return {
        galleryId: match[1],
        pageRaw: match[2],
        extension: match[3].toLowerCase()
    };
}

const nhentaiSourceAdapter = createStructuredSourceAdapter({
    id: 'nhentai',
    supportsHost: isNhentaiHost,
    isReaderPath: (pathname) => /^\/g\/\d+\/\d+\/?$/i.test(pathname),
    isImageHost: isNhentaiImageHost,
    parsePathname: parseNhentaiImagePath,
    issueStore: loggedNhentaiParseIssues,
    buildParsedImageMeta: ({ parsed, pathInfo, page }) => ({
        parsedUrl: parsed,
        galleryId: pathInfo.galleryId,
        page,
        extension: pathInfo.extension,
        pageKey: `${pathInfo.galleryId}/${page}`
    })
});

registerSourceAdapter(nhentaiSourceAdapter);
