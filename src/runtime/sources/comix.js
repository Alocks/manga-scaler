// Source adapter for comix.to URL and reader/image detection

const loggedComixParseIssues = new Set();

function isComixHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)comix\.to$/i.test(hostname);
}

function isComixImageHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)wowpic\d+\.store$/i.test(hostname);
}

function parseComixImagePath(pathname) {
    if (typeof pathname !== 'string' || !pathname) return null;

    const iiMatch = pathname.match(/^\/ii\/([^/]+)\/(\d+)\.(webp|jpe?g|png|avif)$/i);
    if (iiMatch) {
        return {
            hash: iiMatch[1],
            pageRaw: iiMatch[2],
            extension: iiMatch[3].toLowerCase()
        };
    }

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

const comixSourceAdapter = createStructuredSourceAdapter({
    id: 'comix',
    supportsHost: isComixHost,
    isReaderPath: (pathname) => /^\/title\/[^/]+\/[^/]+\/?$/i.test(pathname),
    isImageHost: (hostname) => isComixImageHost(hostname) || isComixHost(hostname),
    parsePathname: parseComixImagePath,
    issueStore: loggedComixParseIssues,
    buildParsedImageMeta: ({ parsed, pathInfo, page }) => ({
        parsedUrl: parsed,
        hash: pathInfo.hash,
        page,
        extension: pathInfo.extension,
        pageKey: `${pathInfo.hash}/${page}`
    })
});

registerSourceAdapter(comixSourceAdapter);
