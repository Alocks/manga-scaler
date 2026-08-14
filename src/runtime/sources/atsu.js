// Source adapter for atsu.moe reader and static image detection

const loggedAtsuParseIssues = new Set();

function isAtsuHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)atsu\.moe$/i.test(hostname);
}

function isAtsuImageHost(hostname) {
    return typeof hostname === 'string' && /(^|\.)cdn\.atsu\.moe$/i.test(hostname);
}

function parseAtsuImagePath(pathname) {
    if (typeof pathname !== 'string' || !pathname) return null;

    const match = pathname.match(/^\/static\/pages\/([^/]+)\/(\d+)\.(webp|jpe?g|png|avif)$/i);
    if (!match) return null;

    return {
        chapterId: match[1],
        pageRaw: match[2],
        extension: match[3].toLowerCase()
    };
}

const atsuSourceAdapter = createStructuredSourceAdapter({
    id: 'atsu',
    supportsHost: isAtsuHost,
    isReaderPath: (pathname) => /^\/read\/[^/]+\/[^/]+\/?$/i.test(pathname),
    isImageHost: (hostname) => isAtsuImageHost(hostname) || isAtsuHost(hostname),
    parsePathname: parseAtsuImagePath,
    issueStore: loggedAtsuParseIssues,
    buildParsedImageMeta: ({ parsed, pathInfo, page }) => ({
        parsedUrl: parsed,
        chapterId: pathInfo.chapterId,
        page,
        extension: pathInfo.extension,
        pageKey: `${pathInfo.chapterId}/${page}`
    })
});

registerSourceAdapter(atsuSourceAdapter);
