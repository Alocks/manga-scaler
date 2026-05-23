// Template source adapter scaffold for adding a new website
// Copy this file, rename identifiers, and implement the three adapter methods.

const TEMPLATE_SOURCE_ID = 'template-site';

const templateSiteSourceAdapter = {
    id: TEMPLATE_SOURCE_ID,
    supportsUrl(_url) {
        // Return true when the URL belongs to your target website.
        return false;
    },
    isReaderPageUrl(_url) {
        // Return true only on pages where image upscaling should run.
        return false;
    },
    parseImageUrl(_url) {
        // Return null for non-matching image URLs.
        // For matching URLs, return an object with at least:
        // { page: Number, pageKey: String }
        return null;
    },
    getActiveContainer(_pageUrl) {
        // Return the Element that should be observed/processed.
        // Return null if the reader container does not exist yet.
        return null;
    },
    selectForegroundImage(_container, _pageUrl) {
        // Return the single image element to process next.
        // Return null when no source image is currently available.
        return null;
    }
};

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

registerSourceAdapter(templateSiteSourceAdapter);
