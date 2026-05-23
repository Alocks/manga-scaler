// Base engine adapter helpers for implementing additional upscale backends.

function createBaseEngineAdapter(overrides = {}) {
    const adapter = {
        isSupported: () => false,
        upscale: async () => {
            throw new Error('Base engine adapter: upscale() is not implemented');
        },
        prewarm: async () => {},
        reset: () => {}
    };

    return {
        ...adapter,
        ...overrides
    };
}

function createEngineAdapter(overrides = {}) {
    return createBaseEngineAdapter(overrides);
}

function createLibraryBackedEngineAdapter({
    getLibrary,
    isLibrarySupported,
    upscale,
    ensureReady = async () => {},
    resetState = () => {},
    isInitialized = () => false,
    isRuntimeSupported
} = {}) {
    if (typeof getLibrary !== 'function') {
        throw new Error('createLibraryBackedEngineAdapter requires getLibrary()');
    }
    if (typeof isLibrarySupported !== 'function') {
        throw new Error('createLibraryBackedEngineAdapter requires isLibrarySupported()');
    }
    if (typeof upscale !== 'function') {
        throw new Error('createLibraryBackedEngineAdapter requires upscale()');
    }

    return createEngineAdapter({
        isSupported: () => {
            const lib = getLibrary();
            return !!lib && isLibrarySupported(lib);
        },
        upscale,
        prewarm: async (runtimeSettings = getRuntimePreferenceSnapshot()) => {
            await ensureReady(runtimeSettings);
        },
        reset: () => {
            resetState();
        },
        getDiagnosticsStatus: () => {
            const lib = getLibrary();
            const capable = !!lib && isLibrarySupported(lib);
            const initialized = !!isInitialized();
            const isSupported = typeof isRuntimeSupported === 'function'
                ? !!isRuntimeSupported()
                : capable;
            return { capable, initialized, isSupported };
        }
    });
}

function engineLibraryHasAnyFunctions(lib, functionNames = []) {
    if (!lib || typeof lib !== 'object' || !Array.isArray(functionNames)) return false;
    return functionNames.some((name) => typeof lib[name] === 'function');
}

// Expose shared engine helpers for backend-specific adapters.
window.createBaseEngineAdapter = createBaseEngineAdapter;
window.createEngineAdapter = createEngineAdapter;
window.createLibraryBackedEngineAdapter = createLibraryBackedEngineAdapter;
window.engineLibraryHasAnyFunctions = engineLibraryHasAnyFunctions;
