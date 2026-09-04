// Image processing cache: in-memory Map and IndexedDB persistence

const processedCache = new Map();
const MAX_PROCESSED_CACHE_ENTRIES = 100;
const PROCESSED_CACHE_DB_NAME = 'manga-scaler-processed-cache';
const PROCESSED_CACHE_STORE_NAME = 'images';
const MIN_VALID_PROCESSED_BLOB_BYTES = 723000;
let processedCacheDbPromise = null;

function resetProcessedMemoryCache() {
    processedCache.clear();
}

function getProcessedCacheEntry(cacheKey) {
    if (!processedCache.has(cacheKey)) return null;

    const blob = processedCache.get(cacheKey) || null;
    if (!blob) {
        processedCache.delete(cacheKey);
        return null;
    }

    processedCache.delete(cacheKey);
    processedCache.set(cacheKey, blob);
    return blob;
}

function trimProcessedCacheEntries() {
    while (processedCache.size > MAX_PROCESSED_CACHE_ENTRIES) {
        const oldestKey = processedCache.keys().next().value;
        if (!oldestKey) break;
        processedCache.delete(oldestKey);
    }
}

function rememberProcessedCacheEntry(cacheKey, blob) {
    if (!cacheKey || !blob) return;

    if (processedCache.has(cacheKey)) {
        processedCache.delete(cacheKey);
    }

    processedCache.set(cacheKey, blob);
    trimProcessedCacheEntries();
}

async function decodeValidateProcessedBlob(blob) {
    try {
        const bitmap = await createImageBitmap(blob);
        const isValid = bitmap.width > 0 && bitmap.height > 0;
        if (typeof bitmap.close === 'function') {
            bitmap.close();
        }
        return isValid;
    } catch {
        return false;
    }
}

async function isValidProcessedBlob(blob) {
    // Guard against tiny/corrupt cache entries before decode validation.
    if (!(blob instanceof Blob) || blob.size <= MIN_VALID_PROCESSED_BLOB_BYTES) return false;
    return decodeValidateProcessedBlob(blob);
}

function isLikelyFreshProcessedBlob(blob) {
    return blob instanceof Blob && blob.size > MIN_VALID_PROCESSED_BLOB_BYTES;
}

function getProcessedCacheSignature(runtimeSettings = getRuntimePreferenceSnapshot()) {
    const settings = getNormalizedRuntimePreferenceSnapshot(runtimeSettings);
    const backend = getEffectiveBackend(settings);
    if (backend === 'off') return 'off';
    return `${backend}|${settings.selectedSimplePreset}|${settings.selectedWebGpuModel}|${settings.selectedWebGpuScale}|${settings.selectedOnnxModel}`;
}

function getProcessedCacheKey(url, runtimeSettings = getRuntimePreferenceSnapshot()) {
    if (typeof url !== 'string' || !url) return null;
    return `${getProcessedCacheSignature(runtimeSettings)}|${url}`;
}

function openProcessedCacheDb() {
    if (!('indexedDB' in window)) {
        return Promise.resolve(null);
    }

    if (processedCacheDbPromise) {
        return processedCacheDbPromise;
    }

    processedCacheDbPromise = new Promise((resolve) => {
        const request = indexedDB.open(PROCESSED_CACHE_DB_NAME, 1);

        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(PROCESSED_CACHE_STORE_NAME)) {
                db.createObjectStore(PROCESSED_CACHE_STORE_NAME, { keyPath: 'cacheKey' });
            }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => {
            runtimeLog('cache:db-open-error', { error: String(request.error) });
            resolve(null);
        };
        request.onblocked = () => {
            runtimeLog('cache:db-open-blocked');
            resolve(null);
        };
    });

    return processedCacheDbPromise;
}

async function deleteProcessedCacheEntryByKey(cacheKey) {
    if (!cacheKey) return false;

    processedCache.delete(cacheKey);

    const db = await openProcessedCacheDb();
    if (!db) return true;

    return new Promise((resolve) => {
        const tx = db.transaction(PROCESSED_CACHE_STORE_NAME, 'readwrite');
        const store = tx.objectStore(PROCESSED_CACHE_STORE_NAME);
        store.delete(cacheKey);
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => {
            runtimeLog('cache:db-delete-error', { cacheKey, error: String(tx.error) });
            resolve(false);
        };
        tx.onabort = () => {
            runtimeLog('cache:db-delete-abort', { cacheKey, error: String(tx.error) });
            resolve(false);
        };
    });
}

async function getProcessedCacheBlob(url, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const cacheKey = getProcessedCacheKey(url, runtimeSettings);
    if (!cacheKey) return null;

    const memoryBlob = getProcessedCacheEntry(cacheKey);
    if (memoryBlob) {
        runtimeLog('cache:memory-hit', { cacheKey, url, size: memoryBlob.size });
        return memoryBlob;
    }

    const db = await openProcessedCacheDb();
    if (!db) return null;

    return new Promise((resolve) => {
        const tx = db.transaction(PROCESSED_CACHE_STORE_NAME, 'readonly');
        const store = tx.objectStore(PROCESSED_CACHE_STORE_NAME);
        const request = store.get(cacheKey);

        request.onsuccess = async () => {
            const blob = request.result?.blob || null;
            if (!blob) {
                resolve(null);
                return;
            }

            if (await isValidProcessedBlob(blob)) {
                rememberProcessedCacheEntry(cacheKey, blob);
                resolve(blob);
                return;
            }

            await deleteProcessedCacheEntryByKey(cacheKey);
            runtimeLog('cache:evicted-invalid-db-blob', { cacheKey, url, size: blob.size });
            resolve(null);
        };

        request.onerror = () => {
            runtimeLog('cache:db-read-error', { cacheKey, error: String(request.error) });
            resolve(null);
        };
    });
}

async function setProcessedCacheBlob(url, blob, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const cacheKey = getProcessedCacheKey(url, runtimeSettings);
    if (!cacheKey || !blob) return false;

    // Freshly rendered canvas blobs have already passed canvas validity checks;
    // only persisted/restored blobs need a full decode validation pass.
    if (!isLikelyFreshProcessedBlob(blob)) {
        await deleteProcessedCacheEntryByKey(cacheKey);
        runtimeLog('cache:skip-invalid-write', { cacheKey, url, size: blob.size });
        return false;
    }

    rememberProcessedCacheEntry(cacheKey, blob);

    const db = await openProcessedCacheDb();
    if (!db) return true;

    return new Promise((resolve) => {
        const tx = db.transaction(PROCESSED_CACHE_STORE_NAME, 'readwrite');
        const store = tx.objectStore(PROCESSED_CACHE_STORE_NAME);
        store.put({ cacheKey, url, blob, updatedAt: Date.now() });
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => {
            runtimeLog('cache:db-write-error', { cacheKey, error: String(tx.error) });
            resolve(false);
        };
        tx.onabort = () => {
            runtimeLog('cache:db-write-abort', { cacheKey, error: String(tx.error) });
            resolve(false);
        };
    });
}

function hasProcessedCacheEntry(url, runtimeSettings = getRuntimePreferenceSnapshot()) {
    const cacheKey = getProcessedCacheKey(url, runtimeSettings);
    if (!cacheKey || !processedCache.has(cacheKey)) return false;

    const blob = processedCache.get(cacheKey) || null;
    if (!(blob instanceof Blob) || blob.size <= MIN_VALID_PROCESSED_BLOB_BYTES) {
        processedCache.delete(cacheKey);
        return false;
    }

    return true;
}

async function clearProcessedCache() {
    resetProcessedMemoryCache();

    const db = await openProcessedCacheDb();
    if (!db) return true;

    return new Promise((resolve) => {
        const tx = db.transaction(PROCESSED_CACHE_STORE_NAME, 'readwrite');
        const store = tx.objectStore(PROCESSED_CACHE_STORE_NAME);
        store.clear();
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => {
            runtimeLog('cache:db-clear-error', { error: String(tx.error) });
            resolve(false);
        };
        tx.onabort = () => {
            runtimeLog('cache:db-clear-abort', { error: String(tx.error) });
            resolve(false);
        };
    });
}
