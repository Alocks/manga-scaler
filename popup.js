const SIMPLE_PRESET_KEY = 'simplePreset';
const ENGINE_BACKEND_KEY = 'engineBackend';
const WEBGPU_MODEL_KEY = 'webgpuModel';
const WEBGPU_SCALE_KEY = 'webgpuScale';

const DEFAULT_SIMPLE_PRESET = 'M';
const DEFAULT_ENGINE_BACKEND = 'webgl';
const DEFAULT_WEBGPU_MODEL = 'ModeA';
const DEFAULT_WEBGPU_SCALE = 2;
const CLEAR_CACHE_MESSAGE_TYPE = 'manga-scaler:clear-cache';
const GET_DIAGNOSTICS_MESSAGE_TYPE = 'manga-scaler:get-diagnostics';
const SUPPORTED_TAB_URL_PATTERNS = [
  /^https?:\/\/(?:[^/]+\.)?nhentai\.net\//i,
  /^https?:\/\/(?:[^/]+\.)?comix\.to\//i,
  /^https?:\/\/(?:[^/]+\.)?mangadex\.org\//i
];
const POPUP_MESSAGE_TIMEOUT_MS = 5000;
let isWebGpuSupported = true;
let diagnosticsRefreshTimer = null;
let diagnosticsFollowUpTimer = null;

const clearCacheButton = document.getElementById('clearCacheButton');
const cacheActionStatus = document.getElementById('cacheActionStatus');
const runtimeDiagnostics = document.getElementById('runtimeDiagnostics');
const webgpuScaleSelect = document.getElementById('webgpuScale');

function normalizeWebGpuScale(value) {
  const scale = Number(value);
  return scale === 2 || scale === 3 || scale === 4 ? scale : DEFAULT_WEBGPU_SCALE;
}

function setCacheActionStatus(message, tone = '') {
  if (!cacheActionStatus) return;
  cacheActionStatus.textContent = message;
  cacheActionStatus.classList.remove('error', 'success');
  if (tone) {
    cacheActionStatus.classList.add(tone);
  }
}

function isSupportedMangaTab(tab) {
  return typeof tab?.url === 'string' && SUPPORTED_TAB_URL_PATTERNS.some((pattern) => pattern.test(tab.url));
}

function setRuntimeDiagnosticsText(text) {
  if (!runtimeDiagnostics) return;
  runtimeDiagnostics.textContent = text;
}

function scheduleRuntimeDiagnosticsRefresh() {
  if (diagnosticsRefreshTimer !== null) {
    clearTimeout(diagnosticsRefreshTimer);
  }
  diagnosticsRefreshTimer = window.setTimeout(() => {
    diagnosticsRefreshTimer = null;
    refreshRuntimeDiagnostics();
  }, 60);

  if (diagnosticsFollowUpTimer !== null) {
    clearTimeout(diagnosticsFollowUpTimer);
  }
  diagnosticsFollowUpTimer = window.setTimeout(() => {
    diagnosticsFollowUpTimer = null;
    refreshRuntimeDiagnostics();
  }, 360);
}

async function persistSettingAndRefresh(patch) {
  await chrome.storage.sync.set(patch);
  scheduleRuntimeDiagnosticsRefresh();
}

async function sendMessageWithTimeout(tabId, payload, timeoutMs = POPUP_MESSAGE_TIMEOUT_MS) {
  let timeoutId = null;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = window.setTimeout(() => {
      reject(new Error(`Request timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });

  const messagePromise = chrome.tabs.sendMessage(tabId, payload);

  try {
    return await Promise.race([messagePromise, timeoutPromise]);
  } finally {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
  }
}

function formatRuntimeDiagnostics(diagnostics) {
  if (!diagnostics || typeof diagnostics !== 'object') {
    return 'Diagnostics unavailable.';
  }

  const prefs = diagnostics.preferences || {};
  const hooks = diagnostics.hooks || {};
  const queue = diagnostics.queue || {};
  const adapters = diagnostics.adapters || {};
  const generatedAt = Number.isFinite(diagnostics.generatedAt) ? new Date(diagnostics.generatedAt).toLocaleTimeString() : 'unknown';
  const pageUrl = diagnostics.pageUrl || 'unknown';

  return [
    `Snapshot: ${generatedAt}`,
    `Page: ${pageUrl}`,
    `Backend: ${prefs.selectedEngineBackend || 'unknown'} (effective: ${diagnostics.effectiveBackend || 'unknown'})`,
    `Preset: ${prefs.selectedSimplePreset || 'unknown'} | WebGPU model: ${prefs.selectedWebGpuModel || 'unknown'} | WebGPU scale: ${prefs.selectedWebGpuScale || DEFAULT_WEBGPU_SCALE}x`,
    `Route: ${diagnostics.readerRoute ? 'reader' : 'non-reader'} | Foreground: ${diagnostics.foreground ? 'yes' : 'no'}`,
    `Hooks - fetch: ${hooks.fetch ? 'ok' : 'missing'}, image-src: ${hooks.imageSrc ? 'ok' : 'missing'}, Image(): ${hooks.imageConstructor ? 'ok' : 'missing'}`,
    `Adapters - WebGL: capable=${adapters.webgl?.capable ? 'yes' : 'no'}, initialized=${adapters.webgl?.initialized ? 'yes' : 'no'}, supported=${adapters.webgl?.isSupported ? 'yes' : 'no'}`,
    `Adapters - WebGPU: capable=${adapters.webgpu?.capable ? 'yes' : 'no'}, initialized=${adapters.webgpu?.initialized ? 'yes' : 'no'}, supported=${adapters.webgpu?.isSupported ? 'yes' : 'no'}`,
    `Queue - size: ${queue.size ?? 0}, in-flight: ${queue.inFlightCount ?? 0}, processed: ${queue.processedCount ?? 0}`
  ].join('\n');
}

async function refreshRuntimeDiagnostics() {
  const activeTab = await getActiveTab();
  if (!activeTab?.id || !isSupportedMangaTab(activeTab)) {
    setRuntimeDiagnosticsText('Open a supported manga tab to view runtime diagnostics.');
    return;
  }

  try {
    const response = await sendMessageWithTimeout(activeTab.id, { type: GET_DIAGNOSTICS_MESSAGE_TYPE });
    if (!response?.ok) {
      throw new Error(response?.error || 'Diagnostics request failed');
    }
    setRuntimeDiagnosticsText(formatRuntimeDiagnostics(response.diagnostics));
  } catch (error) {
    const message = String(error?.message || 'Diagnostics unavailable');
    setRuntimeDiagnosticsText(
      message.includes('Receiving end does not exist')
        ? 'Runtime not ready in this tab yet. Reload the manga page.'
        : `Diagnostics unavailable: ${message}`
    );
  }
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs?.[0] || null;
}

async function refreshCacheActionAvailability() {
  if (!clearCacheButton) return;

  try {
    const activeTab = await getActiveTab();
    const isEligible = !!activeTab?.id && isSupportedMangaTab(activeTab);
    clearCacheButton.disabled = !isEligible;

    if (!isEligible) {
      setCacheActionStatus('Open a supported manga tab to clear cached images.');
      return;
    }

    setCacheActionStatus('');
  } catch {
    clearCacheButton.disabled = true;
    setCacheActionStatus('Could not inspect the active tab.', 'error');
  }
}

async function detectWebGpuSupport() {
  if (!navigator?.gpu || typeof navigator.gpu.requestAdapter !== 'function') {
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

function applyWebGpuAvailabilityUi(supported) {
  const webgpuTab = document.querySelector('.tab-pill[data-tab="webgpu"]');
  if (webgpuTab) {
    webgpuTab.classList.toggle('disabled', !supported);
    webgpuTab.setAttribute('aria-disabled', String(!supported));
    webgpuTab.title = supported ? '' : 'WebGPU is not supported by this browser';
  }

  document.querySelectorAll('input[name="webgpuModel"]').forEach((input) => {
    input.disabled = !supported;
    const option = input.closest('.preset-option');
    if (option) option.classList.toggle('disabled', !supported);
  });

  if (webgpuScaleSelect) {
    webgpuScaleSelect.disabled = !supported;
  }
}

function setActiveEnginePanel(backend) {
  const normalizedBackend = String(backend || DEFAULT_ENGINE_BACKEND).toLowerCase();
  const panels = document.querySelectorAll('.engine-panel');
  const tabs = document.querySelectorAll('.tab-pill');

  panels.forEach((panel) => {
    panel.classList.toggle('active', panel.id === `enginePanel-${normalizedBackend}`);
  });

  tabs.forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.tab === normalizedBackend);
  });
}

async function loadCurrentSettings() {
  isWebGpuSupported = await detectWebGpuSupport();

  const result = await chrome.storage.sync.get({
    [SIMPLE_PRESET_KEY]: DEFAULT_SIMPLE_PRESET,
    [ENGINE_BACKEND_KEY]: DEFAULT_ENGINE_BACKEND,
    [WEBGPU_MODEL_KEY]: DEFAULT_WEBGPU_MODEL,
    [WEBGPU_SCALE_KEY]: DEFAULT_WEBGPU_SCALE
  });

  const currentPreset = String(result[SIMPLE_PRESET_KEY] || DEFAULT_SIMPLE_PRESET).toUpperCase();
  let currentBackend = String(result[ENGINE_BACKEND_KEY] || DEFAULT_ENGINE_BACKEND).toLowerCase();
  const currentWebGpuModel = String(result[WEBGPU_MODEL_KEY] || DEFAULT_WEBGPU_MODEL);
  const currentWebGpuScale = normalizeWebGpuScale(result[WEBGPU_SCALE_KEY]);

  if (!isWebGpuSupported && currentBackend === 'webgpu') {
    currentBackend = 'webgl';
    chrome.storage.sync.set({ [ENGINE_BACKEND_KEY]: currentBackend });
  }

  const presetRadio = document.querySelector(`input[name="preset"][value="${currentPreset}"]`);
  if (presetRadio) presetRadio.checked = true;

  const webgpuModelRadio = document.querySelector(`input[name="webgpuModel"][value="${currentWebGpuModel}"]`);
  if (webgpuModelRadio) webgpuModelRadio.checked = true;

  if (webgpuScaleSelect) {
    webgpuScaleSelect.value = String(currentWebGpuScale);
  }

  applyWebGpuAvailabilityUi(isWebGpuSupported);
  setActiveEnginePanel(currentBackend);
}

document.querySelectorAll('input[name="preset"]').forEach((radio) => {
  radio.addEventListener('change', async (e) => {
    if (e.target.checked) {
      await persistSettingAndRefresh({ [SIMPLE_PRESET_KEY]: e.target.value });
    }
  });
});

document.querySelectorAll('.tab-pill').forEach((tab) => {
  tab.addEventListener('click', async () => {
    const backend = String(tab.dataset.tab || DEFAULT_ENGINE_BACKEND).toLowerCase();
    if (backend === 'webgpu' && !isWebGpuSupported) {
      return;
    }
    setActiveEnginePanel(backend);
    await persistSettingAndRefresh({ [ENGINE_BACKEND_KEY]: backend });
  });
});

document.querySelectorAll('input[name="webgpuModel"]').forEach((radio) => {
  radio.addEventListener('change', async (e) => {
    if (e.target.checked) {
      await persistSettingAndRefresh({ [WEBGPU_MODEL_KEY]: e.target.value });
    }
  });
});

if (webgpuScaleSelect) {
  webgpuScaleSelect.addEventListener('change', async (event) => {
    const nextScale = normalizeWebGpuScale(event.target?.value);
    webgpuScaleSelect.value = String(nextScale);
    await persistSettingAndRefresh({ [WEBGPU_SCALE_KEY]: nextScale });
  });
}

if (chrome?.storage?.onChanged) {
  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== 'sync') return;
    if (
      changes[SIMPLE_PRESET_KEY] ||
      changes[ENGINE_BACKEND_KEY] ||
      changes[WEBGPU_MODEL_KEY] ||
      changes[WEBGPU_SCALE_KEY]
    ) {
      scheduleRuntimeDiagnosticsRefresh();
    }
  });
}

loadCurrentSettings();
refreshCacheActionAvailability();
refreshRuntimeDiagnostics();

if (clearCacheButton) {
  clearCacheButton.addEventListener('click', async () => {
    clearCacheButton.disabled = true;
    setCacheActionStatus('Clearing cache...');

    try {
      const activeTab = await getActiveTab();
      if (!activeTab?.id) {
        throw new Error('Open a supported manga tab first');
      }

      if (!isSupportedMangaTab(activeTab)) {
        throw new Error('Open a supported manga tab first');
      }

      const response = await sendMessageWithTimeout(activeTab.id, { type: CLEAR_CACHE_MESSAGE_TYPE });
      if (!response?.ok) {
        throw new Error(response?.error || 'Cache clear request failed');
      }

      setCacheActionStatus('Cached images cleared for this tab.', 'success');
    } catch (error) {
      const message = String(error?.message || '');
      setCacheActionStatus(
        message.includes('Receiving end does not exist')
          ? 'This tab is not ready yet. Reload the manga page and try again.'
          : error?.message || 'Could not clear cache. Open a supported manga tab and try again.',
        'error'
      );
    } finally {
      refreshCacheActionAvailability();
      refreshRuntimeDiagnostics();
    }
  });
}