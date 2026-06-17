const SIMPLE_PRESET_KEY = 'simplePreset';
const ENGINE_BACKEND_KEY = 'engineBackend';
const WEBGPU_MODEL_KEY = 'webgpuModel';
const WEBGPU_SCALE_KEY = 'webgpuScale';
const ONNX_MODEL_KEY = 'onnxModel';

const DEFAULT_SIMPLE_PRESET = 'M';
const DEFAULT_ENGINE_BACKEND = 'webgl';
const DEFAULT_WEBGPU_MODEL = 'ModeA';
const DEFAULT_WEBGPU_SCALE = 2;
const DEFAULT_ONNX_MODEL = 'realesr-animevideov3';
const WEBGPU_MODEL_VALUES = new Set(['ModeA', 'ModeAA', 'ModeB', 'ModeBB', 'ModeC', 'ModeCA']);
const FALLBACK_ONNX_MODEL_OPTIONS = [
  { key: 'realesr-animevideov3', title: 'RealESR AnimeVideo v3' },
  { key: 'up2x-latest-conservative', title: 'RealCUGAN Conservative' },
  { key: 'up2x-latest-denoise1x', title: 'RealCUGAN 2X Denoise 1x' },
  { key: 'mangajanai-1200p-v1', title: 'MangaJaNai 2x 1200p V1' },
  { key: 'mangajanai-1600p-v1', title: 'MangaJaNai 2x 1600p V1' },
  { key: 'illustrationjanai-v1', title: 'IllustrationJaNai 2x V1' }
];
const ONNX_MODEL_HINTS = {
  'realesr-animevideov3': {
    runtime: 'Runtime ≈ 1.1 second'
  },
  'up2x-latest-conservative': {
    runtime: 'Runtime ≈ 4 seconds'
  },
  'up2x-latest-denoise1x': {
    runtime: 'Runtime ≈ 4 seconds'
  },
  'mangajanai-1200p-v1': {
    runtime: 'Runtime ≈ 4 seconds'
  },
  'mangajanai-1600p-v1': {
    runtime: 'Runtime ≈ 2.5 seconds'
  },
  'illustrationjanai-v1': {
    runtime: 'Runtime ≈ 2.5 seconds'
  }
};
const CLEAR_CACHE_MESSAGE_TYPE = 'manga-scaler:clear-cache';
const SUPPORTED_TAB_URL_PATTERNS = [
  /^https?:\/\/(?:[^/]+\.)?nhentai\.net\//i,
  /^https?:\/\/(?:[^/]+\.)?comix\.to\//i,
  /^https?:\/\/(?:[^/]+\.)?mangadex\.org\//i
];
const POPUP_MESSAGE_TIMEOUT_MS = 5000;
let isWebGpuSupported = true;

const clearCacheButton = document.getElementById('clearCacheButton');
const cacheActionStatus = document.getElementById('cacheActionStatus');
const shaderBackendSelect = document.getElementById('shaderBackend');
const webgpuScaleSelect = document.getElementById('webgpuScale');
const onnxModelList = document.getElementById('onnxModelList');

function getPopupOnnxModelOptions() {
  if (typeof globalThis.getOnnxModelOptions === 'function') {
    const options = globalThis.getOnnxModelOptions();
    if (Array.isArray(options) && options.length > 0) {
      return options;
    }
  }
  return FALLBACK_ONNX_MODEL_OPTIONS;
}

function normalizeOnnxModel(value) {
  const raw = String(value || '').trim();
  if (typeof globalThis.normalizeOnnxModelKey === 'function') {
    return globalThis.normalizeOnnxModelKey(raw);
  }

  const options = getPopupOnnxModelOptions();
  return options.some((option) => option.key === raw) ? raw : DEFAULT_ONNX_MODEL;
}

function buildOnnxModelSelectOptions(selectedValue = DEFAULT_ONNX_MODEL) {
  if (!onnxModelList) return;

  const normalizedSelected = normalizeOnnxModel(selectedValue);
  const options = getPopupOnnxModelOptions().slice().sort((a, b) => {
    return String(a.title || a.label || a.key).localeCompare(String(b.title || b.label || b.key));
  });

  onnxModelList.textContent = '';

  for (const option of options) {
    if (!option?.key) continue;
    const hint = ONNX_MODEL_HINTS[option.key] || {
      runtime: 'Runtime: +-2.5 seconds'
    };

    const modelRow = document.createElement('div');
    modelRow.className = 'onnx-model-row';

    const row = document.createElement('label');
    row.className = 'preset-option';

    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'onnxModel';
    input.value = option.key;
    input.checked = option.key === normalizedSelected;

    const text = document.createElement('span');
    text.className = 'preset-label';
    text.textContent = option.title || option.label || option.key;

    row.appendChild(input);
    row.appendChild(text);

    const desc = document.createElement('div');
    desc.className = 'onnx-model-desc';
    desc.textContent = hint.runtime;

    modelRow.appendChild(row);
    modelRow.appendChild(desc);
    onnxModelList.appendChild(modelRow);
  }
}

function normalizeWebGpuScale(value) {
  const scale = Number(value);
  return scale === 2 || scale === 3 || scale === 4 ? scale : DEFAULT_WEBGPU_SCALE;
}

function normalizeWebGpuModel(value) {
  const normalized = String(value || '').trim();
  return WEBGPU_MODEL_VALUES.has(normalized) ? normalized : DEFAULT_WEBGPU_MODEL;
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

async function persistSettingAndRefresh(patch) {
  await chrome.storage.sync.set(patch);
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
  if (shaderBackendSelect) {
    const webgpuOption = shaderBackendSelect.querySelector('option[value="webgpu"]');
    if (webgpuOption) {
      webgpuOption.disabled = !supported;
    }
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

function setActiveShaderBackendPanel(backend) {
  const normalizedBackend = backend === 'webgpu' ? 'webgpu' : 'webgl';
  document.querySelectorAll('.shader-backend-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `shaderBackendPanel-${normalizedBackend}`);
  });

  if (shaderBackendSelect) {
    shaderBackendSelect.value = normalizedBackend;
  }
}

function setActiveEnginePanel(backend) {
  const normalizedBackend = String(backend || DEFAULT_ENGINE_BACKEND).toLowerCase();
  const isShaderBackend = normalizedBackend === 'webgl' || normalizedBackend === 'webgpu';
  const activePanelId = isShaderBackend ? 'enginePanel-shaders' : `enginePanel-${normalizedBackend}`;
  const activeTabKey = isShaderBackend ? 'shaders' : normalizedBackend;
  const panels = document.querySelectorAll('.engine-panel');
  const tabs = document.querySelectorAll('.tab-pill');

  panels.forEach((panel) => {
    panel.classList.toggle('active', panel.id === activePanelId);
  });

  tabs.forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.tab === activeTabKey);
  });

  if (isShaderBackend) {
    setActiveShaderBackendPanel(normalizedBackend);
  }
}

async function loadCurrentSettings() {
  isWebGpuSupported = await detectWebGpuSupport();

  const result = await chrome.storage.sync.get({
    [SIMPLE_PRESET_KEY]: DEFAULT_SIMPLE_PRESET,
    [ENGINE_BACKEND_KEY]: DEFAULT_ENGINE_BACKEND,
    [WEBGPU_MODEL_KEY]: DEFAULT_WEBGPU_MODEL,
    [WEBGPU_SCALE_KEY]: DEFAULT_WEBGPU_SCALE,
    [ONNX_MODEL_KEY]: DEFAULT_ONNX_MODEL
  });

  const currentPreset = String(result[SIMPLE_PRESET_KEY] || DEFAULT_SIMPLE_PRESET).toUpperCase();
  let currentBackend = String(result[ENGINE_BACKEND_KEY] || DEFAULT_ENGINE_BACKEND).toLowerCase();
  const currentWebGpuModel = normalizeWebGpuModel(result[WEBGPU_MODEL_KEY]);
  const currentWebGpuScale = normalizeWebGpuScale(result[WEBGPU_SCALE_KEY]);
  const currentOnnxModel = normalizeOnnxModel(result[ONNX_MODEL_KEY]);

  if (!isWebGpuSupported && currentBackend === 'webgpu') {
    currentBackend = 'webgl';
    chrome.storage.sync.set({ [ENGINE_BACKEND_KEY]: currentBackend });
  }

  if (shaderBackendSelect) {
    shaderBackendSelect.value = currentBackend === 'webgpu' ? 'webgpu' : 'webgl';
  }

  const presetRadio = document.querySelector(`input[name="preset"][value="${currentPreset}"]`);
  if (presetRadio) presetRadio.checked = true;

  const webgpuModelRadio = document.querySelector(`input[name="webgpuModel"][value="${currentWebGpuModel}"]`);
  if (webgpuModelRadio) webgpuModelRadio.checked = true;

  if (webgpuScaleSelect) {
    webgpuScaleSelect.value = String(currentWebGpuScale);
  }

  buildOnnxModelSelectOptions(currentOnnxModel);

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
    const tabKey = String(tab.dataset.tab || 'shaders').toLowerCase();
    let backend = tabKey;

    if (tabKey === 'shaders') {
      const selectedShaderBackend = shaderBackendSelect?.value === 'webgpu' ? 'webgpu' : 'webgl';
      if (!isWebGpuSupported && selectedShaderBackend === 'webgpu') {
        backend = 'webgl';
      } else {
        backend = selectedShaderBackend;
      }
    }

    setActiveEnginePanel(backend);
    await persistSettingAndRefresh({ [ENGINE_BACKEND_KEY]: backend });
  });
});

if (shaderBackendSelect) {
  shaderBackendSelect.addEventListener('change', async (event) => {
    let backend = String(event.target?.value || 'webgl').toLowerCase();
    if (!isWebGpuSupported && backend === 'webgpu') {
      backend = 'webgl';
      shaderBackendSelect.value = 'webgl';
    }

    setActiveEnginePanel(backend);
    await persistSettingAndRefresh({ [ENGINE_BACKEND_KEY]: backend });
  });
}

document.querySelectorAll('input[name="webgpuModel"]').forEach((radio) => {
  radio.addEventListener('change', async (e) => {
    if (e.target.checked) {
      const nextModel = normalizeWebGpuModel(e.target.value);
      await persistSettingAndRefresh({ [WEBGPU_MODEL_KEY]: nextModel });
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

if (onnxModelList) {
  onnxModelList.addEventListener('change', async (event) => {
    const target = event.target;
    if (!target || target.name !== 'onnxModel' || !target.checked) return;

    const nextModel = normalizeOnnxModel(target.value);
    await persistSettingAndRefresh({ [ONNX_MODEL_KEY]: nextModel });
  });
}

if (chrome?.storage?.onChanged) {
  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== 'sync') return;
    if (changes[ONNX_MODEL_KEY] && onnxModelList) {
      buildOnnxModelSelectOptions(changes[ONNX_MODEL_KEY].newValue);
    }
  });
}

loadCurrentSettings();
refreshCacheActionAvailability();

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
    }
  });
}