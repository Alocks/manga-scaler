// Flush ONNX GPU VRAM when tab is hidden (switch away)
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
        if (typeof window.resetOnnxAdapterState === 'function') {
            try {
                window.resetOnnxAdapterState();
                window.MangaScalerLog?.('onnx:vram-flush', { reason: 'tab-hidden' });
            } catch (err) {
                window.MangaScalerLog?.('onnx:vram-flush-error', { error: String(err) });
            }
        }
    }
});
// Bootstrap entrypoint: logger setup and CSS injection

const DEBUG = false;
const ENABLE_WEBGPU_PROFILING = false;
const ENABLE_ONNX_PROFILING = false;
const ALWAYS_LOG_LABELS = new Set([
    'bg-process:skip-cached',
    'bg-process:upscale-time',
    'process:upscale-time',
    'process:apply-cached',
    'process:cache-write',
    'cache:memory-hit',
    'onnx:run-tiled-start',
    'onnx:tiled-layout',
    'onnx:tiled-run-success',
    'onnx:run-success',
    'onnx:run-failed',
    'onnx:pass-profile',
    'onnx:worker-pass-inference-complete',
    'onnx:worker-pass-still-running'
]);

function bootstrapLog(label, data = {}) {
    const isProfileLabel = label.startsWith('profile:');
    const isOnnxProfileLabel = label.startsWith('profile:onnx');
    const profilingEnabled = (ENABLE_WEBGPU_PROFILING && isProfileLabel && !isOnnxProfileLabel) || (ENABLE_ONNX_PROFILING && isOnnxProfileLabel);
    if (!DEBUG && !profilingEnabled && !ALWAYS_LOG_LABELS.has(label)) return;
    console.log('[Manga Scaler]', label, { ts: new Date().toISOString(), ...data });
}

window.MangaScalerDebugEnabled = DEBUG;
window.MangaScalerLog = bootstrapLog;
window.MangaScalerProfiling = {
    enabled: ENABLE_WEBGPU_PROFILING,
    onnxEnabled: ENABLE_ONNX_PROFILING,
    isEnabled() {
        return ENABLE_WEBGPU_PROFILING;
    },
    isOnnxEnabled() {
        return ENABLE_ONNX_PROFILING;
    }
};

if (!document.querySelector('style[data-ai-scaler]')) {
    const style = document.createElement('style');
    style.setAttribute('data-ai-scaler', 'true');
    style.textContent = `
        #image-container {
            overflow: hidden;
        }
    `;
    document.head.appendChild(style);
}
