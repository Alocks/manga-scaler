// Bootstrap entrypoint: logger setup and CSS injection

const DEBUG = false;
const ALWAYS_LOG_LABELS = new Set([
    'bg-process:skip-cached',
    'bg-process:upscale-time',
    'process:upscale-time'
]);

function bootstrapLog(label, data = {}) {
    if (!DEBUG && !ALWAYS_LOG_LABELS.has(label)) return;
    console.log('[Manga Scaler]', label, { ts: new Date().toISOString(), ...data });
}

window.MangaScalerLog = bootstrapLog;

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
