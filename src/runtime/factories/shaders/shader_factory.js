// Shared helpers for runtime shader processing factories.

function callShaderFactoryHook(fn) {
    if (typeof fn !== 'function') return;
    try {
        fn();
    } catch {}
}

function createShaderDisposeFromMethods(target, methodNames = []) {
    return function disposeShaderMethods() {
        for (const methodName of methodNames) {
            callShaderFactoryHook(target?.[methodName]);
        }
    };
}

function createShaderFactory({
    providerName,
    kind,
    modelName,
    createResources
}) {
    if (!providerName || typeof providerName !== 'string') {
        throw new Error('createShaderFactory requires providerName');
    }
    if (typeof createResources !== 'function') {
        throw new Error(`createShaderFactory requires createResources() for provider ${providerName}`);
    }

    return {
        providerName,
        kind: kind || 'unknown',
        modelName: modelName || 'default',
        createResources(args) {
            const created = createResources(args || {});
            if (!created || typeof created !== 'object') {
                throw new Error(`Shader factory ${providerName} returned an invalid resource descriptor`);
            }
            if (typeof created.runPass !== 'function') {
                throw new Error(`Shader factory ${providerName} must provide runPass()`);
            }
            if (!created.outputTexture) {
                throw new Error(`Shader factory ${providerName} must provide outputTexture`);
            }

            return {
                ...created,
                modelUsed: created.modelUsed || modelName,
                dispose: typeof created.dispose === 'function' ? created.dispose : null
            };
        }
    };
}

// Backwards-compatibility aliases for older references.
const createAiModelFactory = createShaderFactory;
const createAiModelDisposeFromMethods = createShaderDisposeFromMethods;
