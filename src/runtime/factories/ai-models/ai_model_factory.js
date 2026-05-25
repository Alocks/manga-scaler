// Shared helpers for AI model processing factories.

function callAiModelFactoryHook(fn) {
    if (typeof fn !== 'function') return;
    try {
        fn();
    } catch {}
}

function createAiModelDisposeFromMethods(target, methodNames = []) {
    return function disposeAiModelMethods() {
        for (const methodName of methodNames) {
            callAiModelFactoryHook(target?.[methodName]);
        }
    };
}

function createAiModelFactory({
    providerName,
    kind,
    modelName,
    createResources
}) {
    if (!providerName || typeof providerName !== 'string') {
        throw new Error('createAiModelFactory requires providerName');
    }
    if (typeof createResources !== 'function') {
        throw new Error(`createAiModelFactory requires createResources() for provider ${providerName}`);
    }

    return {
        providerName,
        kind: kind || 'unknown',
        modelName: modelName || 'default',
        createResources(args) {
            const created = createResources(args || {});
            if (!created || typeof created !== 'object') {
                throw new Error(`AI model factory ${providerName} returned an invalid resource descriptor`);
            }
            if (typeof created.runPass !== 'function') {
                throw new Error(`AI model factory ${providerName} must provide runPass()`);
            }
            if (!created.outputTexture) {
                throw new Error(`AI model factory ${providerName} must provide outputTexture`);
            }

            return {
                ...created,
                modelUsed: created.modelUsed || modelName,
                dispose: typeof created.dispose === 'function' ? created.dispose : null
            };
        }
    };
}
