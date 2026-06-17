#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DIST_DIR="$ROOT_DIR/dist"
TMP_DIR="$ROOT_DIR/tmp"
PAYLOAD_DIR="$TMP_DIR/payload"
ANIME4K_SRC_DIR="$TMP_DIR/anime4k-src"
KEY_FILE_TMP="$TMP_DIR/extension-key.pem"

echo "[build] Starting local extension build..."

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[build] Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd node
require_cmd npm
require_cmd yarn
require_cmd zip

KEY_FROM_ENV_FILE=0
if [[ -z "${CRX_PRIVATE_KEY:-}" && -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  if [[ -n "${CRX_PRIVATE_KEY:-}" ]]; then
    KEY_FROM_ENV_FILE=1
  fi
fi

rm -rf "$DIST_DIR" "$TMP_DIR"
mkdir -p "$DIST_DIR"

echo "[build] Installing npm dependencies..."
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi

echo "[build] Building runtime bundle..."
node tools/build-runtime-bundle.mjs

echo "[build] Building Anime4K.js (webgl runtime)..."
mkdir -p "$TMP_DIR"
git clone https://github.com/monyone/Anime4K.js "$ANIME4K_SRC_DIR"
pushd "$ANIME4K_SRC_DIR" >/dev/null
yarn install
yarn build >/dev/null 2>&1
popd >/dev/null

echo "[build] Preparing minimal extension payload..."
mkdir -p "$PAYLOAD_DIR/src/runtime"
mkdir -p "$PAYLOAD_DIR/src/workers"
for file in manifest.json content.js popup.html popup.js rules.json LICENSE; do
  cp "$file" "$PAYLOAD_DIR/"
done
cp src/runtime/runtime.bundle.js "$PAYLOAD_DIR/src/runtime/runtime.bundle.js"
cp src/workers/onnxruntime.worker.js "$PAYLOAD_DIR/src/workers/onnxruntime.worker.js"

mkdir -p "$PAYLOAD_DIR/node_modules/anime4k-webgpu/lib"
cp -R node_modules/anime4k-webgpu/lib/. "$PAYLOAD_DIR/node_modules/anime4k-webgpu/lib/"

mkdir -p "$PAYLOAD_DIR/models"
cp models/*.onnx "$PAYLOAD_DIR/models/"

mkdir -p "$PAYLOAD_DIR/node_modules/anime4k-webgl"

cp "$ANIME4K_SRC_DIR/dist/anime4k.js" "$PAYLOAD_DIR/node_modules/anime4k-webgl/anime4k.js"

mkdir -p "$PAYLOAD_DIR/node_modules/onnxruntime-web/dist"
cp node_modules/onnxruntime-web/dist/ort.all.min.js "$PAYLOAD_DIR/node_modules/onnxruntime-web/dist/ort.all.min.js"
cp node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs "$PAYLOAD_DIR/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs"
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs "$PAYLOAD_DIR/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs"
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm "$PAYLOAD_DIR/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm"

echo "[build] Verifying required payload files..."
required_files=(
  "$PAYLOAD_DIR/manifest.json"
  "$PAYLOAD_DIR/LICENSE"
  "$PAYLOAD_DIR/content.js"
  "$PAYLOAD_DIR/popup.html"
  "$PAYLOAD_DIR/popup.js"
  "$PAYLOAD_DIR/rules.json"
  "$PAYLOAD_DIR/src/runtime/runtime.bundle.js"
  "$PAYLOAD_DIR/src/workers/onnxruntime.worker.js"
  "$PAYLOAD_DIR/node_modules/anime4k-webgpu/lib/index.js"
  "$PAYLOAD_DIR/node_modules/anime4k-webgl/anime4k.js"
  "$PAYLOAD_DIR/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs"
  "$PAYLOAD_DIR/node_modules/onnxruntime-web/dist/ort.all.min.js"
  "$PAYLOAD_DIR/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs"
  "$PAYLOAD_DIR/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "[build] Missing required file: $file" >&2
    exit 1
  fi
done

for model in models/*.onnx; do
  model_basename="$(basename "$model")"
  if [[ ! -f "$PAYLOAD_DIR/models/$model_basename" ]]; then
    echo "[build] Missing required model in payload: $model_basename" >&2
    exit 1
  fi
done

echo "[build] Creating ZIP artifact..."
rm -f "$DIST_DIR/manga-scaler.zip"
(
  cd "$PAYLOAD_DIR"
  zip -rq "$DIST_DIR/manga-scaler.zip" .
)

if [[ ! -f "$DIST_DIR/manga-scaler.zip" ]]; then
  echo "[build] Failed to create $DIST_DIR/manga-scaler.zip" >&2
  exit 1
fi

CHROME_BIN=""
for candidate in google-chrome google-chrome-stable chrome; do
  if command -v "$candidate" >/dev/null 2>&1; then
    CHROME_BIN="$candidate"
    break
  fi
done

if [[ -n "$CHROME_BIN" ]]; then
  echo "[build] Packing CRX artifact..."
  PACK_KEY_FILE=""
  if [[ -n "${CRX_PRIVATE_KEY:-}" ]]; then
    printf '%s\n' "$CRX_PRIVATE_KEY" > "$KEY_FILE_TMP"
    PACK_KEY_FILE="$KEY_FILE_TMP"
    if [[ "$KEY_FROM_ENV_FILE" -eq 1 ]]; then
      echo "[build] Key source: .env file (CRX_PRIVATE_KEY)."
    else
      echo "[build] Key source: CRX_PRIVATE_KEY environment variable."
    fi
  elif [[ -f "$ROOT_DIR/dist.pem" ]]; then
    mv "$ROOT_DIR/dist.pem" "$KEY_FILE_TMP"
    PACK_KEY_FILE="$KEY_FILE_TMP"
    echo "[build] Key source: local dist.pem file (moved to tmp)."
  else
    echo "[build] Key source: none (creating unsigned/random-key CRX)."
  fi

  if [[ -n "$PACK_KEY_FILE" ]]; then
    "$CHROME_BIN" --headless --pack-extension="$PAYLOAD_DIR" --pack-extension-key="$PACK_KEY_FILE"
  else
    "$CHROME_BIN" --headless --pack-extension="$PAYLOAD_DIR"
  fi
  if [[ -f "$TMP_DIR/payload.crx" ]]; then
    mv "$TMP_DIR/payload.crx" "$DIST_DIR/manga-scaler.crx"
  fi
else
  echo "[build] Skipping CRX pack (Chrome binary not found)."
fi

rm -rf "$TMP_DIR"

echo "[build] Done. Artifacts:"
echo "  - $DIST_DIR/manga-scaler.zip"
if [[ -f "$DIST_DIR/manga-scaler.crx" ]]; then
  echo "  - $DIST_DIR/manga-scaler.crx"
fi
