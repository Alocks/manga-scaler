## !! Be aware. This is only worth on 2K/4K monitors. 
Try checking Before and After sample bellow and check if it is really worth. 
I tested in my 1080p monitor and the difference is very small compared to my 4K monitor.

## manga-scaler
A GPU-accelerated Chrome extension designed to upscale and sharpen images in real-time. With processing speeds as low as 100ms, the enhancement is seamless and unnoticeable. It is ideal for 4K monitor users looking to improve image clarity for reading.

Currently works for the following websites:
- Nhentai
- Comix
- MangaDex
- Mangakakalot
- Manganato

Available engines and profiles (Refactor in progress. I may remove WebGPU and WebGL entirely if my onnxruntime implementation works better using Anime4K model using ONNX):
- **WebGL** Low performance [Uses Anime4k model]
- **WebGPU** High performance [Uses Anime4k model] *(compatibility may vary on browser)*
- **ONNX** GPU Heavy, but best option. [Uses few models. Still implementing]
<img width="844" height="590" alt="image" src="https://github.com/user-attachments/assets/71ff345a-34bf-4c8c-a469-c39de40cab01" />

## Models available for ONNX
- RealESR General x4 v3
- RealESRGAN 2x+
- RealCUGAN Conservative
- RealCUGAN 2X Latest Denoise 1x
- More SoonTM

## Before
<img width="532" height="398" alt="image" src="https://github.com/user-attachments/assets/0700422a-7543-42e0-927c-d1cf2b4c460b" />

## After
<img width="532" height="398" alt="image" src="https://github.com/user-attachments/assets/891cf80f-7c7a-4efd-a2bd-22fd02e89442" />

## How to build the extension locally

Both scripts generate artifacts inside the `dist` folder:
- `dist/manga-scaler.zip`
- `dist/manga-scaler.crx` (if browser packing is available)

Optional signing key:
- You can provide the private key using the environment variable `CRX_PRIVATE_KEY` or creating a .env file inside the project root.
- If `CRX_PRIVATE_KEY` is not provided, the scripts still build and attempt unsigned CRX packing.

### Windows (build.cmd)

Requirements:
- Git
- Node.js + npm
- Yarn
- PowerShell
- Microsoft Edge (for CRX packing)

Run from the repository root:

```bat
build.cmd
```

### Linux/macOS (build.sh)

Requirements:
- Git
- Node.js + npm
- Yarn
- zip
- Chrome (`google-chrome`, `google-chrome-stable`, or `chrome`) for CRX packing

Run from the repository root:

```bash
chmod +x build.sh
./build.sh
```

