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

Available engines and profiles:
- **Shaders** Best performance for low-end computer
- **ONNX** GPU Heavy, but best option. [Use AI models]
- <img width="1316" height="1394" alt="image" src="https://github.com/user-attachments/assets/c9b7d0d2-2d07-4457-b2fd-fe9b689a0991" />


## Models available for ONNX
- RealESRGAN 2x+
- RealESR AnimeVideo v3
- MangaJaNai 2x 1200p V1
- RealCUGAN Conservative
- RealCUGAN 2X Latest Denoise 1x
- AnimeSharp V2 MoSR Sharp
- AnimeSharp V2 RPLKSR Sharp

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

