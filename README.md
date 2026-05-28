## manga-scaler
A GPU-accelerated Chrome extension designed to upscale and sharpen images in real-time. With processing speeds as low as 100ms, the enhancement is seamless and unnoticeable. It is ideal for 4K monitor users looking to improve image clarity for reading.
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/de08d44f-b7f5-4f9e-863b-2c09cf0cfe69" alt="Original">Original</td>
    <td><img src="https://github.com/user-attachments/assets/d4d20b3c-baf5-47c2-b44e-c6dd8fe5f378" alt="Anime4K">Anime4K</td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/801b492d-1742-4191-a7ea-9478361bbebb" alt="RealCUGAN">RealCUGAN</td>
    <td><img src="https://github.com/user-attachments/assets/69f7fe8e-aceb-4110-a3e7-296e49226848" alt="MangaJaNai">MangaJaNai</td>
  </tr>
</table>  

### Currently works for the following websites:

- **Nhentai**
- **Comix**
- **MangaDex**
- **Mangakakalot**
- **Manganato**

### Available engines and profiles:
- **Shaders** *Best performance for low-end computers*
- **ONNX** *GPU Heavy, but best quality. [Uses AI models]*

*All included ONNX models have been fine-tuned specifically for this extension.  
If you choose to use them in other applications USE AT YOUR OWN RISK.*

## Minimum Requirements to use ONNX:
- **NVIDIA RTX 20 Series or newer** *note: GTX 10 series emulates FP16 and is hella slow*
- **AMD RX 5000 Series or newer**
- **Apple M1 or newer**
- **Intel 11th Gen or newer**

## Models available for ONNX
- **RealESRGAN 2x+**
- **RealESR AnimeVideo v3**
- **MangaJaNai 2x 1200p V1**
- **RealCUGAN Conservative**
- **RealCUGAN 2X Latest Denoise 1x**

# How to build the extension locally

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

