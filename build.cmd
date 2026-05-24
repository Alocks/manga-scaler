@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "DIST_DIR=dist"
set "TMP_DIR=tmp"
set "PAYLOAD_DIR=%TMP_DIR%\payload"
set "ANIME4K_SRC_DIR=%TMP_DIR%\anime4k-src"

echo [build] Starting local extension build...

call :require_cmd git || goto :fail
call :require_cmd node || goto :fail
call :require_cmd npm || goto :fail
call :require_cmd yarn || goto :fail
call :require_cmd powershell || goto :fail

if exist "%DIST_DIR%" (
  rmdir /s /q "%DIST_DIR%"
  if exist "%DIST_DIR%" (
    echo [build] Could not remove %DIST_DIR%. Close files/processes using it and try again.
    goto :fail
  )
)
if exist "%TMP_DIR%" (
  rmdir /s /q "%TMP_DIR%"
  if exist "%TMP_DIR%" (
    echo [build] Could not remove %TMP_DIR%. Close files/processes using it and try again.
    goto :fail
  )
)
mkdir "%DIST_DIR%" || goto :fail

echo [build] Installing anime4k-webgpu...
call npm install anime4k-webgpu || goto :fail

echo [build] Building runtime bundle...
call node tools\build-runtime-bundle.mjs || goto :fail

echo [build] Building Anime4K.js (webgl runtime)...
mkdir "%TMP_DIR%" || goto :fail
call git clone https://github.com/monyone/Anime4K.js "%ANIME4K_SRC_DIR%" || goto :fail
pushd "%ANIME4K_SRC_DIR%" || goto :fail
call yarn install || goto :fail
call yarn build >nul 2>&1 || goto :fail
popd || goto :fail

echo [build] Preparing minimal extension payload...
mkdir "%PAYLOAD_DIR%" || goto :fail
mkdir "%PAYLOAD_DIR%\src" || goto :fail
mkdir "%PAYLOAD_DIR%\src\runtime" || goto :fail
for %%F in (manifest.json content.js popup.html popup.js rules.json LICENSE) do (
  copy /y "%%F" "%PAYLOAD_DIR%\" >nul || goto :fail
)
copy /y src\runtime\runtime.bundle.js "%PAYLOAD_DIR%\src\runtime\runtime.bundle.js" >nul || goto :fail

mkdir "%PAYLOAD_DIR%\node_modules\anime4k-webgpu\lib" || goto :fail
robocopy node_modules\anime4k-webgpu\lib "%PAYLOAD_DIR%\node_modules\anime4k-webgpu\lib" /E /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 goto :fail

mkdir "%PAYLOAD_DIR%\node_modules\anime4k-webgl" || goto :fail
copy /y "%ANIME4K_SRC_DIR%\dist\anime4k.js" "%PAYLOAD_DIR%\node_modules\anime4k-webgl\anime4k.js" >nul || goto :fail

echo [build] Verifying required payload files...
for %%F in (
  "manifest.json"
  "LICENSE"
  "content.js"
  "popup.html"
  "popup.js"
  "rules.json"
  "src\runtime\runtime.bundle.js"
  "node_modules\anime4k-webgpu\lib\index.js"
  "node_modules\anime4k-webgl\anime4k.js"
) do (
  if not exist "%PAYLOAD_DIR%\%%~F" goto :missing_payload
)

echo [build] Creating ZIP artifact...
powershell -NoProfile -Command "if (Test-Path '%DIST_DIR%\manga-scaler.zip') { Remove-Item '%DIST_DIR%\manga-scaler.zip' -Force }; Get-ChildItem '%PAYLOAD_DIR%' | Compress-Archive -DestinationPath '%DIST_DIR%\manga-scaler.zip' -Force" || goto :fail
if not exist "%DIST_DIR%\manga-scaler.zip" goto :fail

set "EDGE_EXE="
if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" set "EDGE_EXE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if not defined EDGE_EXE if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set "EDGE_EXE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not defined EDGE_EXE (
  for /f "delims=" %%I in ('where msedge.exe 2^>nul') do if not defined EDGE_EXE set "EDGE_EXE=%%I"
)

if defined EDGE_EXE (
  echo [build] Packing CRX artifact...
  set "PACK_KEY_PATH="
  set "KEY_SOURCE_LABEL="
  if defined CRX_PRIVATE_KEY (
    powershell -NoProfile -Command "$ErrorActionPreference='Stop'; [System.IO.File]::WriteAllText('%TMP_DIR%\\extension-key.pem', $env:CRX_PRIVATE_KEY)" || goto :fail
    set "PACK_KEY_PATH=%CD%\%TMP_DIR%\extension-key.pem"
    set "KEY_SOURCE_LABEL=CRX_PRIVATE_KEY environment variable"
  )

  if not defined PACK_KEY_PATH if exist .env (
    call :load_key_from_env
    if exist "%TMP_DIR%\extension-key.pem" (
      set "PACK_KEY_PATH=%CD%\%TMP_DIR%\extension-key.pem"
      set "KEY_SOURCE_LABEL=.env file ^(CRX_PRIVATE_KEY^)"
    )
  )

  if not defined PACK_KEY_PATH if exist dist.pem (
    move /y dist.pem "%TMP_DIR%\extension-key.pem" >nul || goto :fail
    set "PACK_KEY_PATH=%CD%\%TMP_DIR%\extension-key.pem"
    set "KEY_SOURCE_LABEL=local dist.pem file ^(moved to tmp^)"
  )

  if not defined PACK_KEY_PATH (
    set "KEY_SOURCE_LABEL=none ^(creating unsigned/random-key CRX^)"
  )

  if defined KEY_SOURCE_LABEL (
    echo [build] Key source: !KEY_SOURCE_LABEL!.
  )

  if defined PACK_KEY_PATH (
    "%EDGE_EXE%" --headless --pack-extension="%CD%\%PAYLOAD_DIR%" --pack-extension-key="%PACK_KEY_PATH%"
  ) else (
    "%EDGE_EXE%" --headless --pack-extension="%CD%\%PAYLOAD_DIR%"
  )
  if exist "%PAYLOAD_DIR%.crx" move /y "%PAYLOAD_DIR%.crx" "%DIST_DIR%\manga-scaler.crx" >nul
) else (
  echo [build] Skipping CRX pack ^(Edge not found^).
)

if exist "%TMP_DIR%" rmdir /s /q "%TMP_DIR%"

echo [build] Done. Artifacts:
echo   - %CD%\%DIST_DIR%\manga-scaler.zip
if exist "%DIST_DIR%\manga-scaler.crx" echo   - %CD%\%DIST_DIR%\manga-scaler.crx
exit /b 0

:missing_payload
echo [build] Missing required files in payload.
goto :fail

:require_cmd
where %1 >nul 2>nul || (
  echo [build] Missing required command: %1
  exit /b 1
)
exit /b 0

:load_key_from_env
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $raw=Get-Content '.env' -Raw; $m=[regex]::Match($raw,'(?s)CRX_PRIVATE_KEY\s*=\s*\"(.*?)\"'); if($m.Success){ [System.IO.File]::WriteAllText('%TMP_DIR%\\extension-key.pem',$m.Groups[1].Value) }" >nul 2>&1
if errorlevel 1 (
  if exist "%TMP_DIR%\extension-key.pem" del /q "%TMP_DIR%\extension-key.pem" >nul 2>&1
)
exit /b 0

:fail
echo [build] Build failed.
exit /b 1
