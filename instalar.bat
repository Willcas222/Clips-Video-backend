@echo off
setlocal
cd /d "%~dp0"
echo === Entorno de Python ===
if not exist .venv python -m venv .venv || goto :error
.venv\Scripts\python -m pip install --upgrade pip || goto :error
.venv\Scripts\pip install -r requirements.txt || goto :error
if not exist .env copy .env.example .env >nul

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo === GPU NVIDIA detectada: instalando librerias CUDA para Whisper ===
  .venv\Scripts\pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
)

echo === Videos de fondo (gameplay, ~2.3 GB) ===
.venv\Scripts\python descargar_fondos.py

where ollama >nul 2>nul
if %errorlevel% neq 0 (
  echo.
  echo AVISO: Ollama no esta instalado. Instalalo con:  winget install Ollama.Ollama
  echo y despues descarga el modelo:  ollama pull qwen2.5:3b
) else (
  ollama pull qwen2.5:3b
)
echo.
echo Backend instalado. Ejecuta iniciar.bat
exit /b 0

:error
echo.
echo ERROR durante la instalacion. Revisa el mensaje anterior.
exit /b 1
