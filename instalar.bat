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

rem Modelo de IA recomendado para este equipo (7b con 16 GB de RAM o GPU de 8 GB, si no 3b)
for /f %%m in ('.venv\Scripts\python -c "from core import hardware; print(hardware.recommended_ollama_model({}))"') do set MODEL=%%m
if not defined MODEL set MODEL=qwen2.5:3b
rem Recien instalado con winget, Ollama aun no esta en el PATH de esta consola
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
where ollama >nul 2>nul && set OLLAMA=ollama
if not "%OLLAMA%"=="ollama" if not exist "%OLLAMA%" (
  echo.
  echo AVISO: Ollama no esta instalado. Instalalo con:  winget install Ollama.Ollama
  echo y despues descarga el modelo:  ollama pull %MODEL%
  goto :done
)
echo === Modelo de IA para este equipo: %MODEL% ===
"%OLLAMA%" pull %MODEL%
:done
echo.
echo Backend instalado. Ejecuta iniciar.bat
exit /b 0

:error
echo.
echo ERROR durante la instalacion. Revisa el mensaje anterior.
exit /b 1
