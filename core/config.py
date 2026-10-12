"""Configuración central del backend. Todo se puede sobreescribir con variables de entorno."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
# backend/.env (opcional) para ajustar cada equipo sin tocar código; ver .env.example
load_dotenv(BASE_DIR / ".env")
ASSETS_DIR = BASE_DIR / "assets"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"
FONTS_DIR = ASSETS_DIR / "fonts"
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"

for _d in (BACKGROUNDS_DIR, FONTS_DIR, TEMP_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# "auto" = el mayor modelo que cabe en el equipo y está descargado (ver hardware.apply_profile)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "auto")

# "auto" = small en CPU; medium o large-v3 según la VRAM de la GPU NVIDIA
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "auto")
# 1 = decodificación voraz: ~30 % más rápido que 5 con la misma calidad práctica en "small"
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto | cuda | cpu
# Hilos de CPU para Whisper (0 = núcleos físicos detectados)
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "0"))
# Duración de cada bloque de transcripción en segundos ("auto" = según la RAM del equipo)
TRANSCRIBE_CHUNK_SECONDS = os.getenv("TRANSCRIBE_CHUNK_SECONDS", "auto")
# Mantener Whisper en memoria entre tareas (más rápido, pero ~0,5–1 GB de RAM ocupados).
# None ("auto") se resuelve al arrancar: sí con 16 GB de RAM o GPU NVIDIA
_keep = os.getenv("WHISPER_KEEP_LOADED", "auto")
WHISPER_KEEP_LOADED: bool | None = None if _keep == "auto" else _keep == "1"

# Prioridad del proceso: auto (por debajo de lo normal, el equipo sigue usable) | normal
PROCESS_PRIORITY = os.getenv("PROCESS_PRIORITY", "auto")

NUM_CLIPS = int(os.getenv("NUM_CLIPS", "3"))
# Duración máxima aceptada (los VODs largos se procesan enteros: audio completo + IA)
MAX_VIDEO_HOURS = float(os.getenv("MAX_VIDEO_HOURS", "12"))
# Análisis por bloques de la transcripción (para no saturar el contexto del LLM local)
ANALYSIS_CHUNK_MINUTES = float(os.getenv("ANALYSIS_CHUNK_MINUTES", "15"))
ANALYSIS_OVERLAP_MINUTES = float(os.getenv("ANALYSIS_OVERLAP_MINUTES", "1"))
CLIP_MIN_SECONDS = 30
CLIP_MAX_SECONDS = 60

# Capa de audio: pitch shift sin cambiar el tempo (1.0 = desactivado)
AUDIO_PITCH = float(os.getenv("AUDIO_PITCH", "1.03"))

# Tamaño final 9:16 y de cada mitad del split-screen
OUT_W, OUT_H = 1080, 1920
HALF_H = OUT_H // 2

# Codificador de video: auto | h264_nvenc | h264_amf | h264_qsv | libx264
VIDEO_ENCODER = os.getenv("VIDEO_ENCODER", "auto")
# Bitrate del video final: 3500k ≈ 13 MB por short de 30 s (más alto = más nitidez y peso)
VIDEO_BITRATE = os.getenv("VIDEO_BITRATE", "3500k")

# Caracteres máximos de transcripción enviados a Ollama (videos largos se muestrean)
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "40000"))

SUBTITLE_FONT = os.getenv("SUBTITLE_FONT", "Montserrat Black")

# Borrar el video/audio descargados al terminar (se conservan los JSON)
CLEANUP_TEMP = os.getenv("CLEANUP_TEMP", "1") == "1"

FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
