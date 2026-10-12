"""Detección del hardware para usar la mejor configuración disponible en cada equipo.

Todo se decide al arrancar y se cachea. Cada valor se puede forzar con su variable de entorno
(ver config.py / .env.example); "auto" deja que se detecte.
"""
import logging
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

import psutil

from . import config
from .ffmpeg_utils import get_ffmpeg

log = logging.getLogger("shorts.hardware")

def _kbps(value: str) -> int:
    v = value.strip().lower()
    return int(float(v[:-1]) * 1000) if v.endswith("m") else int(v.rstrip("k"))


# Bitrate objetivo del short (1080x1920, 30 fps). 3,5 Mbps ≈ 13 MB por clip de 30 s, sin
# diferencia visible con calidad máxima (SSIM 0,98 medido con fondos de gameplay) y por encima
# de lo que las plataformas conservan al recomprimir. Los picos se limitan a +30 %.
_BR = _kbps(config.VIDEO_BITRATE)
_RATE = ["-b:v", f"{_BR}k", "-maxrate", f"{int(_BR * 1.3)}k", "-bufsize", f"{_BR * 2}k"]

# Orden de preferencia de codificadores por hardware (NVIDIA, AMD, Intel) y sus ajustes,
# todos con el mismo bitrate objetivo para obtener tamaños parecidos en cualquier equipo.
ENCODERS = {
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", *_RATE],
    "h264_amf": ["-c:v", "h264_amf", "-quality", "quality", "-rc", "vbr_peak", *_RATE],
    "h264_qsv": ["-c:v", "h264_qsv", "-preset", "medium", *_RATE],
    # CRF mantiene la calidad en escenas fáciles; el tope de bitrate limita las difíciles
    "libx264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", *_RATE[2:]],
}


def add_cuda_dll_dirs() -> None:
    """En Windows, registra las DLL de CUDA/cuDNN instaladas por pip (paquetes nvidia-*)."""
    if sys.platform != "win32":
        return
    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if base.is_dir():
        for bin_dir in base.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    add_cuda_dll_dirs()
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def whisper_compute_type(device: str) -> str:
    """Tipo de cálculo más rápido que soporta el dispositivo sin perder precisión apreciable."""
    import ctranslate2

    supported = ctranslate2.get_supported_compute_types(device)
    prefs = ("float16", "int8_float16", "int8_float32", "int8") if device == "cuda" else ("int8", "int8_float32", "float32")
    return next((t for t in prefs if t in supported), "default")


def cpu_threads() -> int:
    # Núcleos físicos: con hyperthreading, CTranslate2 no gana usando los lógicos
    return psutil.cpu_count(logical=False) or os.cpu_count() or 4


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / 2**30


def available_ram_gb() -> float:
    return psutil.virtual_memory().available / 2**30


# Umbrales sobre la RAM visible: con gráfica integrada, Windows reserva 1-2 GB de la instalada
# (un equipo de 16 GB puede mostrar 13,9 GB), así que cada "clase" admite ese margen.
RAM_8, RAM_16, RAM_32 = 6.5, 13.0, 26.0


def transcribe_chunk_seconds() -> int:
    """Bloques más largos (menos cortes) cuanto más RAM hay. ~280 MB por cada 5 min en CPU."""
    ram = total_ram_gb()
    if ram >= RAM_32:
        return 1800
    if ram >= RAM_16:
        return 900
    if ram >= RAM_8:
        return 600
    return 300


@lru_cache(maxsize=1)
def gpu_vram_gb() -> float:
    """VRAM de la GPU NVIDIA más grande (0 si no hay CUDA)."""
    if not cuda_available():
        return 0.0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(float(x) for x in out.split()) / 1024
    except Exception:
        return 0.0


# --- Perfil automático (valores "auto" del .env) ---

# Modelos de análisis de mayor a menor: (nombre, RAM visible mínima en CPU, VRAM mínima en GPU)
OLLAMA_LADDER = [("qwen2.5:7b", RAM_16, 7.0), ("qwen2.5:3b", RAM_8, 3.0), ("qwen2.5:1.5b", 0.0, 0.0)]


def ollama_models() -> dict[str, int]:
    """Modelos descargados en Ollama -> tamaño en bytes ({} si Ollama no responde)."""
    import httpx

    try:
        r = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        return {m["name"]: int(m.get("size", 0)) for m in r.json().get("models", [])}
    except Exception:
        return {}


def recommended_ollama_model(installed: dict[str, int]) -> str:
    """El mayor modelo que cabe en el equipo y está descargado. Si el ideal no está descargado
    se usa el mayor descargado de la escalera; si no hay ninguno, el ideal (health pedirá el pull)."""
    ram, vram = total_ram_gb(), gpu_vram_gb()
    names = [m for m, _, _ in OLLAMA_LADDER]
    fitting = [m for m, r, v in OLLAMA_LADDER if ram >= r or (vram and vram >= v)]
    for m in fitting:
        if m in installed:
            return m
    for m in names:
        if m in installed:
            return m
    return fitting[0]


def recommended_whisper_model() -> str:
    if config.WHISPER_DEVICE != "cpu" and cuda_available():
        vram = gpu_vram_gb()
        return "large-v3" if vram >= 8 else "medium" if vram >= 5 else "small"
    # En CPU "medium" es ~3 veces más lento que "small" para poca mejora: small salvo con poca RAM
    return "small" if total_ram_gb() >= RAM_8 - 2 else "base"


def recommended_keep_loaded() -> bool:
    # Whisper en GPU no ocupa RAM del sistema; en CPU solo compensa con 16 GB o más
    return total_ram_gb() >= RAM_16 or (config.WHISPER_DEVICE != "cpu" and gpu_vram_gb() >= 5)


def _lower_priority() -> str:
    """Prioridad por debajo de lo normal: Whisper y ffmpeg (que la heredan) solo usan la CPU que
    el resto de programas no necesita, así el equipo sigue respondiendo durante un render."""
    pref = config.PROCESS_PRIORITY
    if pref == "normal":
        return "normal"
    try:
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)
        return "below_normal"
    except Exception as exc:
        log.warning("No se pudo bajar la prioridad del proceso: %s", exc)
        return "normal"


def apply_profile() -> dict:
    """Resuelve los valores "auto" del .env según el equipo y los escribe en config.
    Se llama una vez al arrancar el servidor."""
    if config.OLLAMA_MODEL == "auto":
        config.OLLAMA_MODEL = recommended_ollama_model(ollama_models())
    if config.WHISPER_MODEL == "auto":
        config.WHISPER_MODEL = recommended_whisper_model()
    if config.WHISPER_KEEP_LOADED is None:
        config.WHISPER_KEEP_LOADED = recommended_keep_loaded()
    return {
        "ollama_model": config.OLLAMA_MODEL,
        "whisper_model": config.WHISPER_MODEL,
        "whisper_keep_loaded": config.WHISPER_KEEP_LOADED,
        "priority": _lower_priority(),
    }


def fit_ollama_model(model: str, free_memory: Callable[[], None]) -> str:
    """Antes de cada análisis: comprueba que el modelo cabe en la RAM libre en ese momento.
    Si no cabe, primero libera lo propio (Whisper) y, si sigue sin caber, baja a un modelo
    menor ya descargado en vez de hacer que el sistema pagine a disco."""
    if gpu_vram_gb() >= 3:
        return model  # Ollama lo carga en VRAM
    installed = ollama_models()
    if model not in installed:
        return model

    def need(m: str) -> float:  # pesos + caché KV y búferes de Ollama
        return installed[m] / 2**30 * 1.2 + 0.8

    if available_ram_gb() >= need(model):
        return model
    free_memory()
    free = available_ram_gb()
    if free >= need(model):
        log.info("RAM liberada para el análisis (%.1f GB libres).", free)
        return model
    for m, _, _ in OLLAMA_LADDER:
        if m in installed and installed[m] < installed[model] and free >= need(m):
            log.warning("Solo %.1f GB de RAM libres: el análisis usará '%s' en lugar de '%s'.", free, m, model)
            return m
    log.warning("Solo %.1f GB de RAM libres para '%s' (necesita ~%.1f GB): el análisis puede ir lento. "
                "Cierra otros programas si es posible.", free, model, need(model))
    return model


def _encoder_works(name: str) -> bool:
    args = [get_ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=s=1080x1920:d=0.3:r=30", *ENCODERS[name], "-pix_fmt", "yuv420p", "-f", "null", "-"]
    try:
        return subprocess.run(args, capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def video_encoder(preference: str = "auto") -> str:
    if preference != "auto":
        if preference not in ENCODERS:
            raise ValueError(f"VIDEO_ENCODER desconocido: {preference}. Opciones: auto, {', '.join(ENCODERS)}")
        return preference
    for name in ("h264_nvenc", "h264_amf", "h264_qsv"):
        if _encoder_works(name):
            return name
    return "libx264"


def summary(whisper_device: str, encoder: str) -> dict:
    return {
        "cpu_cores": cpu_threads(),
        "ram_gb": round(total_ram_gb(), 1),
        "ram_free_gb": round(available_ram_gb(), 1),
        "cuda": cuda_available(),
        "vram_gb": round(gpu_vram_gb(), 1),
        "ollama_model": config.OLLAMA_MODEL,
        "whisper_model": config.WHISPER_MODEL,
        "whisper_keep_loaded": config.WHISPER_KEEP_LOADED,
        "whisper_device": whisper_device,
        "whisper_compute_type": whisper_compute_type(whisper_device),
        "transcribe_chunk_seconds": transcribe_chunk_seconds(),
        "video_encoder": encoder,
    }
