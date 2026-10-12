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


def transcribe_chunk_seconds() -> int:
    """Bloques más largos (menos cortes) cuanto más RAM hay. ~280 MB por cada 5 min en CPU."""
    ram = total_ram_gb()
    if ram >= 30:
        return 1800
    if ram >= 15:
        return 900
    if ram >= 7.5:
        return 600
    return 300


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
        "cuda": cuda_available(),
        "whisper_device": whisper_device,
        "whisper_compute_type": whisper_compute_type(whisper_device),
        "transcribe_chunk_seconds": transcribe_chunk_seconds(),
        "video_encoder": encoder,
    }
