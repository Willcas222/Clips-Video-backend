"""Validación de dependencias externas (ffmpeg y Ollama)."""
import subprocess

import httpx

from . import config, hardware
from .ffmpeg_utils import get_ffmpeg


def check_ffmpeg() -> dict:
    try:
        exe = get_ffmpeg()
        out = subprocess.run([exe, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=15).stdout
        missing = [f for f in ("subtitles", "rubberband") if f" {f} " not in out]
        if missing:
            return {"ok": False, "path": exe, "error": f"ffmpeg no tiene los filtros: {', '.join(missing)}"}
        return {"ok": True, "path": exe}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_ollama() -> dict:
    try:
        r = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        has_model = any(m == config.OLLAMA_MODEL or m.startswith(config.OLLAMA_MODEL + ":") for m in models)
        res = {"ok": has_model, "host": config.OLLAMA_HOST, "model": config.OLLAMA_MODEL, "models": models}
        if not has_model:
            res["error"] = f"El modelo '{config.OLLAMA_MODEL}' no está descargado. Ejecuta: ollama pull {config.OLLAMA_MODEL}"
        return res
    except Exception:
        return {"ok": False, "host": config.OLLAMA_HOST,
                "error": f"Ollama no responde en {config.OLLAMA_HOST}. Instálalo desde https://ollama.com y ejecútalo."}


def check_hardware() -> dict:
    from .transcriber import chunk_seconds, resolve_device

    info = hardware.summary(resolve_device(), hardware.video_encoder(config.VIDEO_ENCODER))
    info["transcribe_chunk_seconds"] = chunk_seconds()
    return {"ok": True, **info}


def check_dependencies() -> dict:
    return {"ffmpeg": check_ffmpeg(), "ollama": check_ollama(), "hardware": check_hardware()}
