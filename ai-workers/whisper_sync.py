"""Transcribe audio/video con timestamps (faster-whisper).

Uso:
  python whisper_sync.py --input video.mp4 --model small --language es
Salida: {"ok": true, "language": "es", "text": "...", "segments": [{"start", "end", "text", "words"}]}
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np

from common import log, run_main

SAMPLE_RATE = 16000


def load_audio(path: str, ffmpeg: str) -> np.ndarray:
    """Decodifica a PCM mono 16 kHz con FFmpeg (evita depender de la versión de PyAV)."""
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg no pudo extraer el audio: {proc.stderr.decode(errors='ignore')[-400:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="small", help="tiny | base | small | medium | large-v3")
    parser.add_argument("--language", default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Ruta al binario de FFmpeg")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock:
        return {
            "language": args.language or "es",
            "text": "Texto de prueba",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Texto de prueba", "words": []}],
        }

    from faster_whisper import WhisperModel

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    audio = load_audio(args.input, args.ffmpeg)
    if audio.size < SAMPLE_RATE // 2:
        return {"language": args.language or "", "text": "", "segments": []}

    log(f"Cargando Whisper '{args.model}' en {device} ({compute_type})...")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        audio,
        language=args.language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    log(f"Idioma detectado: {info.language} ({info.language_probability:.0%}), duración {info.duration:.1f}s")

    segments = []
    for seg in segments_iter:
        segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "words": [
                    {"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()}
                    for w in (seg.words or [])
                ],
            }
        )
        log(f"  [{seg.start:6.1f}s] {seg.text.strip()}")

    return {
        "language": info.language,
        "text": " ".join(s["text"] for s in segments).strip(),
        "segments": segments,
    }


if __name__ == "__main__":
    run_main(main)
