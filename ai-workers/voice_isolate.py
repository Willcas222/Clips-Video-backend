"""Aísla la voz de un video/audio: quita música y ruido de fondo (Demucs htdemucs).

Uso:
  python voice_isolate.py --input clip.mp4 --output voz.wav --ffmpeg ruta/ffmpeg

El audio se decodifica y codifica con FFmpeg (no se necesita torchaudio).
Salida: {"ok": true, "output": ..., "engine": "demucs", "hasVoice": true|false}
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

from common import MODELS_DIR, log, run_main, torch_device

os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
SAMPLE_RATE = 44100


def decode(path: str, ffmpeg: str) -> np.ndarray:
    """Audio estéreo float32 [2, muestras] a 44,1 kHz."""
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg no pudo leer el audio: {proc.stderr.decode(errors='ignore')[-300:]}")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size < SAMPLE_RATE // 2:
        raise RuntimeError("El clip no tiene audio")
    return data.reshape(-1, 2).T.copy()


def encode_wav(audio: np.ndarray, path: str, ffmpeg: str) -> None:
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-i", "-", path]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(audio.T, dtype=np.float32).tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg no pudo escribir el audio: {proc.stderr.decode(errors='ignore')[-300:]}")


def main() -> dict:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    wav = decode(args.input, args.ffmpeg)
    log(f"Audio: {wav.shape[1] / SAMPLE_RATE:.1f} s. Separando la voz…")
    model = get_model("htdemucs")
    model.eval()
    device = torch_device()
    mix = torch.from_numpy(wav)
    # Normalización como en demucs.separate (mejora la separación con volúmenes extremos).
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.inference_mode():
        sources = apply_model(model, ((mix - mean) / std)[None].to(device), device=device, split=True, overlap=0.25, progress=False)[0]
    vocals = (sources[model.sources.index("vocals")] * std + mean).cpu().numpy()

    # ¿Hay voz? Compara la energía de la voz con la del audio original.
    voice_rms = float(np.sqrt(np.mean(vocals**2)))
    mix_rms = float(np.sqrt(np.mean(wav**2))) + 1e-9
    has_voice = voice_rms > 0.003 and voice_rms / mix_rms > 0.08
    log(f"Voz: {voice_rms:.4f} RMS ({voice_rms / mix_rms:.0%} del total){'' if has_voice else ' — prácticamente sin voz'}")

    encode_wav(np.clip(vocals, -1, 1), args.output, args.ffmpeg)
    return {"output": args.output, "engine": "demucs", "hasVoice": has_voice}


if __name__ == "__main__":
    run_main(main)
