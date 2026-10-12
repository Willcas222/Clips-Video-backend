"""Fase 2.2: transcripción local con faster-whisper y timestamps por palabra."""
import gc
import json
import logging
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np
from faster_whisper import WhisperModel

from . import config, hardware

log = logging.getLogger("shorts.transcriber")

_model: WhisperModel | None = None
_model_device: str | None = None
_lock = threading.Lock()


def resolve_device() -> str:
    if config.WHISPER_DEVICE == "cpu":
        return "cpu"
    if hardware.cuda_available():
        return "cuda"
    if config.WHISPER_DEVICE == "cuda":
        log.warning("WHISPER_DEVICE=cuda pero no se detecta ninguna GPU CUDA. Usando CPU.")
    return "cpu"


def _load(device: str) -> WhisperModel:
    compute = hardware.whisper_compute_type(device)
    threads = config.WHISPER_CPU_THREADS or hardware.cpu_threads()
    log.info("Cargando faster-whisper '%s' en %s (%s, %d hilos)", config.WHISPER_MODEL, device, compute, threads)
    return WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute, cpu_threads=threads)


def _get_model(force_cpu: bool = False) -> WhisperModel:
    global _model, _model_device
    with _lock:
        if force_cpu and _model_device != "cpu":
            _model = None
        if _model is None:
            device = "cpu" if force_cpu else resolve_device()
            if device == "cuda":
                try:
                    _model, _model_device = _load("cuda"), "cuda"
                    return _model
                except Exception as exc:
                    log.warning("No se pudo cargar Whisper en CUDA (%s). Usando CPU.", exc)
            _model, _model_device = _load("cpu"), "cpu"
        return _model


def chunk_seconds() -> int:
    if config.TRANSCRIBE_CHUNK_SECONDS != "auto":
        return max(60, int(config.TRANSCRIBE_CHUNK_SECONDS))
    return hardware.transcribe_chunk_seconds()


SR = 16000
# Bloque de transcripción. faster-whisper calcula el espectrograma de todo lo que recibe de
# una vez (~1 GB de RAM por cada 30 min de audio), así que los videos largos se procesan por
# bloques leídos del disco: la memoria queda fija sea cual sea la duración.
# El tamaño del bloque se ajusta a la RAM del equipo (ver chunk_seconds()).
# Margen al final de cada bloque donde se busca el punto más silencioso para cortar
SPLIT_SEARCH_SECONDS = 10


def _open_wav(audio_path: Path) -> wave.Wave_read:
    # WAV mono 16 kHz pcm_s16le generado en la fase 2.1. Se lee a mano para no depender
    # del decodificador de faster-whisper (incompatible con PyAV >= 15).
    wf = wave.open(str(audio_path), "rb")
    if wf.getframerate() != SR or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        wf.close()
        raise ValueError("Se esperaba un WAV mono 16 kHz de 16 bits")
    return wf


def _read(wf: wave.Wave_read, start: int, n: int) -> np.ndarray:
    wf.setpos(start)
    audio = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    return audio


def _load_wav(audio_path: Path) -> np.ndarray:
    """Audio completo en float32 (solo para pruebas y utilidades; la transcripción va por bloques)."""
    with _open_wav(audio_path) as wf:
        return _read(wf, 0, wf.getnframes())


def _quietest_cut(audio: np.ndarray) -> int:
    """Índice de corte en la ventana de 0,2 s con menos energía dentro del margen final."""
    win = SR // 5
    search = min(len(audio), SPLIT_SEARCH_SECONDS * SR)
    tail = audio[len(audio) - search:]
    n = len(tail) // win
    if n < 2:
        return len(audio)
    energy = (tail[: n * win].reshape(n, win) ** 2).mean(axis=1)
    return len(audio) - search + int(energy.argmin()) * win + win // 2


ProgressFn = Callable[[float], None]


def _run(model: WhisperModel, audio_path: Path, on_progress: ProgressFn | None) -> tuple[list[dict], list[dict], str]:
    words, segs = [], []
    language = None
    with _open_wav(audio_path) as wf:
        total = wf.getnframes()
        pos = 0
        chunk = chunk_seconds()
        while pos < total:
            n = min(chunk * SR, total - pos)
            audio = _read(wf, pos, n)
            if pos + n < total:
                audio = audio[: _quietest_cut(audio)]
            offset = pos / SR
            segments, info = model.transcribe(
                audio, word_timestamps=True, vad_filter=True,
                beam_size=config.WHISPER_BEAM_SIZE, language=language,
            )
            for seg in segments:  # el generador ejecuta la inferencia aquí
                segs.append({"text": seg.text.strip(), "start": round(float(seg.start) + offset, 2),
                             "end": round(float(seg.end) + offset, 2)})
                for w in seg.words or []:
                    text = w.word.strip()
                    if text:
                        words.append({"word": text, "start": round(float(w.start) + offset, 3),
                                      "end": round(float(w.end) + offset, 3)})
                if on_progress:
                    on_progress(min(1.0, (offset + seg.end) / (total / SR)))
            # El idioma del primer bloque con voz se mantiene en el resto (coherencia y velocidad)
            if language is None and segs:
                language = info.language
            pos += len(audio)
            del audio
    return words, segs, language or "unknown"


def unload() -> None:
    """Libera el modelo de la RAM (se recarga en la próxima transcripción)."""
    global _model, _model_device
    with _lock:
        _model, _model_device = None, None
    gc.collect()


def transcribe(audio_path: Path, work_dir: Path, on_progress: ProgressFn | None = None) -> dict:
    """Devuelve {"words": [...], "segments": [...], "language": str} y guarda words.json."""
    try:
        words, segs, lang = _run(_get_model(), audio_path, on_progress)
    except Exception as exc:
        if _model_device == "cpu":
            raise
        # Típico en GPUs sin las DLL de cuBLAS/cuDNN: el fallo aparece al inferir.
        log.warning("Fallo en CUDA durante la transcripción (%s). Reintentando en CPU.", exc)
        words, segs, lang = _run(_get_model(force_cpu=True), audio_path, on_progress)

    (work_dir / "words.json").write_text(json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8")
    (work_dir / "segments.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"words": words, "segments": segs, "language": lang}
