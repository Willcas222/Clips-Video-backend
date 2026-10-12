"""Sincroniza un fragmento de video descargado con el audio completo del stream.

Al descargar un fragmento sin recodificar, el corte cae en el fotograma clave anterior al
pedido, así que el fragmento empieza unos segundos antes de lo solicitado y no se sabe
cuánto. Comparando su audio con el audio completo (ya descargado y transcrito) se obtiene el
segundo exacto en que empieza, y con él los subtítulos quedan perfectamente sincronizados.
"""
import logging
import subprocess
from pathlib import Path

import numpy as np

from .ffmpeg_utils import get_ffmpeg

log = logging.getLogger("shorts.sync")

RATE = 4000            # la voz se distingue de sobra a 4 kHz y la correlación es 16× más barata
SEARCH_SECONDS = 15    # margen de búsqueda alrededor del inicio esperado
PROBE_SECONDS = 20     # trozo del fragmento que se compara


def _decode(path: Path, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    cmd = [get_ffmpeg(), "-v", "error", "-ss", f"{max(0.0, start):.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def find_section_start(section: Path, full_audio: Path, expected_start: float) -> float:
    """Segundo del video original en que empieza `section`. Si el audio no permite
    sincronizar con fiabilidad (silencio, música repetitiva), devuelve expected_start."""
    try:
        clip = _decode(section, 0, PROBE_SECONDS)
        win_start = max(0.0, expected_start - SEARCH_SECONDS)
        window = _decode(full_audio, win_start, PROBE_SECONDS + 2 * SEARCH_SECONDS)
    except subprocess.CalledProcessError as exc:
        log.warning("No se pudo leer audio para sincronizar: %s", exc)
        return expected_start
    if len(clip) < RATE * 3 or len(window) <= len(clip) or clip.std() < 1:
        return expected_start

    clip = (clip - clip.mean()) / clip.std()
    # Correlación cruzada normalizada por FFT
    n = 1 << int(np.ceil(np.log2(len(window) + len(clip))))
    corr = np.fft.irfft(np.fft.rfft(window, n) * np.conj(np.fft.rfft(clip, n)), n)[: len(window) - len(clip) + 1]
    csum = np.concatenate(([0.0], np.cumsum(window)))
    csum2 = np.concatenate(([0.0], np.cumsum(window ** 2)))
    L = len(clip)
    mean = (csum[L:] - csum[:-L]) / L
    std = np.sqrt(np.maximum((csum2[L:] - csum2[:-L]) / L - mean ** 2, 1e-6))
    score = corr / (std * L)
    best = int(score.argmax())
    confidence = float(score[best])
    if confidence < 0.5:
        log.warning("Sincronización poco fiable (%.2f); se usa el inicio pedido.", confidence)
        return expected_start
    actual = win_start + best / RATE
    log.info("Fragmento sincronizado: pedido %.2f s, real %.2f s (confianza %.2f)", expected_start, actual, confidence)
    return actual
