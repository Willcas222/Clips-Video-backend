"""Fase 2.1: descarga del video con yt-dlp y extracción del audio a WAV 16 kHz."""
import logging
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func

from . import config
from .ffmpeg_utils import get_ffmpeg, probe, run_ffmpeg
from .kick import KickVideoIE

log = logging.getLogger("shorts.downloader")


class DownloadError(Exception):
    """Error presentable al usuario (se traduce a HTTP / mensaje de tarea)."""


def _friendly_error(msg: str) -> str:
    low = msg.lower()
    if any(k in low for k in ("could not resolve host", "getaddrinfo failed", "name resolution",
                              "failed to resolve", "network is unreachable")):
        return "No hay conexión a internet (o el sitio no responde). Revisa tu conexión e inténtalo de nuevo."
    if "sign in to confirm your age" in low or "age-restricted" in low or "age restricted" in low:
        return "El video tiene restricción de edad y no se puede descargar sin iniciar sesión."
    if "private video" in low:
        return "El video es privado."
    if "unavailable" in low or "has been removed" in low or "not available" in low:
        return "El video no está disponible (eliminado, bloqueado o caído)."
    if "is not a valid url" in low or "unsupported url" in low:
        return "Ese enlace no es de un sitio compatible o no apunta a un video."
    if "http error 404" in low:
        return "No se encontró la página del enlace (error 404). Revisa que la URL sea correcta."
    if "drm" in low:
        return "El video está protegido con DRM y no se puede descargar."
    if "login" in low or "log in" in low or "cookies" in low or "authentication" in low:
        return "El sitio exige iniciar sesión para ver este video."
    if "not currently live" in low:
        return "Ese enlace es de un canal en directo que ahora no está emitiendo. Usa el enlace de un VOD o un clip."
    if "live event" in low or "is live" in low:
        return "No se pueden procesar transmisiones en directo."
    return f"No se pudo descargar el video: {msg.strip()[:300]}"


MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5
# Fallos pasajeros de YouTube/red que suelen resolverse reintentando
_TRANSIENT = ("http error 403", "http error 429", "http error 5", "timed out", "timeout",
              "connection reset", "connection aborted", "temporarily", "remote end closed",
              "incomplete read", "unable to download video data", "ffmpeg exited with code",
              "could not resolve host", "getaddrinfo failed", "connection timed out")


def _is_transient(msg: str) -> bool:
    low = msg.lower()
    permanent = ("private video", "sign in to confirm your age", "unavailable", "removed",
                 "unsupported url", "not a valid url")
    return not any(p in low for p in permanent) and any(t in low for t in _TRANSIENT)


def _js_runtimes() -> dict:
    # YouTube exige resolver retos JS; usamos node o deno si están instalados.
    runtimes = {}
    for name in ("deno", "node"):
        if shutil.which(name):
            runtimes[name] = {}
    return runtimes


def _base_opts() -> dict:
    opts = {
        "ffmpeg_location": get_ffmpeg(),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Reintentos internos de yt-dlp para cortes a mitad de descarga
        "retries": 5,
        "fragment_retries": 5,
    }
    runtimes = _js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes
        opts["remote_components"] = ["ejs:github"]
    return opts


def _new_ydl(opts: dict) -> yt_dlp.YoutubeDL:
    ydl = yt_dlp.YoutubeDL(opts)
    ydl.add_info_extractor(KickVideoIE())
    return ydl


def _extract_info(ydl: yt_dlp.YoutubeDL, url: str) -> dict:
    """Metadatos sin descargar. Los videos de Kick pasan primero por el extractor propio
    (formato nuevo); si este delega en otro extractor, se resuelve aquí para tener la duración."""
    ie_key = KickVideoIE.ie_key() if KickVideoIE.suitable(url) else None
    info = ydl.extract_info(url, download=False, ie_key=ie_key)
    if info.get("_type") in ("url", "url_transparent"):
        info = ydl.extract_info(info["url"], download=False, ie_key=info.get("ie_key"))
    return info


def _with_retries(action, cleanup=None):
    """Ejecuta action() reintentando los fallos pasajeros; traduce el resto a DownloadError."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return action()
        except DownloadError:
            raise
        except Exception as exc:
            msg = str(exc)
            if attempt < MAX_ATTEMPTS and _is_transient(msg):
                wait = RETRY_WAIT_SECONDS * attempt
                log.warning("Descarga fallida (intento %d/%d): %s. Reintentando en %d s.",
                            attempt, MAX_ATTEMPTS, msg.strip()[:150], wait)
                if cleanup:
                    cleanup()
                time.sleep(wait)
                continue
            raise DownloadError(_friendly_error(msg)) from exc


# Fragmentos de video (fase D). Primero video y audio separados por descarga directa
# (DASH/https): al pedir un fragmento, combinar un video HLS con un audio aparte deja el archivo
# sin pista de video. Los sitios que solo ofrecen HLS con audio y video juntos (Kick, Twitch)
# caen en "b".
FORMAT = "/".join([
    "bv*[ext=mp4][height<=1080][protocol=https]+ba[ext=m4a][protocol=https]",
    "bv*[height<=1080][protocol=https]+ba[protocol=https]",
    "b[height<=1080]",
    "bv*[height<=1080]+ba",
    "b",
])

# Audio del stream completo (fase A). Si el sitio no ofrece pista de solo audio (Kick), se usa
# la variante combinada más ligera ("w", p. ej. 160p a ~230 kbps: ~3 % de la de 1080p) y se
# conserva solo el audio.
AUDIO_FORMAT = "ba[ext=m4a]/ba/w"


def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _check_info(info: dict) -> None:
    if info.get("_type") == "playlist" or info.get("entries"):
        raise DownloadError("El enlace es una lista o un canal, no un video. Usa el enlace de un video concreto.")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise DownloadError("No se pueden procesar transmisiones en directo. Espera a que termine y usa el VOD.")
    duration = float(info.get("duration") or 0)
    if duration > config.MAX_VIDEO_HOURS * 3600:
        raise DownloadError(
            f"El video dura {fmt_time(duration)} y el máximo es {config.MAX_VIDEO_HOURS} h "
            "(ajustable con MAX_VIDEO_HOURS)."
        )


@dataclass
class AudioResult:
    audio_path: Path  # WAV mono 16 kHz del video completo
    title: str
    duration: float


def _progress_hook(on_progress):
    """Adapta el progreso de yt-dlp (bytes o fragmentos) a una fracción 0-1."""
    def hook(d):
        if d.get("status") != "downloading":
            return
        if d.get("fragment_count"):
            frac = (d.get("fragment_index") or 0) / d["fragment_count"]
        else:
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            frac = (d.get("downloaded_bytes") or 0) / total if total else 0
        on_progress(max(0.0, min(1.0, frac)))
    return hook


def _ffmpeg_audio(media_url: str, headers: dict, out_wav: Path, duration: float, on_progress=None) -> None:
    """Lee un stream (HLS) con ffmpeg y escribe directamente el WAV de audio.

    Para los VODs HLS es mucho más fiable que la descarga por fragmentos de yt-dlp (que en
    Windows falla con descargas paralelas) y no deja miles de fragmentos ni archivos
    intermedios en disco. Medido: ~50× tiempo real con la variante de 160p de Kick.
    """
    cmd = [get_ffmpeg(), "-v", "error", "-nostats", "-progress", "pipe:1", "-y"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += ["-i", media_url, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if on_progress and duration and line.startswith("out_time_us="):
            try:
                on_progress(min(1.0, int(line.split("=")[1]) / 1e6 / duration))
            except ValueError:
                pass
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {err.strip()[-500:]}")


def download_audio(url: str, work_dir: Path, on_progress=None) -> AudioResult:
    """Fase A: descarga solo el audio del video completo y lo deja como WAV 16 kHz."""
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "audio.wav"
    opts = {**_base_opts(), "format": AUDIO_FORMAT, "outtmpl": str(work_dir / "stream.%(ext)s")}
    if on_progress:
        opts["progress_hooks"] = [_progress_hook(on_progress)]

    def cleanup():
        for p in work_dir.glob("stream.*"):
            p.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

    def action():
        with _new_ydl(opts) as ydl:
            info = _extract_info(ydl, url)
            _check_info(info)
            selected = ydl.process_ie_result(info, download=False)  # solo elige el formato
            duration = float(info.get("duration") or 0)
            if str(selected.get("protocol", "")).startswith("m3u8") and selected.get("url"):
                _ffmpeg_audio(selected["url"], selected.get("http_headers") or {}, audio_path, duration, on_progress)
            else:
                ydl.process_ie_result(info, download=True)
                downloaded = sorted(p for p in work_dir.glob("stream.*") if p.suffix not in (".part", ".ytdl"))
                if not downloaded:
                    raise DownloadError("La descarga terminó pero no se encontró el archivo de audio.")
                run_ffmpeg(["-i", str(downloaded[0]), "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(audio_path)])
            return info

    info = _with_retries(action, cleanup)
    for p in work_dir.glob("stream.*"):  # el WAV es lo único que se necesita a partir de aquí
        p.unlink(missing_ok=True)
    with wave.open(str(audio_path), "rb") as wf:
        wav_seconds = wf.getnframes() / wf.getframerate()
    if wav_seconds < 1:
        raise DownloadError("No se pudo obtener el audio del video.")
    return AudioResult(audio_path=audio_path, title=info.get("title") or "video",
                       duration=float(info.get("duration") or wav_seconds))


def download_section(url: str, work_dir: Path, start: float, end: float, name: str) -> Path:
    """Fase D: descarga solo el fragmento [start, end] del video (con imagen).

    El corte cae en el fotograma clave más cercano, así que el inicio real puede diferir unos
    segundos del pedido; el llamador lo corrige sincronizando con el audio completo (sync.py).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        **_base_opts(),
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(work_dir / f"{name}.%(ext)s"),
        "download_ranges": download_range_func(None, [(max(0.0, start), end)]),
    }

    def cleanup():
        for p in work_dir.glob(f"{name}.*"):
            p.unlink(missing_ok=True)

    def action():
        with _new_ydl(opts) as ydl:
            ydl.process_ie_result(_extract_info(ydl, url), download=True)

    _with_retries(action, cleanup)
    video_path = work_dir / f"{name}.mp4"
    if not video_path.exists():
        candidates = sorted(p for p in work_dir.glob(f"{name}.*") if p.suffix not in (".part", ".ytdl"))
        if not candidates:
            raise DownloadError("No se encontró el fragmento de video descargado.")
        video_path = candidates[0]
    if probe(video_path)["width"] == 0:
        raise DownloadError("El fragmento descargado no contiene video (solo audio).")
    return video_path
