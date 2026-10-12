"""Localización de ffmpeg y utilidades de ejecución/sondeo."""
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import av


class FFmpegError(RuntimeError):
    pass


BIN_DIR = Path(__file__).resolve().parent.parent / "bin"


@lru_cache(maxsize=1)
def get_ffmpeg() -> str:
    """ffmpeg del sistema si existe; si no, el binario incluido en imageio-ffmpeg.

    El binario de imageio se expone como bin/ffmpeg(.exe): yt-dlp identifica ffmpeg por el
    nombre del ejecutable, y sin eso no permite descargas parciales (tramos de VODs).
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # pragma: no cover
        raise FFmpegError(
            "No se encontró ffmpeg. Instálalo (winget install Gyan.FFmpeg) "
            "o reinstala el paquete imageio-ffmpeg."
        ) from exc
    target = BIN_DIR / ("ffmpeg.exe" if bundled.suffix == ".exe" else "ffmpeg")
    if not target.exists() or target.stat().st_size != bundled.stat().st_size:
        BIN_DIR.mkdir(exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            target.hardlink_to(bundled)  # sin copiar 80 MB si están en el mismo disco
        except OSError:
            shutil.copy2(bundled, target)
    # Algunas comprobaciones de yt-dlp buscan ffmpeg solo en el PATH (ignoran ffmpeg_location)
    os.environ["PATH"] = str(BIN_DIR) + os.pathsep + os.environ.get("PATH", "")
    return str(target)


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    cmd = [get_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg falló ({proc.returncode}): {proc.stderr.strip()[-2000:]}")


def probe(path: Path) -> dict:
    """Duración (s), ancho, alto y si tiene audio."""
    with av.open(str(path)) as container:
        duration = container.duration / av.time_base if container.duration else 0.0
        v = next((s for s in container.streams if s.type == "video"), None)
        has_audio = any(s.type == "audio" for s in container.streams)
        return {
            "duration": float(duration),
            "width": v.codec_context.width if v else 0,
            "height": v.codec_context.height if v else 0,
            "has_audio": has_audio,
        }
