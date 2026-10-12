"""Fase 2.4: renderizado vertical 9:16 con split-screen, pitch shift y subtítulos palabra a palabra.

Se usa ffmpeg directamente (un único proceso por clip) en lugar de moviepy: es más rápido
y no acumula memoria entre clips.
"""
import json
import logging
import random
import re
from pathlib import Path

from . import config, hardware, reframe
from .ffmpeg_utils import FFmpegError, probe, run_ffmpeg

log = logging.getLogger("shorts.editor")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def _background_files() -> list[Path]:
    # ".raw" = descarga a medio recortar
    return [p for p in config.BACKGROUNDS_DIR.iterdir()
            if p.suffix.lower() in VIDEO_EXTS and not p.stem.endswith(".raw")]


def background_credit(bg: Path | None) -> dict | None:
    """Créditos del fondo según assets/backgrounds/credits.json, con el texto listo para la descripción.

    Para fondos sin entrada en credits.json (añadidos a mano) solo se devuelve el nombre del archivo.
    """
    if bg is None:
        return None
    credits_path = config.BACKGROUNDS_DIR / "credits.json"
    entry = {}
    if credits_path.exists():
        try:
            entries = json.loads(credits_path.read_text(encoding="utf-8"))
            entry = next((c for c in entries if c.get("file") == bg.name), {})
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("No se pudo leer %s: %s", credits_path, exc)
    if not entry.get("author"):
        return {"file": bg.name, "attribution": None}
    license_short = "CC BY" if "creative commons attribution" in (entry.get("license") or "").lower() else entry.get("license")
    text = f'Gameplay de fondo: "{entry["title"]}" por {entry["author"]} ({entry["url"]})'
    if license_short:
        text += f" · Licencia {license_short}"
    return {
        "file": bg.name,
        "title": entry.get("title"),
        "author": entry["author"],
        "author_url": entry.get("channel_url"),
        "url": entry.get("url"),
        "license": license_short,
        "attribution": text,
    }


def pick_backgrounds(n: int) -> list[Path | None]:
    """n fondos al azar sin repetir mientras haya suficientes (se repiten solo si n > disponibles)."""
    files = _background_files()
    if not files:
        return [None] * n
    picks: list[Path] = []
    while len(picks) < n:
        batch = random.sample(files, len(files))
        # Evitar que el mismo fondo quede seguido al empezar una nueva ronda
        if picks and len(batch) > 1 and batch[0] == picks[-1]:
            batch.append(batch.pop(0))
        picks.extend(batch)
    return picks[:n]


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean_word(w: str) -> str:
    w = re.sub(r"[{}\\]", "", w)
    return w.strip(" \"'“”«»").upper()


def build_ass(words: list[dict], clip_start: float, clip_end: float, ass_path: Path, center_y: int) -> int:
    """Escribe un .ass con un evento por palabra centrado en (ancho/2, center_y). Devuelve nº de palabras."""
    in_clip = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]
    cx, cy = config.OUT_W // 2, center_y
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {config.OUT_W}
PlayResY: {config.OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,{config.SUBTITLE_FONT},130,&H0000FFFF,&H0000FFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,9,5,5,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    duration = clip_end - clip_start
    for i, w in enumerate(in_clip):
        text = _clean_word(w["word"])
        if not text:
            continue
        start = max(0.0, w["start"] - clip_start)
        # La palabra permanece hasta que empieza la siguiente (máx. 1 s de silencio visible)
        nxt = in_clip[i + 1]["start"] - clip_start if i + 1 < len(in_clip) else w["end"] - clip_start
        end = min(duration, max(w["end"] - clip_start, min(nxt, w["end"] - clip_start + 1.0)))
        if end <= start:
            end = start + 0.15
        # Efecto "pop": entra al 80 % y crece al 100 % en 80 ms
        fx = rf"{{\pos({cx},{cy})\fscx80\fscy80\t(0,80,\fscx100\fscy100)}}"
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Word,,0,0,0,,{fx}{text}")
    ass_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _cover(label_in: str, label_out: str, w: int, h: int, extra: str = "") -> str:
    """Escala para cubrir w×h y recorta centrado (re-encuadre sin deformar)."""
    return (
        f"[{label_in}]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1{extra}[{label_out}]"
    )


# Subtítulos: en el split-screen van sobre la línea divisoria; a pantalla completa, en el
# tercio inferior para no tapar la cara del streamer (que suele estar centrada).
SUBS_Y_SPLIT = config.HALF_H
SUBS_Y_FULL = int(config.OUT_H * 0.72)


def render_clip(
    source: Path,
    words: list[dict],
    clip: dict,
    index: int,
    work_dir: Path,
    out_path: Path,
    background: Path | None | str = "auto",
    streamer_friendly: bool = False,
) -> str:
    """Renderiza un clip 1080x1920 con subtítulos palabra a palabra.

    streamer_friendly=False (por defecto, "escudo"): split-screen con video de fondo y pitch shift.
    streamer_friendly=True (ligero): recorte vertical a pantalla completa y audio original.
    background: solo en modo escudo. "auto" = uno al azar; None = el propio clip desenfocado.
    Devuelve el encuadre usado: "split" (escudo) o "follow" / "cam_game" / "center" (ligero).
    """
    start, end = float(clip["start_time"]), float(clip["end_time"])
    duration = end - start
    src_info = probe(source)
    has_audio = src_info["has_audio"]

    ass_path = work_dir / f"clip_{index}.ass"
    build_ass(words, start, end, ass_path, SUBS_Y_FULL if streamer_friendly else SUBS_Y_SPLIT)

    inputs = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source)]
    if streamer_friendly:
        plan = reframe.analyze(source, start, duration, src_info["width"], src_info["height"])
        log.info("Clip %d: encuadre %s (%s)", index, plan.layout, plan.reason)
        extra_inputs, video_filters, audio_filter = _light_branch(plan, src_info)
        layout = plan.layout
    else:
        extra_inputs, video_filters, audio_filter = _shield_branch(index, duration, background)
        layout = "split"

    filters = [*video_filters, _subtitles_filter("vbase", "vout", ass_path)]
    maps = ["-map", "[vout]"]
    if has_audio:
        filters.append(f"[0:a]{audio_filter}[aout]")
        maps += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "44100"]

    args = [*inputs, *extra_inputs, "-filter_complex", ";".join(filters), *maps]
    tail = ["-pix_fmt", "yuv420p", "-r", "30", "-t", f"{duration:.3f}", "-movflags", "+faststart", str(out_path)]
    _encode(args, tail, index)
    return layout


def _light_branch(plan: "reframe.Plan", src_info: dict) -> tuple[list[str], list[str], str]:
    """Rama B (streamer permite clips): reencuadre 9:16 guiado por la cara, audio intacto."""
    video = reframe.build_filters(plan, src_info["width"], src_info["height"])
    return [], video, "anull"


def _shield_branch(index: int, duration: float, background: Path | None | str) -> tuple[list[str], list[str], str]:
    """Rama A (riesgo de copyright): split-screen con fondo de gameplay + pitch shift del audio."""
    w, half = config.OUT_W, config.HALF_H
    bg = pick_backgrounds(1)[0] if background == "auto" else background
    inputs: list[str] = []
    if bg:
        bg_dur = probe(bg)["duration"]
        if bg_dur > duration + 1:
            offset = random.uniform(0, bg_dur - duration - 0.5)
            inputs = ["-ss", f"{offset:.3f}", "-t", f"{duration:.3f}", "-i", str(bg)]
        else:
            inputs = ["-stream_loop", "-1", "-t", f"{duration:.3f}", "-i", str(bg)]
        video = [_cover("0:v", "top", w, half, ",fps=30"), _cover("1:v", "bottom", w, half, ",fps=30")]
        log.info("Clip %d: fondo %s", index, bg.name)
    else:
        # Sin videos de fondo: versión ampliada y desenfocada del propio clip
        log.warning("No hay videos en %s; usando el propio clip desenfocado como mitad inferior.", config.BACKGROUNDS_DIR)
        video = [
            "[0:v]split[v0a][v0b]",
            _cover("v0a", "top", w, half, ",fps=30"),
            _cover("v0b", "bottom", w, half, ",fps=30,boxblur=20:2,eq=brightness=-0.08"),
        ]
    video.append("[top][bottom]vstack=inputs=2[vbase]")

    if abs(config.AUDIO_PITCH - 1.0) > 1e-3:
        audio = f"rubberband=pitch={config.AUDIO_PITCH}:formant=preserved"
    else:
        audio = "anull"
    return inputs, video, audio


def _subtitles_filter(label_in: str, label_out: str, ass_path: Path) -> str:
    # Rutas relativas a BASE_DIR (cwd de ffmpeg) para evitar el escape de "C:" en el filtro subtitles
    ass_rel = ass_path.relative_to(config.BASE_DIR).as_posix()
    fonts_rel = config.FONTS_DIR.relative_to(config.BASE_DIR).as_posix()
    return f"[{label_in}]subtitles=filename='{ass_rel}':fontsdir='{fonts_rel}'[{label_out}]"


def _encode(args: list[str], tail: list[str], index: int) -> None:
    encoder = hardware.video_encoder(config.VIDEO_ENCODER)
    try:
        run_ffmpeg([*args, *hardware.ENCODERS[encoder], *tail], cwd=config.BASE_DIR)
    except FFmpegError:
        if encoder == "libx264":
            raise
        # La GPU puede fallar puntualmente (sesiones agotadas, driver): reintento en CPU
        log.warning("El codificador %s falló en el clip %d; reintentando con libx264.", encoder, index)
        run_ffmpeg([*args, *hardware.ENCODERS["libx264"], *tail], cwd=config.BASE_DIR)
