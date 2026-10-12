"""Reencuadre inteligente 16:9 → 9:16 para el modo ligero (streamer permite clips).

Analiza fotogramas del clip con el detector de caras YuNet (OpenCV) y elige el encuadre:

- "follow":   cara grande (IRL, just chatting, cámara a pantalla completa). Recorte 9:16 que
              sigue la cara con movimiento suave y zona muerta, como un operador de cámara.
- "cam_game": cara pequeña en un borde (webcam superpuesta al juego). Webcam arriba y
              juego abajo, el formato habitual de los clips de streamers.
- "center":   sin cara fiable o video ya vertical. Recorte centrado.

La cara del streamer se identifica por persistencia: es la que aparece en el mismo sitio
durante la mayor parte del clip. Las caras de personajes del juego aparecen y se mueven,
así que no ganan.
"""
import logging
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config
from .ffmpeg_utils import get_ffmpeg

log = logging.getLogger("shorts.reframe")

MODEL_PATH = config.ASSETS_DIR / "models" / "face_detection_yunet_2023mar.onnx"

SAMPLE_FPS = 2          # fotogramas analizados por segundo
SAMPLE_WIDTH = 960      # ancho de análisis (detecta caras de webcam pequeñas sin ser lento)
MIN_SCORE = 0.75        # confianza mínima del detector
MIN_PRESENCE = 0.35     # la cara debe estar en al menos el 35 % de los fotogramas
CLUSTER_DIST = 0.10     # distancia (fracción del cuadro) para considerar la misma cara
SMALL_FACE = 0.10       # ancho de cara / ancho del cuadro por debajo del cual es "webcam"
EDGE_OFFSET = 0.18      # distancia al centro a partir de la cual la cara está "en un borde"
DEAD_ZONE = 0.035       # movimiento (fracción del ancho) que el encuadre ignora

CAM_H = 760             # alto de la webcam en cam_game (≈4:3 como las webcams; el juego ocupa el resto)


@dataclass
class Plan:
    layout: str                                   # follow | cam_game | center
    # follow: posiciones (t, centro x normalizado) del encuadre a lo largo del clip
    keyframes: list[tuple[float, float]] = field(default_factory=list)
    # cam_game: caja de la webcam (x, y, w, h) normalizada al cuadro
    cam_box: tuple[float, float, float, float] | None = None
    reason: str = ""


@lru_cache(maxsize=1)
def _detector():
    import cv2

    try:  # silenciar avisos internos de OpenCV en consola
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    if not MODEL_PATH.exists():
        log.warning("Falta el modelo de caras %s: se usará recorte centrado.", MODEL_PATH)
        return None
    return cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320), MIN_SCORE, 0.3, 50)


def _sample_frames(source: Path, start: float, duration: float, src_w: int, src_h: int):
    w = SAMPLE_WIDTH
    h = int(round(src_h * w / src_w / 2) * 2)
    cmd = [get_ffmpeg(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
           "-vf", f"fps={SAMPLE_FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    n = frames.size // (w * h * 3)
    return frames[: n * w * h * 3].reshape(n, h, w, 3), w, h


def _detect(frames: np.ndarray, w: int, h: int) -> list[list[tuple]]:
    """Por fotograma: lista de caras (cx, cy, fw, fh, score) normalizadas."""
    det = _detector()
    det.setInputSize((w, h))
    out = []
    for frame in frames:
        _, faces = det.detect(np.ascontiguousarray(frame))
        found = []
        for f in faces if faces is not None else []:
            x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])
            found.append(((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h, score))
        out.append(found)
    return out


def _persistent_face(per_frame: list[list[tuple]]):
    """Agrupa detecciones por posición y devuelve (cluster, presencia) del más persistente."""
    clusters: list[dict] = []
    for i, faces in enumerate(per_frame):
        for cx, cy, fw, fh, score in faces:
            best = min(clusters, key=lambda c: np.hypot(c["cx"] - cx, c["cy"] - cy), default=None)
            if best is not None and np.hypot(best["cx"] - cx, best["cy"] - cy) < CLUSTER_DIST:
                n = len(best["hits"])
                best["cx"], best["cy"] = (best["cx"] * n + cx) / (n + 1), (best["cy"] * n + cy) / (n + 1)
                best["hits"].append((i, cx, cy, fw, fh))
            else:
                clusters.append({"cx": cx, "cy": cy, "hits": [(i, cx, cy, fw, fh)]})
    if not clusters:
        return None, 0.0
    # Presencia = fotogramas distintos en los que aparece
    best = max(clusters, key=lambda c: len({hit[0] for hit in c["hits"]}))
    presence = len({hit[0] for hit in best["hits"]}) / max(1, len(per_frame))
    return best, presence


def _smooth_track(n: int, hits: list[tuple]) -> list[float]:
    """Centro x por fotograma: relleno de huecos, mediana móvil y zona muerta."""
    xs = np.full(n, np.nan)
    for i, cx, *_ in hits:
        xs[i] = cx if np.isnan(xs[i]) else (xs[i] + cx) / 2
    idx = np.arange(n)
    known = ~np.isnan(xs)
    xs = np.interp(idx, idx[known], xs[known])
    # Mediana móvil (ventana 5) para quitar saltos aislados del detector
    padded = np.pad(xs, 2, mode="edge")
    xs = np.array([np.median(padded[i:i + 5]) for i in range(n)])
    # Operador de cámara: solo se mueve si la cara sale de la zona muerta, y lo hace suave
    pos, out = xs[0], []
    for x in xs:
        if abs(x - pos) > DEAD_ZONE:
            pos += (x - pos) * 0.5
        out.append(float(pos))
    return out


def analyze(source: Path, start: float, duration: float, src_w: int, src_h: int) -> Plan:
    if src_w * 16 <= src_h * 9 + 1:
        return Plan("center", reason="el video ya es vertical")
    if _detector() is None:
        return Plan("center", reason="sin modelo de detección")
    try:
        frames, w, h = _sample_frames(source, start, duration, src_w, src_h)
    except subprocess.CalledProcessError as exc:
        log.warning("No se pudieron extraer fotogramas para el reencuadre: %s", exc)
        return Plan("center", reason="error leyendo fotogramas")
    if not len(frames):
        return Plan("center", reason="sin fotogramas")

    per_frame = _detect(frames, w, h)
    face, presence = _persistent_face(per_frame)
    if face is None or presence < MIN_PRESENCE:
        return Plan("center", reason=f"sin cara persistente (presencia {presence:.0%})")

    hits = face["hits"]
    fw = float(np.median([hit[3] for hit in hits]))
    fh = float(np.median([hit[4] for hit in hits]))
    cx, cy = float(np.median([hit[1] for hit in hits])), float(np.median([hit[2] for hit in hits]))
    off_center = abs(cx - 0.5) > EDGE_OFFSET or abs(cy - 0.5) > EDGE_OFFSET

    if fw < SMALL_FACE and off_center:
        # Caja de la webcam: cabeza y hombros alrededor de la cara, con la proporción del panel superior
        box_h = min(1.0, fh * 2.6)
        box_w = min(1.0, box_h * (config.OUT_W / CAM_H) * (src_h / src_w))
        bx = min(max(cx - box_w / 2, 0.0), 1.0 - box_w)
        by = min(max(cy + fh * 0.2 - box_h / 2, 0.0), 1.0 - box_h)
        return Plan("cam_game", cam_box=(bx, by, box_w, box_h),
                    reason=f"webcam en ({cx:.2f}, {cy:.2f}), cara {fw:.0%} del ancho, presencia {presence:.0%}")

    track = _smooth_track(len(per_frame), hits)
    keyframes = [(i / SAMPLE_FPS, x) for i, x in enumerate(track)]
    return Plan("follow", keyframes=keyframes,
                reason=f"cara {fw:.0%} del ancho, presencia {presence:.0%}")


def _crop_x_expr(keyframes: list[tuple[float, float]], scaled_w: int) -> str:
    """Expresión de ffmpeg para la x del recorte: interpolación lineal entre posiciones."""
    max_x = scaled_w - config.OUT_W
    pts = [(t, int(round(min(max(x * scaled_w - config.OUT_W / 2, 0), max_x)))) for t, x in keyframes]
    # Quitar puntos redundantes (posición repetida) para acortar la expresión
    simple = [pts[0]]
    for i in range(1, len(pts) - 1):
        if not (pts[i][1] == simple[-1][1] == pts[i + 1][1]):
            simple.append(pts[i])
    if len(pts) > 1:
        simple.append(pts[-1])
    if len({x for _, x in simple}) == 1:
        return str(simple[0][1])
    expr = str(simple[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(zip(simple, simple[1:]))):
        expr = f"if(lt(t,{t1:.2f}),{x0}+({x1 - x0})*(t-{t0:.2f})/{t1 - t0:.2f},{expr})"
    return expr


def build_filters(plan: Plan, src_w: int, src_h: int, label_in: str = "0:v", label_out: str = "vbase") -> list[str]:
    """Filtros de ffmpeg que producen el video 1080x1920 según el plan."""
    W, H = config.OUT_W, config.OUT_H
    if plan.layout == "follow":
        scaled_w = int(round(src_w * H / src_h / 2) * 2)
        x = _crop_x_expr(plan.keyframes, scaled_w)
        return [f"[{label_in}]scale={scaled_w}:{H},crop={W}:{H}:'{x}':0,setsar=1,fps=30[{label_out}]"]
    if plan.layout == "cam_game":
        bx, by, bw, bh = plan.cam_box
        game_h = H - CAM_H
        return [
            f"[{label_in}]split[rf_cam][rf_game]",
            f"[rf_cam]crop=iw*{bw:.4f}:ih*{bh:.4f}:iw*{bx:.4f}:ih*{by:.4f},"
            f"scale={W}:{CAM_H}:force_original_aspect_ratio=increase,crop={W}:{CAM_H},setsar=1,fps=30[rf_top]",
            f"[rf_game]scale={W}:{game_h}:force_original_aspect_ratio=increase,crop={W}:{game_h},setsar=1,fps=30[rf_bottom]",
            f"[rf_top][rf_bottom]vstack=inputs=2[{label_out}]",
        ]
    return [f"[{label_in}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps=30[{label_out}]"]
