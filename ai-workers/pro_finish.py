"""Acabado profesional: modo retrato, nitidez del sujeto, acabado de color y enderezado leve.

Uso:
  python pro_finish.py --input in.png --output out.png [--blur suave|medio|fuerte] [--no-blur] [--no-straighten]

Regla: la persona NUNCA se deforma ni se redibuja. Solo se desenfoca el FONDO, se ajusta
luz/color/nitidez, y como mucho se gira la foto entera unos grados (giro rígido).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import MODELS_DIR, ensure_model, log, read_image, run_main, write_image

os.environ.setdefault("U2NET_HOME", str(MODELS_DIR / "u2net"))

BLUR_RADIUS = {"suave": 0.006, "medio": 0.010, "fuerte": 0.016}  # radio del desenfoque / diagonal
MAX_STRAIGHTEN_DEG = 5.0


def segment_person(image: np.ndarray) -> np.ndarray | None:
    """Máscara 0-255 de la(s) persona(s) con U²-Net (human_seg); bordes refinados con la imagen."""
    import cv2
    from rembg import new_session, remove

    session = new_session("u2net_human_seg")
    mask = np.array(remove(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), session=session, only_mask=True))
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    coverage = float((mask > 127).mean())
    if coverage < 0.03 or coverage > 0.75:
        log(f"Sin persona clara para modo retrato (cobertura {coverage:.0%}).")
        return None
    # Se queda con las zonas grandes (descarta manchas sueltas del fondo).
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
    keep = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] > mask.size * 0.01:
            keep[labels == i] = 255
    soft = np.where(keep > 0, mask, 0).astype(np.float32) / 255.0
    return (guided_filter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255, soft, r=max(4, max(mask.shape) // 250), eps=1e-3) * 255).clip(0, 255).astype(np.uint8)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Filtro guiado (He et al.): ajusta el borde de la máscara al cabello y contornos reales."""
    import cv2

    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))  # noqa: E731
    mean_i, mean_p = box(guide), box(src)
    cov = box(guide * src) - mean_i * mean_p
    var = box(guide * guide) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return np.clip(box(a) * guide + box(b), 0, 1)


DEPTH_MODEL = ("depth_anything_v2_small.onnx", "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx")


def depth_defocus(image: np.ndarray, person: np.ndarray) -> np.ndarray | None:
    """Mapa 0-1 de cuánto desenfocar cada píxel según su distancia a la cámara comparada con la de
    la persona (Depth Anything V2). 0 = mismo plano que la persona, 1 = muy lejos (o muy cerca)."""
    import cv2

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(ensure_model(*DEPTH_MODEL)), providers=["CPUExecutionProvider"])
        h, w = image.shape[:2]
        scale = 518 / max(h, w)
        iw, ih = max(14, round(w * scale / 14) * 14), max(14, round(h * scale / 14) * 14)
        rgb = cv2.cvtColor(cv2.resize(image, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32)
        disparity = session.run(None, {session.get_inputs()[0].name: tensor})[0].reshape(ih, iw)
    except Exception as exc:  # noqa: BLE001 - sin modelo se usa la distancia en la foto
        log(f"Profundidad no disponible ({exc}); desenfoque por distancia en la foto.")
        return None

    disparity = cv2.resize(disparity.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(disparity, 2), np.percentile(disparity, 98)
    d = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0, 1)  # 1 = cerca de la cámara
    # Bordes del mapa ajustados a los contornos reales de la foto.
    d = guided_filter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255, d, r=max(2, max(h, w) // 120), eps=1e-3)
    body = person > 0.5
    if body.sum() < 50:
        return None
    focus = float(np.median(d[body]))
    # Margen de foco: lo cercano en profundidad a la persona queda nítido; más allá crece el desenfoque.
    # Solo se desenfoca lo que está DETRÁS de la persona: lo que está delante (un lavamanos, una
    # mesa) se ve deformado al desenfocarlo, sobre todo en selfies frente al espejo.
    defocus = np.clip((focus - d - 0.08) / max(focus - 0.08, 0.15), 0, 1)
    return (defocus * defocus * (3 - 2 * defocus)).astype(np.float32)


def portrait_blur(image: np.ndarray, mask: np.ndarray, strength: str) -> np.ndarray:
    """Desenfoque tipo lente SOLO en el fondo, sin que los colores de la persona se derramen en él."""
    import cv2

    h, w = image.shape[:2]
    alpha = mask.astype(np.float32) / 255.0
    # Trabajo a 1/4 de resolución: el fondo va desenfocado, así que no se pierde calidad.
    f = 4
    sw, sh = max(8, w // f), max(8, h // f)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    bg_weight = 1 - cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
    bg_weight = cv2.erode(bg_weight, np.ones((3, 3), np.uint8))  # aleja el borde de la persona
    radius = max(2, int(np.hypot(sw, sh) * BLUR_RADIUS[strength]))

    def lens_blur(r: int) -> np.ndarray:
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)).astype(np.float32)
        disk /= disk.sum()
        # Convolución normalizada: promedia solo píxeles de fondo (la persona no "mancha" el desenfoque).
        num = cv2.filter2D(small * bg_weight[..., None], -1, disk)
        den = cv2.filter2D(bg_weight, -1, disk)[..., None]
        return num / np.maximum(den, 1e-3)

    # Profundidad de campo: lo que está a la misma distancia que la persona (la máquina donde se
    # sienta, el suelo a sus pies) queda nítido y el desenfoque crece con la distancia REAL a la
    # cámara, como en un lente. Sin modelo de profundidad se usa la distancia en la foto.
    near, far = lens_blur(max(1, radius // 3)), lens_blur(radius)
    person_small = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
    defocus = depth_defocus(small.astype(np.uint8), person_small)
    if defocus is None:
        dist = cv2.distanceTransform(1 - (person_small > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        defocus = np.clip(dist / (np.hypot(sw, sh) * 0.18), 0, 1) ** 0.8
        sharp = np.zeros_like(defocus)
    else:
        sharp = 1 - np.clip(defocus / 0.25, 0, 1)  # zona de foco: se conserva la foto original
    ramp = defocus[..., None]
    blurred = near * (1 - ramp) + far * ramp
    blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_CUBIC)
    keep = cv2.resize(sharp, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    blurred = image.astype(np.float32) * keep + blurred * (1 - keep)
    # Fondo ligeramente más tenue y menos saturado: la persona resalta.
    hsv = cv2.cvtColor(np.clip(blurred, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.9
    hsv[..., 2] *= 0.96
    blurred = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    a = alpha[..., None]
    return (image.astype(np.float32) * a + blurred * (1 - a)).round().clip(0, 255).astype(np.uint8)


def sharpen_subject(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Nitidez (unsharp mask) en luminancia; más suave en la piel para no marcar poros."""
    import cv2

    from enhance import from_lab, skin_mask, to_lab

    lab = to_lab(image)
    L = lab[..., 0]
    sigma = max(0.8, max(image.shape[:2]) / 1600)
    detail = L - cv2.GaussianBlur(L, (0, 0), sigma)
    amount = np.full(L.shape, 0.55, np.float32)
    amount[skin_mask(lab)] = 0.2
    if mask is not None:
        amount *= 0.35 + 0.65 * (mask.astype(np.float32) / 255.0)  # el fondo casi no se enfoca
    lab[..., 0] = np.clip(L + detail * amount, 0, 100)
    return from_lab(lab)


def color_finish(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Curva tonal suave (contraste "de revista") y viñeta sutil que no toca a la persona."""
    from enhance import from_lab, to_lab

    lab = to_lab(image)
    x = np.clip(lab[..., 0] / 100, 0, 1)
    x = x + 0.10 * x * (1 - x) * (x - 0.5) * 4  # S-curve: extremos fijos, medios con más cuerpo
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if mask is not None and (mask > 127).any():
        ys, xs = np.nonzero(mask > 127)
        cy, cx = float(ys.mean()), float(xs.mean())
    else:
        cy, cx = h / 2, w / 2
    r = np.hypot((xx - cx) / w, (yy - cy) / h) / 0.75
    vignette = 1 - 0.12 * np.clip(r, 0, 1) ** 2.2
    if mask is not None:
        vignette = vignette + (1 - vignette) * (mask.astype(np.float32) / 255.0)
    lab[..., 0] = x * 100 * vignette
    return from_lab(lab)


def straighten(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, float | None, str]:
    """Endereza inclinaciones leves usando las líneas verticales del fondo. Giro rígido + recorte."""
    import cv2

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    if mask is not None:
        edges[cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 60] = 0  # solo líneas del fondo
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80, minLineLength=max(h, w) // 12, maxLineGap=8)
    if lines is None:
        return image, mask, None, "sin líneas de referencia"
    angles, weights = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # OpenCV 4: (N,1,4); OpenCV 5: (N,4)
        ang = np.degrees(np.arctan2(x2 - x1, y2 - y1))  # 0 = vertical
        ang = (ang + 90) % 180 - 90
        if abs(ang) <= 20:
            angles.append(ang)
            weights.append(np.hypot(x2 - x1, y2 - y1))
    if len(angles) < 3:
        return image, mask, None, "pocas verticales en el fondo"
    order = np.argsort(angles)
    cum = np.cumsum(np.array(weights)[order])
    tilt = float(np.array(angles)[order][np.searchsorted(cum, cum[-1] / 2)])
    spread = float(np.median(np.abs(np.array(angles) - tilt)))
    if abs(tilt) < 1.0:
        return image, mask, tilt, "ya está derecha"
    if abs(tilt) > MAX_STRAIGHTEN_DEG or spread > 4:
        return image, mask, tilt, f"inclinación de {abs(tilt):.0f}° (estilo selfie o perspectiva): se mantiene para no recortar"

    # Giro con recorte del mayor rectángulo de la misma proporción sin bordes vacíos.
    theta = np.radians(abs(tilt))
    scale = 1 / (np.cos(theta) + max(h / w, w / h) * np.sin(theta))
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), -tilt, 1 / scale)
    out = cv2.warpAffine(image, rot, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    out_mask = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_LINEAR) if mask is not None else None
    if out_mask is not None:
        ys = np.nonzero(out_mask > 127)[0]
        if mask is not None and ys.size and (ys.min() <= 2 or (mask > 127).sum() * 0.97 > (out_mask > 127).sum() / (1 / scale) ** 2):
            return image, mask, tilt, "enderezar recortaría a la persona: se mantiene"
    return out, out_mask, tilt, f"enderezada {abs(tilt):.1f}°"


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blur", default="suave", choices=list(BLUR_RADIUS))
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--no-straighten", action="store_true")
    parser.add_argument("--reference", help="Foto original para la puntuación \"antes\" (por defecto, la entrada)")
    args = parser.parse_args()

    from quality import technical_score

    image = read_image(args.input)
    adjustments: list[str] = []
    notes: list[str] = []

    log("Separando a la persona del fondo…")
    try:
        mask = segment_person(image)
    except Exception as exc:  # noqa: BLE001 - sin modelo se sigue sin modo retrato
        log(f"Segmentación no disponible: {exc}")
        mask = None
    if args.reference:
        reference = read_image(args.reference)
        import cv2

        ref_mask = cv2.resize(mask, (reference.shape[1], reference.shape[0])) if mask is not None else None
        score_before = technical_score(reference, ref_mask)
    else:
        score_before = technical_score(image, mask)

    if not args.no_straighten:
        image, mask, tilt, note = straighten(image, mask)
        if note.startswith("enderezada"):
            adjustments.append(f"Foto {note}")
        elif tilt is not None and abs(tilt) >= 1:
            notes.append(f"Inclinación: {note}.")

    if mask is not None and not args.no_blur:
        log(f"Modo retrato ({args.blur})…")
        image = portrait_blur(image, mask, args.blur)
        adjustments.append(f"Modo retrato: fondo desenfocado ({args.blur}), persona intacta")
    elif mask is None:
        notes.append("No se detectó una persona clara: se omitió el desenfoque de fondo.")

    image = sharpen_subject(image, mask)
    adjustments.append("Nitidez profesional en el sujeto (suave en la piel)")
    image = color_finish(image, mask)
    adjustments.append("Acabado de color: curva tonal y viñeta sutil")

    write_image(args.output, image)
    score_after = technical_score(image, mask)
    log(f"Puntuación técnica: {score_before['total']} -> {score_after['total']}")
    return {"output": args.output, "adjustments": adjustments, "notes": notes, "scoreBefore": score_before, "scoreAfter": score_after}


if __name__ == "__main__":
    run_main(main)
