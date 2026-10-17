"""Borra objetos de una imagen (inpainting).

Uso:
  python inpainting.py --input in.jpg --output out.png --mask mask.png
  python inpainting.py --input in.jpg --output out.png --bbox x1,y1,x2,y2
Motores: lama (LaMa big-lama, TorchScript) | opencv (Telea) | mock (copia) | auto (lama -> opencv).
"""
from __future__ import annotations

import argparse

import numpy as np

from common import ensure_model, log, read_image, run_main, torch_device, write_image

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
# Lado máximo de cada región que se pasa a LaMa: más grande no mejora y en CPU es muy lento.
LAMA_MAX_SIDE = 1024
# Dilatación de la máscara relativa al lado mayor de la imagen: cubre bordes, halos y sombras
# que el trazo no alcanzó (si queda un borde del objeto, LaMa lo "contagia" al relleno).
DILATE_RATIO = 0.012
# Zonas pintadas más cercanas que esto (relativo al lado mayor) se procesan juntas.
MERGE_RATIO = 0.03
# Franja alrededor del trazo donde GrabCut puede ampliar la máscara hasta el borde real del objeto.
BAND_RATIO = 0.04


def build_mask(image: np.ndarray, mask_path: str | None, bbox: str | None, reflections: bool = True) -> tuple[np.ndarray, bool]:
    """Máscara final (0/255) y si se añadió algún reflejo."""
    import cv2

    h, w = image.shape[:2]
    if mask_path:
        raw = read_image(mask_path, cv2.IMREAD_UNCHANGED)
        if raw.ndim == 3 and raw.shape[2] == 4:
            # PNG con transparencia (pintado en el navegador): el trazo es lo opaco.
            gray = raw[:, :, 3]
        elif raw.ndim == 3:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            gray = raw
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (gray > 40).astype(np.uint8) * 255
        # Rellena huecos pequeños entre trazos del pincel.
        close = max(3, int(max(h, w) * 0.01)) | 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    else:
        x1, y1, x2, y2 = (int(float(v)) for v in bbox.split(","))
        mask = np.zeros((h, w), np.uint8)
        mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 255

    if not mask.any():
        raise ValueError("La máscara está vacía: no hay nada que borrar")
    if mask_path:
        mask = refine_mask(image, mask)
    reflection_found = False
    if reflections:
        with_reflections = add_reflections(image, mask)
        reflection_found = bool((with_reflections != mask).any())
        mask = with_reflections
    size = max(5, int(max(h, w) * DILATE_RATIO)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1), reflection_found


def refine_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ajusta el trazo del usuario a los bordes reales del objeto con GrabCut.

    El trazo se toma como "probable objeto"; una franja alrededor como "probable fondo" y el
    resto como fondo seguro. Así se recuperan los bordes que el pincel no alcanzó, sin
    expandirse más allá de la franja. El resultado siempre incluye el trazo original.
    """
    import cv2

    h, w = mask.shape
    band = max(8, int(max(h, w) * BAND_RATIO))
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1)))
    refined = mask.copy()
    for x1, y1, x2, y2 in mask_regions(mask):
        x1, y1 = max(0, x1 - band * 2), max(0, y1 - band * 2)
        x2, y2 = min(w, x2 + band * 2), min(h, y2 + band * 2)
        sub_mask, sub_near = mask[y1:y2, x1:x2], near[y1:y2, x1:x2]
        gc = np.full(sub_mask.shape, cv2.GC_BGD, np.uint8)
        gc[sub_near > 0] = cv2.GC_PR_BGD
        gc[sub_mask > 0] = cv2.GC_PR_FGD
        # Núcleo del trazo (lejos de su borde): objeto seguro.
        core = cv2.erode(sub_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band + 1, band + 1)))
        gc[core > 0] = cv2.GC_FGD
        try:
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(np.ascontiguousarray(image[y1:y2, x1:x2]), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error as exc:
            log(f"GrabCut omitido en una zona: {exc}")
            continue
        grown = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)) & (sub_near > 0)
        # Solo se aceptan partes conectadas al trazo (descarta islas sueltas del fondo).
        count, labels = cv2.connectedComponents((grown | (sub_mask > 0)).astype(np.uint8))
        keep = np.unique(labels[sub_mask > 0])
        grown = np.isin(labels, keep[keep > 0])
        refined[y1:y2, x1:x2] = np.where(grown | (sub_mask > 0), 255, 0).astype(np.uint8)
    added = (refined > 0).sum() - (mask > 0).sum()
    log(f"Máscara ajustada a los bordes: +{added * 100 / max(1, (mask > 0).sum()):.0f}% de área")
    return refined


def add_reflections(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Añade a la máscara el reflejo de cada objeto sobre superficies brillantes (debajo de él).

    Modelo de mezcla: un píxel reflejado ≈ superficie + α·(objeto − superficie), con 0 < α < 1.
    Se trabaja en Lab dando más peso al tono (a, b), que el reflejo conserva, que a la
    luminosidad, que depende del brillo de la superficie. Solo se aceptan zonas que "cuelgan"
    del borde inferior del objeto.
    """
    import cv2

    h, w = mask.shape
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    lab *= np.array([0.5, 2.0, 2.0], np.float32)
    result = mask.copy()

    for x1, y1, x2, y2 in mask_regions(mask):
        ow, oh = x2 - x1, y2 - y1
        zy1, zy2 = max(0, y2 - int(oh * 0.15)), min(h, y2 + int(oh * 0.9))
        if zy2 - y2 < 8:
            continue  # el objeto toca el borde inferior: no hay dónde reflejarse
        zone = lab[zy1:zy2, x1:x2]
        free = mask[zy1:zy2, x1:x2] == 0
        if free.sum() < 50:
            continue
        obj = np.median(lab[y1:y2, x1:x2][mask[y1:y2, x1:x2] > 0], axis=0)
        surface = np.median(zone[free], axis=0)
        d = obj - surface
        dn = float(np.dot(d, d))
        if dn < 12**2:
            continue  # objeto y superficie demasiado parecidos para distinguir un reflejo

        diff = zone - surface
        alpha = (diff @ d) / dn
        resid = np.linalg.norm(diff - alpha[..., None] * d, axis=-1)
        likely = ((alpha > 0.12) & (alpha < 1.1) & (resid < 0.35 * dn**0.5) & free).astype(np.uint8) * 255
        likely = cv2.morphologyEx(likely, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        likely = cv2.morphologyEx(likely, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(likely)
        contact = (y2 - zy1) + int(oh * 0.12)
        # Un reflejo empieza justo DEBAJO del borde inferior del objeto, columna a columna. Algo claro
        # que está al lado (p. ej. un lavamanos blanco junto a una toalla) no es su reflejo.
        reach = max(6, int(oh * 0.12))
        under = (mask[zy1:zy2, x1:x2] > 0).astype(np.uint8)
        below = under.copy()
        for _ in range(reach):
            under = np.vstack([np.zeros((1, under.shape[1]), np.uint8), under[:-1]])
            below |= under
        found = np.zeros_like(likely)
        for i in range(1, count):
            comp = labels == i
            if stats[i, cv2.CC_STAT_TOP] > contact or stats[i, cv2.CC_STAT_AREA] <= ow * oh * 0.01:
                continue
            if (comp & (below > 0)).sum() < comp.sum() * 0.05:
                continue
            found[comp] = 255
        if found.any():
            log(f"Reflejo detectado bajo la zona ({x1},{y1}): {(found > 0).sum() * 100 / (ow * oh):.0f}% del tamaño del objeto")
            result[zy1:zy2, x1:x2] |= found
    return result


def mask_regions(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cajas (x1, y1, x2, y2) de cada zona a borrar; zonas cercanas se agrupan."""
    import cv2

    h, w = mask.shape
    gap = max(3, int(max(h, w) * MERGE_RATIO)) | 1
    grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap)))
    count, _, stats, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8))
    boxes = []
    for i in range(1, count):
        x, y, bw, bh, _ = stats[i]
        sub = mask[y : y + bh, x : x + bw]
        ys, xs = np.nonzero(sub)
        if len(xs):
            boxes.append((x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1))
    return boxes


def inpaint_opencv(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import cv2

    radius = max(3, int(min(image.shape[:2]) * 0.01))
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def blend(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """La zona enmascarada se reemplaza al 100%; solo el borde exterior se funde suavemente."""
    import cv2

    hard = mask.astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(hard, (0, 0), max(1.5, max(mask.shape) * 0.004))
    alpha = np.maximum(hard, soft)[..., None]
    return (filled.astype(np.float32) * alpha + original.astype(np.float32) * (1 - alpha)).round().astype(np.uint8)


def run_lama(model, device: str, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ejecuta LaMa sobre una región (BGR uint8, máscara 0/255) y devuelve la región rellenada."""
    import cv2
    import torch

    ch, cw = mask.shape
    scale = min(1.0, LAMA_MAX_SIDE / max(ch, cw))
    if scale < 1.0:
        size = (max(8, int(cw * scale)), max(8, int(ch * scale)))
        work_img = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        work_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    else:
        work_img, work_mask = image, mask

    # LaMa requiere dimensiones múltiplo de 8.
    wh, ww = work_mask.shape
    ph, pw = (8 - wh % 8) % 8, (8 - ww % 8) % 8
    rgb = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
    rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    msk = np.pad(work_mask, ((0, ph), (0, pw)), mode="symmetric")

    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    msk_t = torch.from_numpy((msk > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model(img_t, msk_t)[0].permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)[:wh, :ww]
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    if scale < 1.0:
        # Solo se reescala el relleno; los píxeles no enmascarados conservan su nitidez original.
        out = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
    return out


def inpaint_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    model_path = ensure_model("big-lama.pt", LAMA_URL)
    device = torch_device()
    model = torch.jit.load(str(model_path), map_location=device).eval()

    h, w = mask.shape
    result = image.copy()
    regions = mask_regions(mask)
    for i, (mx1, my1, mx2, my2) in enumerate(regions, 1):
        # Cada zona con su propio recorte y contexto: más resolución que un único recorte gigante.
        side = max(mx2 - mx1, my2 - my1)
        ctx = max(int(side * 0.6), 96)
        x1, y1 = max(0, mx1 - ctx), max(0, my1 - ctx)
        x2, y2 = min(w, mx2 + ctx), min(h, my2 + ctx)
        crop_img = result[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        log(f"LaMa en {device}: zona {i}/{len(regions)}, región {x2 - x1}x{y2 - y1}")
        filled = run_lama(model, device, crop_img, crop_mask)
        result[y1:y2, x1:x2] = blend(crop_img, filled, crop_mask)
    return result


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--bbox", help="x1,y1,x2,y2 en píxeles")
    parser.add_argument("--engine", default="auto", choices=["auto", "lama", "opencv", "mock"])
    parser.add_argument("--save-mask", help="Guarda la máscara final usada (depuración)")
    parser.add_argument("--no-reflections", action="store_true", help="No buscar ni borrar reflejos del objeto")
    args = parser.parse_args()
    if not args.mask and not args.bbox:
        raise ValueError("Se requiere --mask o --bbox")

    image = read_image(args.input)
    if args.engine == "mock":
        write_image(args.output, image)
        return {"output": args.output, "engine": "mock"}

    mask, reflection_found = build_mask(image, args.mask, args.bbox, reflections=not args.no_reflections)
    if args.save_mask:
        write_image(args.save_mask, mask)
    log(f"Imagen {image.shape[1]}x{image.shape[0]}, área a borrar: {(mask > 0).mean() * 100:.1f}%")

    engine = args.engine
    if engine in ("auto", "lama"):
        try:
            result, engine = inpaint_lama(image, mask), "lama"
        except Exception as exc:  # noqa: BLE001
            if args.engine == "lama":
                raise
            log(f"LaMa no disponible ({exc}); usando OpenCV.")
            result, engine = blend(image, inpaint_opencv(image, mask), mask), "opencv"
    else:
        result = blend(image, inpaint_opencv(image, mask), mask)

    write_image(args.output, result)
    return {"output": args.output, "engine": engine, "reflectionRemoved": reflection_found}


if __name__ == "__main__":
    run_main(main)
