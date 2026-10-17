"""Diagnóstico y mejora de aspecto de una foto (luz, color, contraste, ruido).

Uso:
  python enhance.py --input in.jpg --output out.png [--analyze-only]

Regla: solo correcciones TONALES y de COLOR (curvas, balance de blancos, contraste local,
viveza, reducción de ruido). Ningún paso mueve ni redibuja píxeles, así que personas y
objetos nunca se deforman.
"""
from __future__ import annotations

import argparse

import numpy as np

from common import log, read_image, run_main, write_image

SKIN_HUE = (20.0, 75.0)  # ángulo de tono en Lab (grados) típico de piel


def to_lab(image: np.ndarray) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def from_lab(lab: np.ndarray) -> np.ndarray:
    import cv2

    out = lab.copy()
    out[..., 0] = np.clip(out[..., 0], 0, 100) * 255.0 / 100.0
    out[..., 1:] = np.clip(out[..., 1:] + 128.0, 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)


def analyze(image: np.ndarray) -> dict[str, float]:
    import cv2

    h, w = image.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    lab = to_lab(small)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Dominante de color: media de a/b en píxeles casi neutros de luminosidad media.
    neutral = (chroma < 18) & (L > 20) & (L < 90)
    cast_a = float(a[neutral].mean()) if neutral.sum() > 500 else 0.0
    cast_b = float(b[neutral].mean()) if neutral.sum() > 500 else 0.0

    # Ruido: estimador de Immerkær sobre el gris.
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    noise = float(np.abs(cv2.filter2D(gray, -1, kernel)).mean() * np.sqrt(np.pi / 2) / 6)

    # Brillos/reflejos de luz: zonas muy claras y poco saturadas (lámparas, vidrio, superficies pulidas).
    glare = (L > 92) & (chroma < 10)
    glare = cv2.morphologyEx(glare.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Zonas amplias sobreiluminadas (halo de lámparas, ventanas, flash): luminosidad local alta,
    # aunque no lleguen a blanco puro.
    base = cv2.GaussianBlur(L, (0, 0), max(L.shape) / 40)

    glare_sources = find_glare_sources(lab)

    return {
        "destellos": len(glare_sources),
        "destellos_con_color": sum(1 for s in glare_sources if s["veil"] is not None),
        "zonas_sobreiluminadas_pct": round(float((base > 75).mean() * 100), 1),
        "brillo_medio": round(float(L.mean()), 1),
        # Blanco quemado real (los tres canales al máximo); un color saturado (rojo puro) no cuenta.
        "luces_quemadas_pct": round(float((small.min(axis=2) >= 248).mean() * 100), 2),
        "sombras_empastadas_pct": round(float((L < 4).mean() * 100), 2),
        "p99_luminosidad": round(float(np.percentile(L, 99)), 1),
        "contraste": round(float(L.std()), 1),
        "saturacion_media": round(float(chroma.mean()), 1),
        "dominante_a": round(cast_a, 1),
        "dominante_b": round(cast_b, 1),
        "ruido": round(noise, 2),
        "brillos_pct": round(float(glare.mean() * 100), 2),
    }


def describe_issues(m: dict[str, float]) -> list[str]:
    issues = []
    if m["destellos"]:
        color = " con halo de color (típico de luces LED) que tiñe" if m["destellos_con_color"] else " con un halo que lava"
        issues.append(f"Destello de {'una lámpara/foco' if m['destellos'] == 1 else f'{int(m['destellos'])} lámparas/focos'}{color} la zona de alrededor.")
    if m["luces_quemadas_pct"] > 1 or m["p99_luminosidad"] > 97:
        issues.append(f"Sobreexposición: {m['luces_quemadas_pct']:.1f}% de la imagen con luces quemadas (blanco sin detalle).")
    if m["zonas_sobreiluminadas_pct"] > 6:
        issues.append(
            f"Exceso de luz en zonas amplias (halo de lámpara, flash o ventana): ~{m['zonas_sobreiluminadas_pct']:.0f}% "
            "de la foto se ve lavada y sin detalle."
        )
    if m["brillos_pct"] > 0.8:
        issues.append(f"Brillos o reflejos de luz intensos (lámparas, vidrio o superficies pulidas) en ~{m['brillos_pct']:.1f}% de la foto.")
    if m["brillo_medio"] < 38:
        issues.append(f"Foto oscura / subexpuesta (brillo medio {m['brillo_medio']:.0f}/100).")
    elif m["brillo_medio"] > 68:
        issues.append(f"Foto muy clara en general (brillo medio {m['brillo_medio']:.0f}/100).")
    if m["sombras_empastadas_pct"] > 5:
        issues.append(f"Sombras empastadas: {m['sombras_empastadas_pct']:.1f}% de zonas negras sin detalle.")
    cast = float(np.hypot(m["dominante_a"], m["dominante_b"]))
    if cast > 3:
        tone = []
        if m["dominante_b"] > 2:
            tone.append("amarilla/cálida")
        if m["dominante_b"] < -2:
            tone.append("azul/fría")
        if m["dominante_a"] > 2:
            tone.append("rojiza/magenta")
        if m["dominante_a"] < -2:
            tone.append("verdosa")
        issues.append(f"Dominante de color {' y '.join(tone) or 'leve'} (balance de blancos desajustado).")
    if m["contraste"] < 16:
        issues.append("Contraste bajo: la imagen se ve plana o lavada.")
    if m["saturacion_media"] < 8:
        issues.append("Colores apagados (poca saturación).")
    if m["ruido"] > 4:
        issues.append(f"Ruido/grano visible (nivel {m['ruido']:.1f}), típico de poca luz o zoom digital.")
    return issues


def skin_mask(lab: np.ndarray) -> np.ndarray:
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    return (hue > SKIN_HUE[0]) & (hue < SKIN_HUE[1]) & (chroma > 8) & (chroma < 50)


def find_glare_sources(lab: np.ndarray) -> list[dict]:
    """Focos de luz con destello: núcleo pequeño, neutro y mucho más brillante que su entorno,
    rodeado de un velo de color (típico de LED azulados) o de un halo de luz que se extiende.
    Se descartan piel, objetos blancos grandes (lavabos, paredes) y velos cálidos."""
    import cv2

    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    h, w = L.shape
    diag = float(np.hypot(h, w))
    local = L - cv2.GaussianBlur(L, (0, 0), diag / 30)
    cand = ((local > 9) & (L > 75)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    count, _, stats, centers = cv2.connectedComponentsWithStats(cand)
    chroma = np.hypot(A, B)
    skin = skin_mask(lab)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    sources = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < h * w * 0.0001 or area > h * w * 0.003:
            continue
        if area / (bw * bh) < 0.4 or max(bw, bh) / max(1, min(bw, bh)) > 3:
            continue  # tiras y bordes brillantes no son focos
        cx, cy = float(centers[i][0]), float(centers[i][1])
        r0 = float(np.sqrt(area / np.pi))
        dist = np.hypot(xx - cx, yy - cy)
        core = dist < r0
        if np.median(chroma[core]) > 8 or skin[core].mean() > 0.05 or L[core].mean() < 78:
            continue
        # Una lámpara es un emisor aislado: a su alrededor (1.5–4 radios) todo es bastante más oscuro
        # que el núcleo. Cielo entre hojas, baldosas al sol o un tenis blanco tienen alrededor más
        # zonas claras: no son focos, y "quitarles el halo" los mancharía de negro.
        around = (dist >= r0 * 1.5) & (dist < r0 * 4)
        if np.percentile(L[around], 90) > 68 or (L[around] > 75).mean() > 0.08:
            continue

        rmax = min(diag * 0.3, r0 * 10)
        edges = np.linspace(r0 * 0.9, rmax, 36)
        pl, pa, pb, rc = [], [], [], []
        for a0, a1 in zip(edges[:-1], edges[1:]):
            ring = (dist >= a0) & (dist < a1)
            if ring.sum() < 30:
                continue
            pl.append(np.percentile(L[ring], 30))
            pa.append(np.median(A[ring]))
            pb.append(np.median(B[ring]))
            rc.append((a0 + a1) / 2)
        if len(rc) < 10:
            continue
        pl, pa, pb, rc = map(np.array, (pl, pa, pb, rc))
        bg = (float(np.median(pl[-5:])), float(np.median(pa[-5:])), float(np.median(pb[-5:])))
        # Halo de luz monótono decreciente y que llega exactamente a 0 en el borde (sin escalón visible).
        ex_l = np.maximum.accumulate(np.clip(pl - bg[0], 0, None)[::-1])[::-1]
        ex_l = np.clip(ex_l - ex_l[-1], 0, None) * np.clip(1 - rc / rmax, 0, 1) ** 0.5
        veil = np.array([np.mean(pa[:6]) - bg[1], np.mean(pb[:6]) - bg[2]])
        strength = float(np.hypot(*veil))
        veil_hue = float(np.degrees(np.arctan2(veil[1], veil[0])))
        colored = strength > 8 and not (10 < veil_hue < 95)  # velos cálidos se parecen a piel/tungsteno
        bloom = ex_l[0] > 8 and float(np.interp(r0 * 2.5, rc, ex_l)) > 3
        if colored or bloom:
            # Intensidad del destello: velo de color + halo de luz (sirve para puntuar antes/después).
            # Lo que se ve feo: velos de color y halos de luz fuertes; un punto brillante pequeño
            # (más aún tras una viñeta) es normal y casi no cuenta.
            severity = (strength if colored else 0.0) + max(0.0, float(ex_l[0]) - 12) * 1.5
            sources.append({"x": cx, "y": cy, "r0": r0, "rmax": rmax, "bg": bg, "rc": rc, "ex_l": ex_l, "veil": veil if colored else None, "severity": severity})
    return sources


def remove_glare(lab: np.ndarray, sources: list[dict]) -> np.ndarray:
    """Resta el halo radial y el velo de color de cada foco. Los píxeles de piel no se tocan."""
    import cv2

    h, w = lab.shape[:2]
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    protect = cv2.dilate(skin_mask(lab).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    protect = 1 - cv2.GaussianBlur(protect, (0, 0), 4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for s in sources:
        dist = np.hypot(xx - s["x"], yy - s["y"])
        r0, rmax = s["r0"], s["rmax"]
        glow = np.interp(dist, s["rc"], s["ex_l"], left=s["ex_l"][0], right=0)
        outside_core = np.clip((dist - r0 * 0.9) / (r0 * 0.7), 0, 1)  # el foco en sí se conserva
        # Quitar un halo nunca deja un píxel más oscuro que el fondo real: evita manchas negras.
        L[:] = np.maximum(L - glow * outside_core * 0.85 * protect, np.minimum(L, s["bg"][0]))
        if s["veil"] is not None:
            _, ba, bb = s["bg"]
            u = s["veil"] / np.hypot(*s["veil"])
            proj = (A - ba) * u[0] + (B - bb) * u[1]  # desviación hacia el color del halo
            fall = np.clip(1 - (dist - rmax) / (rmax * 0.8), 0, 1)  # cubre también haces cercanos
            remove = np.clip(proj, 0, None) * fall * 0.85 * protect
            A -= remove * u[0]
            B -= remove * u[1]
            # Junto al foco se neutraliza el color por completo (evita un borde cian/verdoso).
            near = np.clip(1 - (dist - r0) / (r0 * 1.5), 0, 1) * protect
            A[:] = A * (1 - near * 0.8) + ba * near * 0.8
            B[:] = B * (1 - near * 0.8) + bb * near * 0.8
    lab[..., 0] = L
    return lab


def enhance(image: np.ndarray, m: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    import cv2

    adjustments: list[str] = []
    h, w = image.shape[:2]

    # 1) Ruido primero: el resto de pasos lo amplificaría.
    if m["ruido"] > 4:
        strength = float(np.clip(m["ruido"] * 0.8, 3, 7))
        image = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        adjustments.append("Reducción de ruido suave")

    lab = to_lab(image)

    # 2) Destellos de lámparas: se quitan antes que el resto para no "corregir" un halo que va a desaparecer.
    if m["destellos"]:
        glare = find_glare_sources(lab)
        if glare:
            lab = remove_glare(lab, glare)
            adjustments.append("Eliminación del destello de la lámpara (halo y velo de color)")

    L = lab[..., 0]
    L_orig = L.copy()

    # 2) Balance de blancos: desplaza a/b hacia el neutro (parcial, para no dejar la foto "fría").
    cast = float(np.hypot(m["dominante_a"], m["dominante_b"]))
    if cast > 3:
        lab[..., 1] -= m["dominante_a"] * 0.65
        lab[..., 2] -= m["dominante_b"] * 0.65
        adjustments.append("Balance de blancos (corrección de dominante de color)")

    # 3) Luces altas: compresión local de zonas amplias sobreexpuestas + hombro en la curva.
    blown = m["luces_quemadas_pct"] > 1 or m["p99_luminosidad"] > 97 or m["brillos_pct"] > 0.8
    bloom = m["zonas_sobreiluminadas_pct"] > 6
    overexposed = blown or bloom
    if overexposed:
        # Compresión LOCAL: baja la luz solo donde la zona entera está sobreiluminada; los tonos
        # medios (piel, ropa) quedan igual. Sigma grande = transición suave, sin halos nuevos.
        base = cv2.GaussianBlur(L, (0, 0), max(h, w) / 40)
        weight = np.clip((base - 65) / 20, 0, 1)
        weight = weight * weight * (3 - 2 * weight)  # smoothstep
        L = L - weight * np.clip(L - 45, 0, None) * 0.30
        if blown:
            knee = 80.0
            x = np.clip((L - knee) / (100 - knee), 0, 1)
            L = np.where(L > knee, knee + (100 - knee) * x**1.3, L)
        adjustments.append("Recuperación de zonas sobreexpuestas (halo de luz y brillos)")

    # 4) Exposición global con gamma acotada.
    mean = float(L.mean())
    if mean < 38:
        gamma = float(np.clip(np.log(0.45) / np.log(max(mean, 5) / 100), 0.65, 1.0))
        L = 100 * (np.clip(L, 0, 100) / 100) ** gamma
        adjustments.append("Aumento de exposición")
    elif mean > 68 and not overexposed:
        gamma = float(np.clip(np.log(0.58) / np.log(mean / 100), 1.0, 1.3))
        L = 100 * (np.clip(L, 0, 100) / 100) ** gamma
        adjustments.append("Reducción de exposición")

    # 5) Sombras: levanta solo los tonos más oscuros.
    if m["sombras_empastadas_pct"] > 5 or mean < 38:
        lift = 6.0 * np.clip(1 - L / 30, 0, 1) ** 2
        L = L + lift
        adjustments.append("Recuperación de detalle en sombras")

    # 6) Contraste local (CLAHE) mezclado parcialmente para un aspecto natural.
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(L8).astype(np.float32) / 2.55
    mix = 0.45 if m["contraste"] < 16 else 0.25
    L = L * (1 - mix) + clahe * mix
    adjustments.append("Contraste local y claridad" if m["contraste"] < 16 else "Ajuste suave de claridad")
    # Protección de luces: lo que ya era muy claro no se aclara más (evita quemarlo). La transición es
    # GRADUAL (70→90): un corte brusco dibujaba un anillo alrededor de farolas y lámparas.
    protect = np.clip((L_orig - 70) / 20, 0, 1)
    protect = protect * protect * (3 - 2 * protect)
    L = L * (1 - protect) + np.minimum(L, L_orig) * protect
    lab[..., 0] = L

    # 7) Viveza: realza colores apagados, protegiendo los tonos de piel.
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    skin = (hue > SKIN_HUE[0]) & (hue < SKIN_HUE[1]) & (chroma > 5) & (chroma < 45)
    base_boost = 1.18 if m["saturacion_media"] < 8 else 1.08
    boost = 1 + (base_boost - 1) * np.clip(1 - chroma / 60, 0, 1)  # más en colores poco saturados
    boost = np.where(skin, 1 + (boost - 1) * 0.25, boost)
    # Sin realce en luces muy altas (un canal llegaría a 255); transición gradual para no dejar bordes.
    fade = np.clip((lab[..., 0] - 78) / 14, 0, 1)
    boost = boost * (1 - fade) + fade
    lab[..., 1] = a * boost
    lab[..., 2] = b * boost
    adjustments.append("Viveza de color (con protección de piel)")

    return from_lab(lab), adjustments


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    image = read_image(args.input)
    before = analyze(image)
    issues = describe_issues(before)
    log(f"Diagnóstico: {before}")
    if args.analyze_only or not args.output:
        return {"issues": issues, "adjustments": [], "metrics": before}

    result, adjustments = enhance(image, before)
    write_image(args.output, result)
    after = analyze(result)
    log(f"Después: {after}")
    metrics = {**{f"antes_{k}": v for k, v in before.items()}, **{f"despues_{k}": v for k, v in after.items()}}
    return {"output": args.output, "issues": issues, "adjustments": adjustments, "metrics": metrics}


if __name__ == "__main__":
    run_main(main)
