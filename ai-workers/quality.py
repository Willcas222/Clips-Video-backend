"""Puntuación técnica estimada (0-100) de una foto, para comparar antes/después.

No es un juicio estético: mide exposición, color, contraste, nitidez del sujeto, ruido,
destellos y separación sujeto/fondo con métricas objetivas.
"""
from __future__ import annotations

import numpy as np


def technical_score(image: np.ndarray, subject: np.ndarray | None = None) -> dict:
    import cv2

    from enhance import find_glare_sources, to_lab

    h, w = image.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    if subject is not None:
        subject = cv2.resize(subject, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_LINEAR) > 127
    lab = to_lab(small)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

    def band(value: float, ideal_lo: float, ideal_hi: float, worst: float) -> float:
        if ideal_lo <= value <= ideal_hi:
            return 1.0
        dist = ideal_lo - value if value < ideal_lo else value - ideal_hi
        return float(max(0.0, 1 - dist / worst))

    burned = float((small.min(axis=2) >= 248).mean() * 100)
    exposure = band(float(L.mean()), 42, 62, 25) * (1 - min(1.0, burned / 6))
    neutral = (np.hypot(a, b) < 18) & (L > 20) & (L < 90)
    cast = float(np.hypot(a[neutral].mean(), b[neutral].mean())) if neutral.sum() > 500 else 0.0
    # Escenas blancas/neutras tienen poca saturación de forma natural: solo se penaliza si es casi gris.
    color = band(cast, 0, 2.5, 10) * band(float(np.hypot(a, b).mean()), 3.5, 35, 4)
    contrast = band(float(L.std()), 18, 30, 14)

    if subject is not None and subject.mean() > 0.02:
        sharp_subject = float(lap[subject].mean())
        sharp_bg = float(lap[~subject].mean()) if (~subject).any() else sharp_subject
        # En un retrato profesional el sujeto tiene ~8x más detalle que el fondo desenfocado.
        separation = band(sharp_subject / max(sharp_bg, 1e-3), 8, 999, 7)
    else:
        sharp_subject = float(lap.mean())
        separation = 0.5
    sharpness = band(sharp_subject, 20, 999, 15)

    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    noise_level = float(np.abs(cv2.filter2D(gray, -1, kernel)).mean() * np.sqrt(np.pi / 2) / 6)
    noise = band(noise_level, 0, 2.5, 5)
    severity = sum(s["severity"] for s in find_glare_sources(lab))
    glare = float(max(0.0, 1 - severity / 30))

    parts = {
        "exposicion": (exposure, 20),
        "color": (color, 15),
        "contraste": (contrast, 10),
        "nitidez": (sharpness, 20),
        "ruido": (noise, 10),
        "luces_limpias": (glare, 10),
        "separacion_fondo": (separation, 15),
    }
    total = sum(v * weight for v, weight in parts.values())
    return {"total": round(total), **{k: round(v * 100) for k, (v, _) in parts.items()}}
