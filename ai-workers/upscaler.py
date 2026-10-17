"""Mejora de calidad / aumento de resolución.

Uso:
  python upscaler.py --input in.jpg --output out.png --scale 2
Motores: realesrgan (Real-ESRGAN vía spandrel) | opencv (Lanczos + nitidez) | mock (copia) | auto.
"""
from __future__ import annotations

import argparse

import numpy as np

from common import ensure_model, log, read_image, run_main, torch_device, write_image

MODELS = {
    2: ("RealESRGAN_x2plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    4: ("RealESRGAN_x4plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
}
# Ninguna red social aprovecha más de 4K; limitar evita tiempos y archivos absurdos.
MAX_OUTPUT_SIDE = 4096
TILE = 256
TILE_PAD = 32  # margen amplio: evita cortes visibles entre teselas en zonas lisas


def target_size(w: int, h: int, scale: int) -> tuple[int, int]:
    factor = min(scale, MAX_OUTPUT_SIDE / max(w, h))
    return max(1, round(w * factor)), max(1, round(h * factor))


def enhance_opencv(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    up = cv2.resize(denoised, size, interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.2)
    return cv2.addWeighted(up, 1.5, blur, -0.5, 0)  # unsharp mask


def upscale_realesrgan(image: np.ndarray, scale: int) -> np.ndarray:
    import cv2
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    filename, url = MODELS[scale]
    model_path = ensure_model(filename, url)
    descriptor = ModelLoader().load_from_file(str(model_path))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise TypeError("El modelo cargado no es de super-resolución")
    device = torch_device()
    descriptor.to(device).eval()
    model_scale = descriptor.scale

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    _, _, h, w = tensor.shape
    output = torch.zeros((1, 3, h * model_scale, w * model_scale), dtype=torch.float32)

    # Procesado por teselas con solapamiento: memoria acotada en CPU y sin costuras visibles.
    tiles_y, tiles_x = (h + TILE - 1) // TILE, (w + TILE - 1) // TILE
    total = tiles_x * tiles_y
    log(f"Real-ESRGAN x{model_scale} en {device}: {w}x{h}, {total} teselas")
    with torch.inference_mode():
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x0, y0 = tx * TILE, ty * TILE
                x1, y1 = min(x0 + TILE, w), min(y0 + TILE, h)
                px0, py0 = max(x0 - TILE_PAD, 0), max(y0 - TILE_PAD, 0)
                px1, py1 = min(x1 + TILE_PAD, w), min(y1 + TILE_PAD, h)
                out_tile = descriptor(tensor[:, :, py0:py1, px0:px1]).cpu()
                ox0, oy0 = (x0 - px0) * model_scale, (y0 - py0) * model_scale
                ox1, oy1 = ox0 + (x1 - x0) * model_scale, oy0 + (y1 - y0) * model_scale
                output[:, :, y0 * model_scale : y1 * model_scale, x0 * model_scale : x1 * model_scale] = out_tile[:, :, oy0:oy1, ox0:ox1]
                done = ty * tiles_x + tx + 1
                # Node lee las líneas "PROGRESS n" para mostrar el avance en la interfaz.
                log(f"PROGRESS {done * 100 // total}")

    result = output[0].clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((result * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def texture_aware_blend(original: np.ndarray, sr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Real-ESRGAN en bordes y estructuras; ampliación clásica (Lanczos) en texturas al azar.

    Real-ESRGAN define muy bien contornos y líneas, pero sobre texturas aleatorias (asfalto,
    pasto, grano de fotos nocturnas) inventa patrones ("barro agrietado", "pintura"). La
    coherencia del tensor de estructura distingue ambos casos: un borde tiene una dirección
    dominante (coherencia alta); el grano no tiene ninguna (coherencia baja).
    """
    import cv2

    classic = cv2.resize(original, size, interpolation=cv2.INTER_LANCZOS4)
    if (sr.shape[1], sr.shape[0]) != size:
        sr = cv2.resize(sr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(classic, cv2.COLOR_BGR2GRAY).astype(np.float32)
    scale = size[0] / original.shape[1]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sigma = max(1.5, 2.0 * scale)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    diff = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2)
    coherence = diff / (jxx + jyy + 1e-3)  # 0 = sin dirección (grano), 1 = borde/línea
    weight = np.clip((coherence - 0.25) / 0.35, 0, 1)
    weight = cv2.GaussianBlur(weight, (0, 0), sigma)[..., None]
    return (sr.astype(np.float32) * weight + classic.astype(np.float32) * (1 - weight)).round().clip(0, 255).astype(np.uint8)


def main() -> dict:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--engine", default="auto", choices=["auto", "realesrgan", "opencv", "mock"])
    args = parser.parse_args()

    image = read_image(args.input)
    if args.engine == "mock":
        write_image(args.output, image)
        return {"output": args.output, "engine": "mock"}

    h, w = image.shape[:2]
    size = target_size(w, h, args.scale)
    if size[0] < w * 1.15:
        # La imagen ya es grande: solo se limpia y se da nitidez, sin aumentar resolución.
        log("La imagen ya tiene alta resolución; se aplica solo mejora de nitidez.")
        write_image(args.output, enhance_opencv(image, (w, h)))
        return {"output": args.output, "engine": "opencv-enhance", "width": w, "height": h}

    engine = args.engine
    if engine in ("auto", "realesrgan"):
        try:
            # El modelo x4 solo compensa si de verdad se necesita más de ~x3.
            factor = size[0] / w
            model_scale = 4 if factor >= 3 else 2
            # Se reduce la entrada a target/model_scale: el modelo produce justo el tamaño final y
            # no procesa píxeles que luego se descartarían (en CPU es la parte más lenta).
            src_w, src_h = round(size[0] / model_scale), round(size[1] / model_scale)
            source = cv2.resize(image, (src_w, src_h), interpolation=cv2.INTER_AREA) if src_w < w else image
            log(f"Objetivo {size[0]}x{size[1]} con modelo x{model_scale} sobre {source.shape[1]}x{source.shape[0]}")
            result, engine = upscale_realesrgan(source, model_scale), "realesrgan"
            result = texture_aware_blend(image, result, size)
        except Exception as exc:  # noqa: BLE001
            if args.engine == "realesrgan":
                raise
            log(f"Real-ESRGAN no disponible ({exc}); usando OpenCV.")
            result, engine = enhance_opencv(image, size), "opencv"
    else:
        result = enhance_opencv(image, size)

    if (result.shape[1], result.shape[0]) != size:
        result = cv2.resize(result, size, interpolation=cv2.INTER_LANCZOS4)
    write_image(args.output, result)
    return {"output": args.output, "engine": engine, "width": size[0], "height": size[1]}


if __name__ == "__main__":
    run_main(main)
