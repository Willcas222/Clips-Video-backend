"""Utilidades compartidas por los workers.

Protocolo con Node.js (PythonWorkerAdapter):
  * Los logs van a stderr.
  * Al terminar se imprime UNA línea JSON en stdout: {"ok": true, ...} o {"ok": false, "error": "..."}.
"""
from __future__ import annotations

import json
import sys
import traceback
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

WORKERS_DIR = Path(__file__).resolve().parent
MODELS_DIR = WORKERS_DIR / "models"

# torch.jit.load (usado por LaMa) emite FutureWarning en torch >= 2.9; no aporta nada al usuario.
warnings.filterwarnings("ignore", category=FutureWarning)


def log(*parts: Any) -> None:
    print(*parts, file=sys.stderr, flush=True)


def emit(ok: bool, **payload: Any) -> None:
    sys.stdout.write(json.dumps({"ok": ok, **payload}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_main(main: Callable[[], dict]) -> None:
    """Ejecuta `main` y traduce su resultado/excepción al protocolo JSON."""
    try:
        emit(True, **main())
    except Exception as exc:  # noqa: BLE001 - el error debe llegar a Node, sea cual sea
        traceback.print_exc(file=sys.stderr)
        emit(False, error=f"{type(exc).__name__}: {exc}")
        sys.exit(1)


def read_image(path: str, flags: int | None = None) -> np.ndarray:
    """cv2.imread no soporta rutas con caracteres no ASCII en Windows; se decodifica desde bytes."""
    import cv2

    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR if flags is None else flags)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return image


def write_image(path: str, image: np.ndarray) -> None:
    import cv2

    ext = Path(path).suffix or ".png"
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen como {ext}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(path)


def ensure_model(filename: str, url: str) -> Path:
    """Descarga un modelo a ai-workers/models la primera vez que se necesita."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / filename
    if target.exists() and target.stat().st_size > 0:
        return target

    log(f"Descargando modelo {filename} (solo la primera vez)...")
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -10
        while chunk := response.read(1024 * 1024):
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct >= last_pct + 10:
                    log(f"  {pct}% ({done // (1024 * 1024)} MB)")
                    last_pct = pct
    tmp.replace(target)
    return target


def torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
