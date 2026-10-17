"""Comprueba qué dependencias de los workers están instaladas.

Usa importlib.metadata/find_spec en lugar de importar: importar torch tarda decenas de
segundos con la CPU ocupada y haría que el chequeo de salud expire.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import sys

from common import WORKERS_DIR, run_main

# módulo -> nombre de distribución en pip
PACKAGES = {
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "torch": "torch",
    "spandrel": "spandrel",
    "diffusers": "diffusers",
    "faster_whisper": "faster-whisper",
}
REQUIRED = {"numpy", "cv2"}


def version_of(module: str, dist: str) -> str | None:
    if importlib.util.find_spec(module) is None:
        return None
    for name in (dist, "opencv-python", module):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "ok"


def main() -> dict:
    found = {module: version_of(module, dist) for module, dist in PACKAGES.items()}
    missing = [p for p in REQUIRED if not found[p]]
    if missing:
        raise RuntimeError(f"Faltan paquetes obligatorios: {', '.join(missing)}")
    models = sorted(p.name for p in (WORKERS_DIR / "models").glob("*") if p.is_file())
    return {"python": sys.version.split()[0], "packages": found, "models": models}


if __name__ == "__main__":
    run_main(main)
