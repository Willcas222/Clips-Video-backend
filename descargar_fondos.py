"""Descarga los videos de fondo (gameplay) listados en assets/backgrounds/credits.json.

Los fondos pesan ~2,3 GB y no se suben al repositorio; este script los vuelve a descargar en un
equipo nuevo. Solo baja los primeros 12 minutos de cada uno (sin audio), que es lo que se usa.
Los ya presentes se omiten.

Uso (desde backend/):  .venv\\Scripts\\python descargar_fondos.py
"""
import json
import sys
from pathlib import Path

from core import config
from core.downloader import DownloadError, download_section
from core.ffmpeg_utils import run_ffmpeg

SECONDS = 720


def main() -> int:
    credits_path = config.BACKGROUNDS_DIR / "credits.json"
    if not credits_path.exists():
        print(f"No existe {credits_path}")
        return 1
    entries = json.loads(credits_path.read_text(encoding="utf-8"))
    failed = 0
    for entry in entries:
        target = config.BACKGROUNDS_DIR / entry["file"]
        if target.exists():
            print(f"✓ {target.name} (ya existe)")
            continue
        print(f"→ {target.name}: {entry['title']} — {entry['author']}")
        tmp_dir = config.TEMP_DIR / "fondos"
        try:
            section = download_section(entry["url"], tmp_dir, 0, SECONDS, target.stem)
            run_ffmpeg(["-i", str(section), "-an", "-c:v", "copy", "-movflags", "+faststart", str(target)])
            section.unlink(missing_ok=True)
            print(f"  listo ({target.stat().st_size / 2**20:.0f} MB)")
        except DownloadError as exc:
            failed += 1
            print(f"  ERROR: {exc}")
    print("Fondos listos." if not failed else f"{failed} fondo(s) no se pudieron descargar.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
