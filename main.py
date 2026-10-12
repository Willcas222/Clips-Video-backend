"""Punto de entrada: uvicorn main:app --reload --port 8000"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from core import config
from core.health import check_dependencies

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("shorts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    deps = check_dependencies()
    if deps["ffmpeg"]["ok"]:
        log.info("ffmpeg OK: %s", deps["ffmpeg"]["path"])
    else:
        log.error("=" * 70)
        log.error("ffmpeg NO DISPONIBLE: %s", deps["ffmpeg"].get("error"))
        log.error("=" * 70)
        raise RuntimeError("ffmpeg es obligatorio para el backend")
    if deps["ollama"]["ok"]:
        log.info("Ollama OK: %s (modelo %s)", config.OLLAMA_HOST, config.OLLAMA_MODEL)
    else:
        # No se detiene el servidor: Ollama puede arrancarse después.
        log.error("=" * 70)
        log.error("OLLAMA: %s", deps["ollama"]["error"])
        log.error("Sin Ollama los clips se elegirán automáticamente (sin IA).")
        log.error("=" * 70)
    hw = deps["hardware"]
    log.info(
        "Hardware: %s núcleos, %.1f GB RAM | Whisper '%s' en %s (%s, bloques de %d s) | video: %s",
        hw["cpu_cores"], hw["ram_gb"], config.WHISPER_MODEL, hw["whisper_device"],
        hw["whisper_compute_type"], hw["transcribe_chunk_seconds"], hw["video_encoder"],
    )
    yield


app = FastAPI(title="Generador de Shorts", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.mount("/static/output", StaticFiles(directory=config.OUTPUT_DIR), name="output")
