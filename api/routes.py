"""Endpoints REST + WebSocket de progreso."""
import asyncio
from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import AfterValidator, BaseModel

from core import pipeline
from core.health import check_dependencies

router = APIRouter(prefix="/api")


def _validate_url(v: str) -> str:
    # Cualquier sitio soportado por yt-dlp (YouTube, Kick, Twitch, TikTok, Vimeo, X...).
    # Si el sitio no es compatible, el error llega desde la fase de descarga.
    v = v.strip()
    parsed = urlparse(v)
    if parsed.scheme not in ("http", "https") or "." not in parsed.netloc:
        raise ValueError("Introduce un enlace válido (debe empezar por http:// o https://)")
    return v


VideoUrl = Annotated[str, AfterValidator(_validate_url)]


class ProcessRequest(BaseModel):
    url: VideoUrl
    model: str | None = None
    # True = el streamer permite el uso de sus clips: se omite el procesado anti-copyright
    # (split-screen y pitch shift) y se renderiza a pantalla completa con el audio original.
    is_streamer_friendly: bool = False


@router.get("/health")
def health():
    return check_dependencies()


@router.post("/process")
def process(req: ProcessRequest, background: BackgroundTasks):
    task = pipeline.create_task(req.url, req.is_streamer_friendly)
    # BackgroundTasks ejecuta funciones síncronas en el threadpool: no bloquea el event loop
    background.add_task(pipeline.run, task.id, req.model)
    return {"task_id": task.id}


def _get(task_id: str) -> pipeline.Task:
    task = pipeline.get_task(task_id)
    if not task:
        raise HTTPException(404, "Tarea no encontrada")
    return task


@router.get("/status/{task_id}")
def status(task_id: str):
    return _get(task_id).public()


@router.get("/clips/{task_id}")
def clips(task_id: str):
    task = _get(task_id)
    if task.status == "error":
        raise HTTPException(422, task.error or "La tarea falló")
    if task.status != "completed":
        raise HTTPException(409, "La tarea todavía no ha terminado")
    return {"task_id": task.id, "title": task.title, "clips": task.clips}


@router.websocket("/ws/{task_id}")
async def ws_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    task = pipeline.get_task(task_id)
    if not task:
        await websocket.send_json({"status": "error", "message": "Tarea no encontrada"})
        await websocket.close()
        return
    last = -1
    try:
        while True:
            if task.version != last:
                last = task.version
                await websocket.send_json(task.public())
                if task.status in ("completed", "error"):
                    break
            await asyncio.sleep(0.3)
        await websocket.close()
    except WebSocketDisconnect:
        pass
