"""Orquestador del flujo "audio primero" y publicación del progreso de cada tarea.

A. Descarga solo el audio del video completo (aunque sea un stream de horas).
B. Transcribe el audio completo.
C. La IA analiza la transcripción por bloques y elige los mejores momentos.
D. Descarga solo los fragmentos de video de esos momentos, los sincroniza con el audio y los
   renderiza.
"""
import logging
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

from . import analyzer, config, downloader, editor, sync, transcriber

log = logging.getLogger("shorts.pipeline")


@dataclass
class Task:
    id: str
    url: str
    # True = el streamer permite clips: render ligero (pantalla completa, audio original)
    is_streamer_friendly: bool = False
    status: str = "queued"  # queued | running | completed | error
    stage: str = "queued"
    message: str = "En cola..."
    progress: int = 0
    clips: list[dict] = field(default_factory=list)
    error: str | None = None
    title: str | None = None
    version: int = 0  # se incrementa en cada cambio (lo usa el WebSocket)
    created_at: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("version", None)
        return d


_tasks: dict[str, Task] = {}
_tasks_lock = threading.Lock()
# Un solo pipeline a la vez: whisper + ffmpeg saturan la CPU de una máquina local
_run_lock = threading.Lock()


def create_task(url: str, is_streamer_friendly: bool = False) -> Task:
    task = Task(id=uuid.uuid4().hex[:12], url=url, is_streamer_friendly=is_streamer_friendly)
    with _tasks_lock:
        _tasks[task.id] = task
    return task


def get_task(task_id: str) -> Task | None:
    return _tasks.get(task_id)


def _update(task: Task, **kw) -> None:
    for k, v in kw.items():
        setattr(task, k, v)
    task.version += 1


# Segundos de margen al descargar cada fragmento: el corte cae en un fotograma clave y la
# sincronización con el audio necesita algo de holgura a cada lado.
SECTION_PAD = 3.0


def run(task_id: str, model: str | None = None) -> None:
    task = _tasks[task_id]
    work_dir = config.TEMP_DIR / task_id
    with _run_lock:
        try:
            # A. Audio del stream completo
            _update(task, status="running", stage="download",
                    message="Descargando audio del stream completo...", progress=2)
            audio = downloader.download_audio(
                task.url, work_dir,
                on_progress=lambda f: _update(task, progress=2 + int(13 * f),
                                              message=f"Descargando audio del stream completo... {int(f * 100)}%"),
            )
            _update(task, title=audio.title, progress=15)

            # B. Transcripción completa
            _update(task, stage="transcribe", message="Transcribiendo el audio...", progress=15)
            tr = transcriber.transcribe(
                audio.audio_path, work_dir,
                on_progress=lambda f: _update(task, progress=15 + int(35 * f),
                                              message=f"Transcribiendo el audio... {int(f * 100)}%"),
            )
            if not config.WHISPER_KEEP_LOADED:
                transcriber.unload()  # deja la RAM libre para Ollama

            # C. Análisis por bloques
            _update(task, stage="analyze", progress=50,
                    message="IA analizando el contenido buscando momentos virales...")
            clips = analyzer.analyze(
                tr["segments"], audio.duration, model=model,
                on_progress=lambda i, n: _update(
                    task, progress=50 + int(20 * (i - 1) / n),
                    message=f"IA analizando el contenido buscando momentos virales... (bloque {i} de {n})"),
            )

            # D. Fragmentos de video + render
            results = []
            friendly = task.is_streamer_friendly
            # Los fondos solo se usan en el modo escudo (split-screen)
            backgrounds = [None] * len(clips) if friendly else editor.pick_backgrounds(len(clips))
            for i, clip in enumerate(clips, start=1):
                _update(task, stage="render", progress=70 + int(29 * (i - 1) / len(clips)),
                        message=f"¡Momentos encontrados! Descargando y renderizando clip {i} de {len(clips)}...")
                local_clip, local_words, section = _fetch_section(task.url, work_dir, i, clip, tr["words"],
                                                                   audio.audio_path, audio.duration)
                filename = f"{task_id}_clip{i}.mp4"
                bg = backgrounds[i - 1]
                layout = editor.render_clip(section, local_words, local_clip, i, work_dir,
                                            config.OUTPUT_DIR / filename, background=bg, streamer_friendly=friendly)
                section.unlink(missing_ok=True)
                results.append({
                    **clip,  # start_time / end_time: posición en el video original
                    "index": i,
                    "duration": round(clip["end_time"] - clip["start_time"], 2),
                    "filename": filename,
                    "url": f"/static/output/{filename}",
                    "mode": "light" if friendly else "shield",
                    "layout": layout,  # split | follow | cam_game | center
                    "background": editor.background_credit(bg),
                })
                _update(task, clips=list(results))

            _update(task, status="completed", stage="done", message="¡Listo!", progress=100)
        except (downloader.DownloadError, analyzer.AnalyzerError) as exc:
            log.warning("Tarea %s: %s", task_id, exc)
            _update(task, status="error", stage="error", message=str(exc), error=str(exc))
        except Exception as exc:
            log.exception("Tarea %s falló", task_id)
            msg = f"Error inesperado: {exc}"
            _update(task, status="error", stage="error", message=msg[:500], error=msg[:2000])
        finally:
            if config.CLEANUP_TEMP and work_dir.exists():
                for p in work_dir.iterdir():
                    if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".wav", ".m4a", ".part", ".ts"):
                        p.unlink(missing_ok=True)


def _fetch_section(url, work_dir, index, clip, words, full_audio, duration):
    """Descarga el fragmento del clip (con margen), lo sincroniza con el audio completo y
    devuelve el clip y las palabras en la línea de tiempo del fragmento."""
    want_start = max(0.0, clip["start_time"] - SECTION_PAD)
    want_end = min(duration, clip["end_time"] + SECTION_PAD)
    section = downloader.download_section(url, work_dir, want_start, want_end, f"section{index}")
    real_start = sync.find_section_start(section, full_audio, want_start)
    local_clip = {**clip, "start_time": clip["start_time"] - real_start, "end_time": clip["end_time"] - real_start}
    local_words = [{**w, "start": w["start"] - real_start, "end": w["end"] - real_start}
                   for w in words if w["end"] > clip["start_time"] and w["start"] < clip["end_time"]]
    return local_clip, local_words, section


def cleanup_task_dir(task_id: str) -> None:
    shutil.rmtree(config.TEMP_DIR / task_id, ignore_errors=True)
