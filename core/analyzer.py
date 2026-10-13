"""Fase C: selección de los momentos más virales con un LLM local (Ollama).

La transcripción se divide en bloques (por defecto 15 min con 1 min de solape) para no
saturar el contexto del modelo; cada bloque se analiza por separado y al final se unifican
los candidatos y se quedan los N (NUM_CLIPS o los pedidos) con mejor puntuación.
"""
import json
import logging
import math
import re

import ollama

from . import config

log = logging.getLogger("shorts.analyzer")

SYSTEM_PROMPT = (
    "Analiza esta porción de transcripción de un stream. Busca momentos de alta intensidad, "
    "chistes, declaraciones polémicas, o reacciones exageradas. {pick}, "
    "devuelve el timestamp de inicio, el timestamp de fin (duración entre {min_s} y {max_s} "
    "segundos), y un título. {empty}\n"
    "Responde ÚNICAMENTE con un JSON válido con esta estructura: "
    '{{"clips": [{{"title": "título gancho", "start_time": seg_inicio, "end_time": seg_fin, '
    '"reason": "por qué es viral", "score": 1-10}}]}}. '
    "Los tiempos son números en segundos tomados de las marcas [inicio-fin] de la transcripción "
    "(cuentan desde el comienzo de esta porción). Devuelve como máximo {per_chunk} momentos, sin "
    "solaparse. score indica lo viral que es (10 = muy viral). El título debe tener como máximo "
    "8 palabras. Escribe el título y la razón en el idioma de la transcripción."
)
# Muchos bloques: cada uno puede quedar vacío y al final se eligen los mejores entre todos.
PICK_ANY = ("Si encuentras un momento viral",
            "Si no hay nada interesante en este bloque, devuelve una lista vacía.")
# Pocos bloques (videos cortos): con "si encuentras" los modelos suelen devolver un solo
# momento y el resto se rellena a ciegas. Medido con qwen2.5:7b en una charla de 14 min:
# 1 clip con PICK_ANY, 3 clips con sentido con PICK_EXACT.
PICK_EXACT = ("Elige los {per_chunk} mejores momentos y, para cada uno",
              "Devuelve exactamente {per_chunk} momentos aunque alguno sea menos llamativo (pon un score bajo).")


def _schema(per_chunk: int) -> dict:
    """Esquema que Ollama impone al generar: sin él, el modelo a veces se inventa otra estructura
    (medido con qwen2.5:7b en un stream de 47 min: 2 de 4 bloques devolvieron JSON inservible)."""
    clip = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "start_time": {"type": "number"},
                       "end_time": {"type": "number"}, "reason": {"type": "string"}, "score": {"type": "number"}},
        "required": ["title", "start_time", "end_time", "reason", "score"],
    }
    return {"type": "object", "properties": {"clips": {"type": "array", "items": clip, "maxItems": per_chunk}},
            "required": ["clips"]}


class AnalyzerError(Exception):
    pass


def _format_transcript(segments: list[dict]) -> str:
    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    text = "\n".join(lines)
    if len(text) > config.MAX_TRANSCRIPT_CHARS:
        # Muestreo uniforme de líneas para cubrir todo el video
        step = len(text) / config.MAX_TRANSCRIPT_CHARS
        lines = [lines[int(i * step)] for i in range(int(len(lines) / step))]
        text = "\n".join(lines)
    return text


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise AnalyzerError("Ollama no devolvió un JSON válido.")
        return json.loads(match.group(0))


def _overlaps(start: float, end: float, accepted: list[dict], tolerance: float = 0.15) -> bool:
    return any(min(end, r["end_time"]) - max(start, r["start_time"]) > tolerance * (end - start) for r in accepted)


def _short_title(title) -> str:
    words = str(title or "Clip").strip().strip('"').split()
    short = " ".join(words[:10])
    return short if len(words) <= 10 else short.rstrip(",.;:") + "…"


def _num_ctx(prompt_chars: int) -> int:
    # ~3 caracteres por token en español + margen para la respuesta; múltiplo de 2048.
    # Un contexto ajustado reduce mucho la RAM de la caché KV en equipos modestos.
    needed = prompt_chars // 3 + 1024
    return min(32768, max(4096, -(-needed // 2048) * 2048))


def _normalize_clip(c: dict, offset: float, lo: float, hi: float) -> dict | None:
    """Pasa un clip del bloque a tiempo absoluto y fuerza su duración a [min, max] dentro de [lo, hi]."""
    min_s, max_s = config.CLIP_MIN_SECONDS, config.CLIP_MAX_SECONDS
    try:
        start, end = float(c["start_time"]) + offset, float(c["end_time"]) + offset
    except (KeyError, TypeError, ValueError):
        return None
    start = max(lo, min(start, hi))
    end = max(start, min(end, hi))
    if end - start < min_s:
        end = min(hi, start + min_s)
        start = max(lo, end - min_s)
    elif end - start > max_s:
        end = start + max_s
    if end - start < 1:
        return None
    try:
        score = max(1.0, min(10.0, float(c.get("score", 5))))
    except (TypeError, ValueError):
        score = 5.0
    return {
        "title": _short_title(c.get("title")),
        "start_time": round(start, 2),
        "end_time": round(end, 2),
        "reason": str(c.get("reason") or "").strip()[:400],
        "score": score,
    }


def _chunks(segments: list[dict], duration: float) -> list[tuple[float, float, list[dict]]]:
    """Ventanas de ANALYSIS_CHUNK_MINUTES con ANALYSIS_OVERLAP_MINUTES de solape."""
    size = config.ANALYSIS_CHUNK_MINUTES * 60
    step = max(60.0, size - config.ANALYSIS_OVERLAP_MINUTES * 60)
    out, w0 = [], 0.0
    while w0 < duration:
        w1 = min(duration, w0 + size)
        if duration - w1 < 3 * 60:  # resto final corto: se une a este bloque
            w1 = duration
        segs = [sg for sg in segments if w0 <= sg["start"] < w1]
        if segs:
            out.append((w0, w1, segs))
        if w1 >= duration:
            break
        w0 += step
    return out


class _OllamaUnavailable(Exception):
    pass


def _ask(client, model: str, system: str, user: str, per_chunk: int) -> list:
    """Clips crudos de un bloque. Los modelos pequeños a veces entran en bucle o devuelven JSON
    roto: un segundo intento con otra temperatura suele bastar."""
    for attempt, temperature in enumerate((0.4, 0.8), start=1):
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=_schema(per_chunk),
                options={
                    "temperature": temperature,
                    "num_ctx": _num_ctx(len(system) + len(user)),
                    "repeat_penalty": 1.15,
                    # ~120 tokens por clip en JSON; tope para cortar respuestas desbocadas
                    "num_predict": 200 * per_chunk + 200,
                },
                # Entre bloques se mantiene cargado; se libera al terminar el análisis
                keep_alive="2m",
            )
            clips = _extract_json(resp["message"]["content"]).get("clips", [])
            if isinstance(clips, list):
                return [c for c in clips if isinstance(c, dict)]
            log.warning("Intento %d: 'clips' no es una lista.", attempt)
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise AnalyzerError(f"El modelo '{model}' no está descargado. Ejecuta: ollama pull {model}") from exc
            log.warning("Intento %d: error de Ollama (%s).", attempt, exc.error)
        except (ConnectionError, OSError) as exc:
            raise _OllamaUnavailable(str(exc)) from exc
        except (AnalyzerError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            log.warning("Intento %d: respuesta de Ollama inválida (%s).", attempt, exc)
    return []


def _unload(client, model: str) -> None:
    """Libera la RAM del modelo: el render viene después."""
    try:
        client.generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass


def _fill(clips: list[dict], duration: float, reason: str, n: int) -> list[dict]:
    """Completa hasta n clips con tramos repartidos por el video que no se solapen con los elegidos."""
    length = min(config.CLIP_MAX_SECONDS, max(config.CLIP_MIN_SECONDS, duration / n))
    if duration <= length:
        return clips or [{"title": "Clip 1", "start_time": 0.0, "end_time": round(duration, 2), "reason": "Video completo"}]
    result = list(clips)
    # Candidatos en una rejilla fina para encontrar huecos entre los clips de la IA
    steps = max(n, int(duration // (length / 2)))
    grid = [i * (duration - length) / steps for i in range(steps + 1)]
    while len(result) < n:
        free = [s for s in grid if not _overlaps(s, s + length, result, tolerance=0)]
        if not free:
            break
        # El hueco más alejado de los clips ya elegidos: así los rellenos quedan repartidos
        # por todo el video en lugar de amontonarse al principio
        mids = [(c["start_time"] + c["end_time"]) / 2 for c in result]
        s = max(free, key=lambda s: min((abs(s + length / 2 - m) for m in mids), default=-s))
        result.append({"title": f"Momento {int(s // 60)}:{int(s % 60):02d}", "start_time": round(s, 2),
                       "end_time": round(s + length, 2), "reason": reason})
    return sorted(result, key=lambda c: c["start_time"])


def analyze(segments: list[dict], duration: float, model: str | None = None,
            on_progress=None, num_clips: int | None = None) -> list[dict]:
    """Devuelve los num_clips (por defecto NUM_CLIPS) mejores momentos (tiempos absolutos),
    ordenados por tiempo. on_progress(i, n) se llama al empezar cada bloque."""
    model = model or config.OLLAMA_MODEL
    n = num_clips or config.NUM_CLIPS
    if not segments:
        raise AnalyzerError("La transcripción está vacía: el video no parece tener voz.")

    # Videos cortos: no hay nada que elegir
    if duration <= config.CLIP_MIN_SECONDS:
        return [{"title": "Clip completo", "start_time": 0.0, "end_time": round(duration, 2),
                 "reason": "El video es más corto que la duración mínima de un clip"}]

    chunks = _chunks(segments, duration)
    # Momentos pedidos por bloque: los necesarios repartidos entre bloques más uno de margen
    # (mínimo 3). Pedir los n en cada bloque solo alarga la respuesta del modelo.
    per_chunk = min(n, max(3, math.ceil(n / len(chunks)) + 1))
    pick, empty = PICK_EXACT if len(chunks) <= n else PICK_ANY
    system = SYSTEM_PROMPT.format(min_s=config.CLIP_MIN_SECONDS, max_s=config.CLIP_MAX_SECONDS,
                                  per_chunk=per_chunk, pick=pick.format(per_chunk=per_chunk),
                                  empty=empty.format(per_chunk=per_chunk))
    client = ollama.Client(host=config.OLLAMA_HOST)
    candidates: list[dict] = []
    try:
        for i, (w0, w1, segs) in enumerate(chunks, start=1):
            if on_progress:
                on_progress(i, len(chunks))
            # Tiempos relativos al bloque: números pequeños que el modelo maneja mejor
            local = [{**sg, "start": sg["start"] - w0, "end": sg["end"] - w0} for sg in segs]
            user = (f"Porción {i} de {len(chunks)} ({w1 - w0:.0f} s).\n\n"
                    f"Transcripción:\n{_format_transcript(local)}")
            for raw in _ask(client, model, system, user, per_chunk):
                clip = _normalize_clip(raw, w0, w0, w1)
                if clip:
                    candidates.append(clip)
            log.info("Bloque %d/%d: %d candidatos acumulados.", i, len(chunks), len(candidates))
    except _OllamaUnavailable as exc:
        log.error("No se pudo conectar con Ollama en %s (%s). Usando selección automática.", config.OLLAMA_HOST, exc)
    finally:
        _unload(client, model)

    # Unificar: los mejores primero, descartando los que se solapan con uno mejor
    best: list[dict] = []
    for clip in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if len(best) >= n:
            break
        if not _overlaps(clip["start_time"], clip["end_time"], best):
            best.append(clip)

    if not best:
        log.warning("Ollama no devolvió clips utilizables; usando selección automática.")
        return _fill([], duration, "Selección automática (el análisis con IA no estuvo disponible)", n)
    if len(best) < n:
        log.info("Ollama devolvió %d de %d clips; completando automáticamente.", len(best), n)
        best = _fill(best, duration, "Selección automática para completar los clips", n)
    return sorted(best, key=lambda c: c["start_time"])
