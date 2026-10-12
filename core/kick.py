"""Extractor de yt-dlp para los videos de Kick con el formato nuevo.

Kick sirve los VODs recientes con identificadores UUID v7 (p. ej. 01a11846-1ad0-7513-...)
que la API que usa yt-dlp (``/api/v1/video/<id>``) no conoce y responde 404. La página del
video sí incluye los datos (título, duración, ``recording_url`` HLS) en el JSON que la web
incrusta, así que se leen de ahí. Si la página no los trae, se delega en el extractor
original de yt-dlp (``KickVOD``), con lo que los VODs antiguos siguen funcionando igual.
"""
import json
import re

from yt_dlp.extractor.common import InfoExtractor
from yt_dlp.utils import ExtractorError, float_or_none, parse_iso8601, traverse_obj, url_or_none


def _enclosing_object(text: str, idx: int) -> dict | None:
    """Objeto JSON más interno que contiene la posición idx (recorriendo llaves)."""
    depth, start = 0, None
    for i in range(idx, -1, -1):
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            if depth == 0:
                start = i
                break
            depth -= 1
    if start is None:
        return None
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:j + 1])
                except json.JSONDecodeError:
                    return None
    return None


class KickVideoIE(InfoExtractor):
    IE_NAME = "kick:video"
    IE_DESC = "Kick VODs (formato nuevo con datos incrustados en la página)"
    _VALID_URL = r"https?://(?:www\.)?kick\.com/(?P<channel>[\w-]+)/videos/(?P<id>[\da-f]{8}-(?:[\da-f]{4}-){3}[\da-f]{12})"

    def _real_extract(self, url):
        video_id = self._match_id(url)
        webpage = self._download_webpage(url, video_id, impersonate=True)
        # Los datos van dentro de cadenas JS escapadas (Next.js): se desescapan antes de buscar
        text = webpage.replace('\\"', '"').replace("\\/", "/").replace("\\u0026", "&")

        data = None
        for m in re.finditer(rf'"id"\s*:\s*"{re.escape(video_id)}"', text):
            obj = _enclosing_object(text, m.start())
            if obj and url_or_none(obj.get("recording_url")):
                data = obj
                break

        if not data:
            # Página sin datos incrustados: probar con el extractor original de yt-dlp
            return self.url_result(url, "KickVOD", video_id)

        if data.get("status") not in (None, "public"):
            raise ExtractorError(f"El video no es público (estado: {data.get('status')})", expected=True)

        formats = self._extract_m3u8_formats(data["recording_url"], video_id, "mp4", m3u8_id="hls")
        return {
            "id": video_id,
            "formats": formats,
            **traverse_obj(data, {
                "title": ("title", {str}),
                "duration": ("duration", {float_or_none}),
                "channel": ("channel", "slug", {str}),
                "channel_id": ("channel", "id", {lambda x: str(x) if x is not None else None}),
                "uploader": ("channel", "username", {str}),
                "timestamp": ("start_time", {parse_iso8601}),
                "thumbnail": ("thumbnail", "src", {url_or_none}),
                "categories": ("category", "name", {str}, {lambda x: [x]}),
                "view_count": ("viewer_count", {int}),
                "age_limit": ("is_mature", {bool}, {lambda x: 18 if x else 0}),
                "is_live": ("is_live", {bool}),
            }),
        }
