# Generador de Shorts con IA — Backend

Enlace de video (YouTube, Kick, Twitch, TikTok, Vimeo, X… cualquier sitio que soporte yt-dlp) → descarga (yt-dlp) → transcripción palabra a palabra (faster-whisper) →
selección de momentos (Ollama) → render 9:16 con split-screen, pitch y subtítulos (ffmpeg).

## Requisitos
- Windows, Python 3.10+ y Git
- **Ollama** (https://ollama.com) con un modelo: `ollama pull qwen2.5:3b`
  (con 16 GB de RAM o GPU puedes usar `qwen2.5:7b`; ver `.env.example`)
- ffmpeg: no hace falta instalarlo; se usa el binario de `imageio-ffmpeg` (incluye libass y
  rubberband) salvo que haya uno en el PATH.
- Frontend: repositorio aparte, [Clips-Video-Front](https://github.com/Willcas222/Clips-Video-Front) (Node 18+).

## Instalación en un equipo nuevo
```bat
mkdir proyecto_shorts && cd proyecto_shorts
git clone https://github.com/Willcas222/Clips-Video-backend.git backend
git clone https://github.com/Willcas222/Clips-Video-Front.git frontend

cd backend
instalar.bat
cd ..\frontend
instalar.bat
```
`backend\instalar.bat` crea el entorno de Python, instala las dependencias (y las librerías CUDA si
hay GPU NVIDIA), crea `.env` a partir de `.env.example`, descarga los videos de fondo (~2,3 GB,
`descargar_fondos.py`) y el modelo de Ollama si ya está instalado.

## Uso
1. `backend\iniciar.bat` (API en http://127.0.0.1:8000)
2. `frontend\iniciar.bat` y abre http://localhost:5173
3. Revisa la línea `Hardware:` al arrancar el backend para ver qué detectó (GPU, RAM, codificador).

## Plataformas compatibles
Cualquier sitio soportado por [yt-dlp](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)
(más de 1.700): YouTube, Kick (VODs y clips), Twitch, TikTok, Vimeo, X/Twitter, Facebook, Instagram…
Algunos sitios con protección anti-bots (como Kick) necesitan `curl_cffi`, ya incluido en
`requirements.txt`. No se pueden procesar directos en curso ni videos con DRM o que exijan
iniciar sesión.

### Modo de edición: "Streamer permite clips"
Interruptor en el formulario (desmarcado por defecto):

| | Desmarcado (por defecto) | Marcado |
|---|---|---|
| Video | Pantalla dividida: clip arriba, gameplay abajo | Reencuadre 9:16 automático según la cara (ver abajo) |
| Audio | Tono ligeramente modificado (`AUDIO_PITCH`) | Original, sin alterar |
| Subtítulos | Palabra a palabra, en la línea divisoria | Palabra a palabra, en el tercio inferior |
| Render | Normal | ~2× más rápido |

Úsalo marcado solo cuando el streamer permita expresamente el uso de sus clips.

**Reencuadre automático (modo ligero).** Se analizan 2 fotogramas por segundo con el detector
de caras YuNet (`assets/models/`, funciona sin internet) y se elige:
- **Sigue la cara**: cara grande (IRL, just chatting). El recorte 9:16 la sigue con movimiento suave.
- **Webcam + juego**: cara pequeña en un borde (webcam sobre el juego). Webcam arriba, juego abajo.
- **Centrado**: sin cara fiable o video ya vertical.

La cara del streamer se reconoce por persistencia (mismo sitio la mayor parte del clip), así que
las caras de personajes del juego no la confunden. La tarjeta muestra qué encuadre se usó.

### Flujo automático (también para streams de horas)
No hay que elegir nada: se pega la URL y la app procesa el video completo.

1. **Audio del stream completo.** Solo se descarga el audio (en YouTube, la pista de audio; en
   Kick/Twitch, que no la tienen, se lee la variante de vídeo más ligera con ffmpeg y se guarda
   solo el audio). Un VOD de 2,3 h de Kick baja en ~2 min.
2. **Transcripción** completa con faster-whisper, por bloques (memoria fija).
3. **IA por bloques.** La transcripción se divide en bloques de 15 min con 1 min de solape
   (`ANALYSIS_CHUNK_MINUTES`, `ANALYSIS_OVERLAP_MINUTES`). Ollama analiza cada bloque, puntúa
   los momentos (1-10) y al final se quedan los `NUM_CLIPS` mejores sin solaparse.
4. **Solo los fragmentos de video necesarios** se descargan (con imagen, en 1080p). Cada
   fragmento se sincroniza con el audio completo para que los subtítulos queden exactos aunque
   el corte caiga en otro fotograma clave.

Tiempo orientativo en un portátil de 4 núcleos sin GPU: la transcripción tarda ~0,45× la duración
del video y la IA ~3-4 min por bloque; un stream de 4 h puede llevar unas 3 h. Con GPU NVIDIA
baja mucho. Límite: `MAX_VIDEO_HOURS` (12 h por defecto).

## Adaptación automática al hardware
Al arrancar, el backend detecta el equipo y lo muestra en consola y en `GET /api/health`:

| Qué | Cómo se decide |
|---|---|
| Whisper en GPU o CPU | GPU NVIDIA con CUDA → GPU (`float16`); si no, CPU (`int8`) con un hilo por núcleo físico |
| Tamaño de bloque de transcripción | Según la RAM: 5 min (<8 GB), 10 min (8 GB), 15 min (16 GB), 30 min (32 GB+). Memoria fija sea cual sea la duración del video |
| Codificador de video | NVIDIA NVENC → AMD AMF → Intel QSV → libx264 (CPU). Si la GPU falla en un clip, se reintenta en CPU |
| Ollama | Usa la GPU automáticamente si la hay; el contexto se ajusta al tamaño de la transcripción |

### Pasar a un equipo más potente
1. Clona los dos repositorios (ver **Instalación en un equipo nuevo**).
2. Ejecuta `instalar.bat` en `backend` y en `frontend`.
3. Opcional, en `.env`, para aprovechar el equipo:
   - GPU NVIDIA: `WHISPER_MODEL=large-v3` (transcripción más precisa y aun así rápida).
   - GPU de 8 GB+ o 16 GB+ de RAM: `OLLAMA_MODEL=qwen2.5:7b` (mejor selección de clips);
     descárgalo con `ollama pull qwen2.5:7b`.
4. Comprueba la línea `Hardware:` al arrancar el backend para ver qué detectó.

Para Whisper en GPU NVIDIA hace falta un driver compatible con CUDA 12. Si falta, el backend
avisa y sigue funcionando en CPU.

## Configuración
Todo se ajusta en `.env` (plantilla en `.env.example`) o con variables de entorno.

| Variable | Por defecto | Descripción |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Servidor de Ollama |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Modelo para el análisis |
| `MAX_TRANSCRIPT_CHARS` | `40000` | Texto máximo enviado a Ollama (los videos largos se muestrean) |
| `WHISPER_MODEL` | `small` | `base` es ~3× más rápido pero comete errores visibles en los subtítulos (≈10 % de palabras distintas en español); `medium`/`large-v3` para GPU |
| `WHISPER_BEAM_SIZE` | `1` | `1` = voraz (rápido); `5` = búsqueda en haz (~40 % más lento, misma calidad práctica) |
| `WHISPER_DEVICE` | `auto` | `auto`, `cuda` o `cpu` |
| `WHISPER_CPU_THREADS` | `0` | `0` = núcleos físicos detectados |
| `TRANSCRIBE_CHUNK_SECONDS` | `auto` | Bloque de transcripción en segundos (auto = según RAM) |
| `WHISPER_KEEP_LOADED` | `0` | `1` mantiene Whisper en RAM entre tareas |
| `VIDEO_ENCODER` | `auto` | `auto`, `h264_nvenc`, `h264_amf`, `h264_qsv`, `libx264` |
| `VIDEO_BITRATE` | `3500k` | Calidad/peso del short: 3500k ≈ 13 MB por 30 s |
| `NUM_CLIPS` | `3` | Clips por video |
| `MAX_VIDEO_HOURS` | `12` | Duración máxima aceptada |
| `ANALYSIS_CHUNK_MINUTES` | `15` | Tamaño de cada bloque que analiza la IA |
| `ANALYSIS_OVERLAP_MINUTES` | `1` | Solape entre bloques (para no cortar un momento) |
| `AUDIO_PITCH` | `1.03` | Pitch shift sin cambiar tempo (`1.0` = desactivado) |
| `CLEANUP_TEMP` | `1` | Borra video/audio descargados al terminar |

## API
- `POST /api/process` `{ "url": "...", "is_streamer_friendly": false }` (cualquier enlace http/https) → `{ "task_id" }`
- `WS /api/ws/{task_id}` → progreso en tiempo real
- `GET /api/status/{task_id}` → estado (fallback por polling)
- `GET /api/clips/{task_id}` → metadata de los clips
- `GET /static/output/{filename}` → MP4 generados
- `GET /api/health` → estado de ffmpeg, Ollama y hardware detectado

## Estructura
```
backend/
  main.py             FastAPI + validador de arranque
  api/routes.py       Endpoints y WebSocket
  core/downloader.py  Fase A/D - audio completo y fragmentos de video
  core/transcriber.py Fase 2.2 - transcripción
  core/analyzer.py    Fase C - análisis con Ollama por bloques
  core/editor.py      Fase 2.4 - renderizado
  core/pipeline.py    Orquestador de tareas
  core/sync.py        Sincroniza cada fragmento con el audio completo
  core/hardware.py    Detección de GPU, RAM, núcleos y codificador
frontend/src/
  components/         UrlInput, ProgressPanel, ClipCard
  services/api.js     HTTP + WebSocket (con fallback a polling)
```

## Aviso
Las capas de edición (split-screen, subtítulos, pitch) son un estilo de formato: no otorgan
derechos sobre el material original ni garantizan que un video no sea detectado o retirado.
Usa videos propios, con licencia o con permiso del autor.
