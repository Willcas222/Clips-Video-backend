import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { AppError } from '../../domain/errors/AppError';
import type { BoundingBox } from '../../domain/entities/Media';
import type { ContentStrategy } from '../../domain/entities/ContentStrategy';
import type { KeyFrame } from '../../domain/entities/ProcessedContent';
import type { FrameAnalysis, IAIAssistant, StrategyRequest } from '../../domain/ports/IAIAssistant';
import type { CatalogSong } from '../../domain/ports/IMusicCatalog';

/** Lo mínimo que el adaptador necesita para preparar imágenes antes de enviarlas al modelo de visión. */
export interface ImagePreparer {
  getImageDimensions(inputPath: string): Promise<{ width: number; height: number }>;
  resizeImage(inputPath: string, outputPath: string, width: number, height: number): Promise<void>;
}

export interface OllamaConfig {
  url: string;
  textModel: string;
  visionModel: string;
  timeoutMs: number;
  tempDir: string;
  /** Tiempo que Ollama mantiene el modelo en memoria (ej. '0', '5m'). */
  keepAlive: string;
}

interface GenerateParams {
  model: string;
  prompt: string;
  system?: string;
  images?: string[];
  format?: object;
  temperature?: number;
}

interface PreparedImage {
  base64: string;
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
}

const PLATFORM_LABEL: Record<StrategyRequest['platform'], string> = {
  tiktok: 'TikTok',
  reels: 'Instagram Reels',
  stories: 'Historias de Instagram/Facebook',
  post: 'publicación de feed de Instagram/Facebook',
};

// Qwen2.5-VL trabaja con parches de 14 px fusionados 2x2: dimensiones múltiplo de 28.
const VISION_PATCH = 28;
// Ollama escala toda imagen hasta al menos 1024 parches (calibrado: 1036x756 -> 1064x784).
const VISION_MIN_PIXELS = VISION_PATCH * VISION_PATCH * 1024;

/** Réplica del `smart_resize` de Ollama, siempre hacia el mínimo de píxeles (coste fijo por imagen). */
export function visionInputSize(width: number, height: number) {
  const beta = Math.sqrt(VISION_MIN_PIXELS / (width * height));
  return {
    width: Math.max(VISION_PATCH, Math.ceil((width * beta) / VISION_PATCH) * VISION_PATCH),
    height: Math.max(VISION_PATCH, Math.ceil((height * beta) / VISION_PATCH) * VISION_PATCH),
  };
}

const strategySchema = {
  type: 'object',
  properties: {
    copy: { type: 'string' },
    hashtags: { type: 'array', items: { type: 'string' } },
    music: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          artist: { type: 'string' },
          reason: { type: 'string' },
          segment: { type: 'string' },
        },
        required: ['title', 'artist', 'reason', 'segment'],
      },
    },
    onScreenText: { type: 'array', items: { type: 'string' } },
    tips: { type: 'array', items: { type: 'string' } },
  },
  required: ['copy', 'hashtags', 'music', 'onScreenText', 'tips'],
} as const;

/** Adaptador hacia Ollama local (`/api/generate`) para texto y visión. */
export class OllamaAdapter implements IAIAssistant {
  constructor(
    private readonly cfg: OllamaConfig,
    private readonly images: ImagePreparer,
  ) {}

  async describeImage(imagePath: string, language: string, context?: string): Promise<string> {
    const img = await this.prepareImage(imagePath);
    const text = await this.generate({
      model: this.cfg.visionModel,
      images: [img.base64],
      prompt:
        `Describe esta imagen en ${languageName(language)} para un creador de contenido, en 4-6 frases: ` +
        'sujeto principal y su pose/expresión, entorno, colores, emoción/ambiente y qué la haría atractiva en redes. ' +
        'Añade una frase sobre la ILUMINACIÓN: de dónde viene la luz y qué reflejos se ven (espejos, vidrios, superficies brillantes). ' +
        'No juzgues la calidad técnica (exposición, ruido): eso se mide aparte. Si la persona sostiene un móvil, dilo así. ' +
        'Sé concreto y no inventes texto que no veas.' +
        (context ? ` Contexto que da el creador sobre la foto: "${context}"; úsalo para interpretar la escena.` : ''),
      temperature: 0.3,
    });
    return text.trim();
  }

  async locateObject(imagePath: string, objectDescription: string): Promise<BoundingBox | null> {
    const img = await this.prepareImage(imagePath);
    const raw = await this.generate({
      model: this.cfg.visionModel,
      images: [img.base64],
      prompt:
        `Locate "${objectDescription}" in the image and output its bbox coordinates in JSON format ` +
        '{"found": true, "bbox_2d": [x1, y1, x2, y2], "label": "..."}. ' +
        'If the object is not present, output {"found": false, "bbox_2d": [0,0,0,0], "label": ""}.',
      format: {
        type: 'object',
        properties: {
          found: { type: 'boolean' },
          bbox_2d: { type: 'array', items: { type: 'number' }, minItems: 4, maxItems: 4 },
          label: { type: 'string' },
        },
        required: ['found', 'bbox_2d'],
      },
      temperature: 0,
    });

    const parsed = safeJson<{ found?: boolean; bbox_2d?: number[] }>(raw);
    const box = parsed?.bbox_2d;
    if (!parsed?.found || !box || box.length !== 4) return null;

    // Coordenadas en el espacio de la imagen redimensionada -> imagen original.
    const sx = img.originalWidth / img.width;
    const sy = img.originalHeight / img.height;
    let [x1, y1, x2, y2] = box.map(Number);
    if ([x1, y1, x2, y2].some((n) => !Number.isFinite(n))) return null;
    if (x2 < x1) [x1, x2] = [x2, x1];
    if (y2 < y1) [y1, y2] = [y2, y1];

    // Margen del 8% para que el inpainting cubra bordes y sombras del objeto.
    const padX = (x2 - x1) * 0.08;
    const padY = (y2 - y1) * 0.08;
    const result: BoundingBox = {
      x1: clamp((x1 - padX) * sx, 0, img.originalWidth),
      y1: clamp((y1 - padY) * sy, 0, img.originalHeight),
      x2: clamp((x2 + padX) * sx, 0, img.originalWidth),
      y2: clamp((y2 + padY) * sy, 0, img.originalHeight),
    };
    console.log(`[ollama] "${objectDescription}" localizado en`, box, `(${img.width}x${img.height}) ->`, result);
    const area = (result.x2 - result.x1) * (result.y2 - result.y1);
    // Una caja diminuta o casi del tamaño de la imagen indica una detección inútil.
    if (area < 16 || area > img.originalWidth * img.originalHeight * 0.9) return null;
    return result;
  }

  async analyzeFrameSheet({ sheetPath, frames, language, context }: { sheetPath: string; frames: KeyFrame[]; language: string; context?: string }): Promise<FrameAnalysis[]> {
    // Una sola imagen con todos los fotogramas: Ollama cobra ~1000 tokens por imagen sin importar
    // su tamaño, así que N llamadas separadas costarían N veces más en CPU.
    const img = await this.prepareImage(sheetPath);
    const times = frames.map((f, i) => `${i + 1}=${formatTime(f.timestamp)}`).join(', ');
    const raw = await this.generate({
      model: this.cfg.visionModel,
      images: [img.base64],
      prompt:
        `Esta imagen es una cuadrícula con ${frames.length} fotogramas numerados (número amarillo arriba a la izquierda), ` +
        `en orden cronológico de izquierda a derecha y de arriba abajo, tomados de un video en los tiempos: ${times}. ` +
        'Eres editor de videos cortos virales. Para CADA fotograma devuelve "index" (su número), ' +
        '"score" de 0 a 10 (acción, emoción, rostros expresivos, nitidez, composición atractiva) ' +
        `y "description": una frase breve en ${languageName(language)} de lo que ocurre. ` +
        (context ? `Contexto del video según su creador: "${context}". Da más puntuación a los momentos que mejor lo representen. ` : '') +
        'Responde en JSON.',
      format: {
        type: 'object',
        properties: {
          frames: {
            type: 'array',
            items: {
              type: 'object',
              properties: { index: { type: 'integer' }, score: { type: 'number' }, description: { type: 'string' } },
              required: ['index', 'score', 'description'],
            },
          },
        },
        required: ['frames'],
      },
      temperature: 0.2,
    });

    const parsed = safeJson<{ frames?: Array<{ index?: number; score?: number; description?: string }> }>(raw);
    const byIndex = new Map((parsed?.frames ?? []).map((f) => [Number(f.index), f]));
    // Si el modelo omite fotogramas se les da puntuación neutra en lugar de fallar.
    return frames.map((frame, i) => {
      const entry = byIndex.get(i + 1);
      return {
        timestamp: frame.timestamp,
        score: clamp(Number(entry?.score ?? 5), 0, 10),
        description: String(entry?.description ?? '').trim(),
      };
    });
  }

  async generateStrategy(req: StrategyRequest): Promise<ContentStrategy> {
    const platform = PLATFORM_LABEL[req.platform];
    const lines = [
      `Plataforma: ${platform}.`,
      `Tipo de contenido: ${req.mediaKind === 'image' ? 'imagen' : 'video corto'}.`,
      req.durationSeconds ? `Duración del video: ${Math.round(req.durationSeconds)} segundos.` : '',
      req.tone ? `Tono deseado: ${req.tone}.` : 'Tono: cercano, auténtico y con gancho.',
      req.context
        ? `CONTEXTO DADO POR EL CREADOR (máxima prioridad; el copy, los hashtags y la música deben reflejarlo): "${req.context}"`
        : '',
      `Descripción visual:\n${req.visualDescription}`,
      req.transcript?.text ? `Lo que se dice en el audio:\n"${req.transcript.text.slice(0, 2000)}"` : '',
    ].filter(Boolean);

    const candidates = req.musicCandidates ?? [];
    const styleRule = req.musicStyle
      ? `El creador quiere música de tipo "${req.musicStyle}".`
      : 'Elige el género que mejor encaje con el contexto y el ambiente (puedes mezclar géneros).';
    // Con catálogo, la IA solo elige ids: título y artista vienen del catálogo y nunca se inventan.
    const musicInstruction = candidates.length
      ? [
          `- "music": elige las 3 canciones que MEJOR encajen con el contexto y el ambiente, SOLO de esta lista (usa su "id"). ${styleRule}`,
          '  Son canciones REALES y populares AHORA. "fuente": "Top <país> #N" = de lo MÁS ESCUCHADO esta semana en el país ' +
            '(N = puesto; prioriza estas); "Tendencia oficial TikTok #N" = ranking oficial de TikTok; "Tendencia · <estilo>" = ' +
            'estilo pedido; "Viral en TikTok" = virales recientes. Elige las que mejor encajen con el TEMA y el ambiente del ' +
            'contenido (por género y por lo que sabes del artista: fiesta → urbano/reggaetón; romántico → baladas/pop; etc.).',
          // Orden aleatorio: el modelo tiende a elegir las primeras de la lista; el puesto sigue en "fuente".
          ...shuffle(candidates).map(
            (s) =>
              `  · id=${s.id} | "${s.title}" — ${s.artist} | fuente: ${s.source}` +
              (s.genre ? ` | ${s.genre}` : '') +
              (s.moods?.length ? ` | ${s.moods.join(', ')}` : ''),
          ),
          '  Para cada una: "id", "reason" breve (por qué encaja con ESTE contenido) y "segment": la parte a usar ("Coro", "Intro", "Drop", "Estribillo final"...). No inventes minutos.',
        ].join('\n')
      : `- "music": 3 canciones reales, muy conocidas y usadas en redes. ${styleRule} Usa solo canciones de las que estés seguro del título y del artista. Para cada una: "title", "artist", "reason" breve y "segment" (la parte a usar: "Coro", "Intro", "Drop"...).`;

    const format = candidates.length
      ? {
          ...strategySchema,
          properties: {
            ...strategySchema.properties,
            music: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  id: { type: 'string', enum: candidates.map((s) => s.id) },
                  title: { type: 'string' },
                  reason: { type: 'string' },
                  segment: { type: 'string' },
                },
                required: ['id', 'title', 'reason', 'segment'],
              },
            },
          },
        }
      : strategySchema;

    const raw = await this.generate({
      model: this.cfg.textModel,
      system:
        'Eres un estratega experto en redes sociales y marketing de contenido en Latinoamérica y España. ' +
        'Respondes SIEMPRE con JSON válido que siga el esquema pedido, sin texto adicional.',
      prompt: [
        ...lines,
        '',
        `Genera en ${languageName(req.language)}:`,
        '- "copy": texto para la publicación con un gancho fuerte en la primera línea, 2-4 frases, emojis con moderación y una llamada a la acción.',
        '- "hashtags": 8 a 12 hashtags relevantes (mezcla de populares y de nicho), cada uno empezando con #.' +
          (req.trendingHashtags?.length
            ? ` Hashtags en TENDENCIA OFICIAL de TikTok${req.trendsRegion ? ` en ${req.trendsRegion}` : ''} esta semana: ` +
              req.trendingHashtags.map((h) => `${h.tag}${h.category ? ` (${h.category})` : ''}`).join(', ') +
              '. Incluye 2 a 4 de ellos SOLO si tienen relación real con el contenido o su contexto (no fuerces los que no encajen); el resto, relevantes al tema.'
            : ''),
        musicInstruction,
        '- "onScreenText": 2 a 4 textos cortos para poner sobre el video/imagen.',
        '- "tips": 2 a 3 consejos concretos para publicar (hora, formato, interacción).',
      ].join('\n'),
      format,
      temperature: 0.7,
    });

    const parsed = safeJson<Partial<ContentStrategy> & { music?: Array<Record<string, unknown>> }>(raw);
    if (!parsed) throw new AppError('Ollama devolvió una estrategia que no es JSON válido', 'AI_SERVICE_ERROR', 502, raw.slice(0, 500));
    if (candidates.length) parsed.music = resolveCatalogMusic(parsed.music, candidates) as never;
    return normalizeStrategy(parsed);
  }

  async listModels(): Promise<string[]> {
    const res = await fetch(`${this.cfg.url}/api/tags`, { signal: AbortSignal.timeout(10_000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = (await res.json()) as { models?: Array<{ name: string }> };
    return (body.models ?? []).map((m) => m.name);
  }

  private async generate(params: GenerateParams): Promise<string> {
    let res: Response;
    try {
      res = await fetch(`${this.cfg.url}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: params.model,
          prompt: params.prompt,
          system: params.system,
          images: params.images,
          format: params.format,
          stream: false,
          keep_alive: this.cfg.keepAlive,
          options: { temperature: params.temperature ?? 0.5, num_ctx: 8192 },
        }),
        signal: AbortSignal.timeout(this.cfg.timeoutMs),
      });
    } catch (err) {
      throw new AppError(
        `No se pudo conectar con Ollama en ${this.cfg.url} (${(err as Error).message}). ¿Está abierto?`,
        'AI_SERVICE_ERROR',
        503,
      );
    }

    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      const hint = res.status === 404 ? ` Ejecuta: ollama pull ${params.model}` : '';
      throw new AppError(`Ollama respondió ${res.status} con el modelo ${params.model}: ${detail}.${hint}`, 'AI_SERVICE_ERROR', 502);
    }
    const body = (await res.json()) as { response?: string };
    return body.response ?? '';
  }

  /**
   * Redimensiona exactamente al tamaño que Ollama usaría internamente para el modelo de visión,
   * de modo que no vuelva a reescalar y las coordenadas devueltas correspondan a esta imagen.
   */
  private async prepareImage(imagePath: string): Promise<PreparedImage> {
    const { width: ow, height: oh } = await this.images.getImageDimensions(imagePath);
    const { width, height } = visionInputSize(ow, oh);

    const tmp = path.join(this.cfg.tempDir, `vision_${crypto.randomUUID()}.jpg`);
    try {
      await this.images.resizeImage(imagePath, tmp, width, height);
      const base64 = (await fs.readFile(tmp)).toString('base64');
      return { base64, width, height, originalWidth: ow, originalHeight: oh };
    } finally {
      await fs.rm(tmp, { force: true });
    }
  }
}

function normalizeStrategy(raw: Partial<ContentStrategy>): ContentStrategy {
  const strings = (v: unknown) => (Array.isArray(v) ? v.map((s) => String(s).trim()).filter(Boolean) : []);
  return {
    copy: String(raw.copy ?? '').trim(),
    hashtags: strings(raw.hashtags).map((h) => (h.startsWith('#') ? h : `#${h}`).replace(/\s+/g, '')),
    music: Array.isArray(raw.music)
      ? raw.music.slice(0, 5).map((m) => ({
          title: String(m?.title ?? '').trim(),
          artist: String(m?.artist ?? '').trim(),
          reason: String(m?.reason ?? '').trim(),
          segment: m?.segment ? String(m.segment).trim() : undefined,
          source: m?.source,
          segmentStart: m?.segmentStart,
          segmentEnd: m?.segmentEnd,
          hookLyric: m?.hookLyric,
          durationSeconds: m?.durationSeconds,
          previewUrl: m?.previewUrl,
          link: m?.link,
          addedAt: m?.addedAt,
        }))
      : [],
    onScreenText: strings(raw.onScreenText),
    tips: strings(raw.tips),
  };
}

function safeJson<T>(raw: string): T | undefined {
  try {
    return JSON.parse(raw) as T;
  } catch {
    const match = raw.match(/\{[\s\S]*\}/);
    if (!match) return undefined;
    try {
      return JSON.parse(match[0]) as T;
    } catch {
      return undefined;
    }
  }
}

function languageName(code: string): string {
  const names: Record<string, string> = { es: 'español', en: 'inglés', pt: 'portugués', fr: 'francés', it: 'italiano' };
  return names[code] ?? code;
}

const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n));

function formatTime(seconds: number) {
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

/** Convierte las elecciones por id en canciones del catálogo; descarta ids desconocidos o repetidos. */
function resolveCatalogMusic(picks: Array<Record<string, unknown>> | undefined, candidates: CatalogSong[]) {
  const byId = new Map(candidates.map((s) => [s.id, s]));
  const used = new Set<string>();
  const titleKey = (s: string) => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-z0-9]/g, '').slice(0, 12);
  const artistKey = (s: string) => titleKey(s.split(/,|&| x | ft.?| feat/i)[0]);
  const usedArtists = new Set<string>();
  const music = (picks ?? []).flatMap((pick) => {
    let song = byId.get(String(pick.id ?? ''));
    // El modelo a veces escribe el id de una canción y la razón de otra: manda el título que nombró.
    const named = typeof pick.title === 'string' && pick.title ? candidates.find((s) => titleKey(s.title) === titleKey(String(pick.title))) : undefined;
    if (named && (!song || titleKey(song.title) !== titleKey(named.title))) song = named;
    if (!song || used.has(song.id) || usedArtists.has(artistKey(song.artist))) return [];
    used.add(song.id);
    usedArtists.add(artistKey(song.artist));
    return [toSuggestion(song, String(pick.reason ?? ''), String(pick.segment ?? ''))];
  });
  // Si el modelo eligió menos de 3 válidas, se completa con las primeras candidatas.
  for (const song of candidates) {
    if (music.length >= 3) break;
    if (!used.has(song.id) && !usedArtists.has(artistKey(song.artist))) {
      used.add(song.id);
      usedArtists.add(artistKey(song.artist));
      music.push(toSuggestion(song, `En tendencia ahora (${song.source}).`, 'Coro'));
    }
  }

  // Garantiza tendencias actuales: el modelo tiende a quedarse con lo "habitual" que ya conoce.
  // Las candidatas vienen ordenadas (estilo pedido > tema > general; lo más reciente primero).
  // Garantiza lo POPULAR: al menos 1 de las 3 debe estar entre lo más escuchado del país esta semana.
  // Si hay que reemplazar, se elige al azar entre las 12 más escuchadas (no siempre la #1).
  const isTrend = (source?: string) => /^(Top |Tendencia|Viral)/.test(source ?? '');
  const isTop = (source?: string) => /^Top /.test(source ?? '');
  const topPool = shuffle(candidates.filter((s) => isTop(s.source) && !used.has(s.id) && !usedArtists.has(artistKey(s.artist))).slice(0, 12));
  const wanted = Math.min(1, topPool.length + music.filter((m) => isTop(m.source)).length);
  while (music.filter((m) => isTop(m.source)).length < wanted && topPool.length) {
    const flags = music.map((m) => isTop(m.source));
    const replaceAt = flags.lastIndexOf(false);
    if (replaceAt < 0) break;
    const song = topPool.shift()!;
    if (usedArtists.has(artistKey(song.artist))) continue;
    used.add(song.id);
    usedArtists.add(artistKey(song.artist));
    music[replaceAt] = toSuggestion(song, `Es de lo más escuchado ahora (${song.source}${song.genre ? `, ${song.genre}` : ''}).`, 'Coro');
  }
  // Si el usuario importó el ranking oficial de TikTok, al menos una sugerencia sale de él.
  const isOfficial = (source?: string) => Boolean(source?.startsWith('Tendencia oficial'));
  const official = candidates.find((s) => isOfficial(s.source) && !used.has(s.id));
  if (official && !music.some((m) => isOfficial(m.source)) && music.length) {
    // Se reemplaza la última no oficial (preferiblemente una no-tendencia).
    const nonTrend = music.map((m) => isTrend(m.source)).lastIndexOf(false);
    const idx = nonTrend >= 0 ? nonTrend : music.length - 1;
    used.add(official.id);
    music[idx] = toSuggestion(official, `Está en el ranking oficial de TikTok (${official.source.replace(/^Tendencia oficial TikTok\s*·?\s*/, '')}).`, 'Coro');
  }

  // Primero el ranking oficial, luego el resto de tendencias.
  const weight = (s?: string) => (isOfficial(s) ? 3 : isTop(s) ? 2 : isTrend(s) ? 1 : 0);
  return music.slice(0, 3).sort((a, b) => weight(b.source) - weight(a.source));
}

function toSuggestion(song: CatalogSong, reason: string, segment: string) {
  return {
    title: song.title,
    artist: song.artist,
    reason,
    segment: segment || undefined,
    source: song.source,
    previewUrl: song.previewUrl,
    link: song.link,
    addedAt: song.addedAt,
  };
}

/** Copia barajada (Fisher-Yates). */
function shuffle<T>(items: T[]): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
