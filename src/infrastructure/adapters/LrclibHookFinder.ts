import type { IMusicHookFinder, SongHook } from '../../domain/ports/IMusicHookFinder';

interface LrclibTrack {
  trackName: string;
  artistName: string;
  duration: number;
  instrumental?: boolean;
  syncedLyrics?: string | null;
}

interface Line {
  t: number;
  text: string;
  key: string;
}

/** Duración ideal del tramo para TikTok/Reels/Historias. */
const MIN_LEN = 12;
const MAX_LEN = 30;
const CACHE_MS = 24 * 60 * 60 * 1000;

const normalize = (s: string) =>
  s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9 ]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

/**
 * Encuentra el coro con letras sincronizadas de LRCLIB (base pública y gratuita, sin clave).
 * El coro = el bloque de líneas consecutivas que más se repite en la canción; se devuelve
 * su PRIMERA aparición (la que se usa en los videos), ajustada a 12-30 segundos.
 */
export class LrclibHookFinder implements IMusicHookFinder {
  private readonly cache = new Map<string, { at: number; value: SongHook | null }>();

  constructor(private readonly timeoutMs = 10_000) {}

  async findHook(title: string, artist: string): Promise<SongHook | null> {
    const cacheKey = `${normalize(title)}|${normalize(artist)}`;
    const hit = this.cache.get(cacheKey);
    if (hit && Date.now() - hit.at < CACHE_MS) return hit.value;

    let value: SongHook | null = null;
    try {
      const track = await this.search(title, artist);
      if (track?.syncedLyrics) value = findChorus(parseLrc(track.syncedLyrics), track.duration);
    } catch (err) {
      console.warn(`[music] sin letra sincronizada para "${title}":`, (err as Error).message);
    }
    this.cache.set(cacheKey, { at: Date.now(), value });
    return value;
  }

  private async search(title: string, artist: string): Promise<LrclibTrack | undefined> {
    // Título sin "(feat. …)", "(Remix)"…: mejora las coincidencias.
    const cleanTitle = title.replace(/\s*[([].*?[)\]]\s*/g, ' ').trim() || title;
    const mainArtist = artist.split(/,|&| x | ft\.?| feat/i)[0].trim();
    const url = `https://lrclib.net/api/search?track_name=${encodeURIComponent(cleanTitle)}&artist_name=${encodeURIComponent(mainArtist)}`;
    const res = await fetch(url, {
      signal: AbortSignal.timeout(this.timeoutMs),
      headers: { 'User-Agent': 'asistente-contenido-local/1.0 (uso personal)' },
    });
    if (!res.ok) throw new Error(`LRCLIB HTTP ${res.status}`);
    const results = (await res.json()) as LrclibTrack[];
    const wanted = normalize(cleanTitle);
    return results.find((r) => r.syncedLyrics && !r.instrumental && normalize(r.trackName).includes(wanted.slice(0, 10)));
  }
}

function parseLrc(lrc: string): Line[] {
  const lines: Line[] = [];
  for (const raw of lrc.split(/\r?\n/)) {
    const m = /^\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$/.exec(raw.trim());
    if (!m) continue;
    const text = m[3].trim();
    if (!text) continue;
    lines.push({ t: Number(m[1]) * 60 + Number(m[2]), text, key: normalize(text) });
  }
  return lines;
}

/** Bloque de líneas repetidas (coro) más fuerte; primera aparición, ajustada a 12-30 s. */
export function findChorus(lines: Line[], duration: number): SongHook | null {
  if (lines.length < 6) return null;
  const counts = new Map<string, number>();
  for (const l of lines) if (l.key.length >= 6) counts.set(l.key, (counts.get(l.key) ?? 0) + 1);
  const repeated = (l: Line) => (counts.get(l.key) ?? 0) >= 2;

  // Bloques de líneas consecutivas repetidas (se tolera 1 línea suelta en medio).
  const blocks: Array<{ from: number; to: number; score: number }> = [];
  for (let i = 0; i < lines.length; i++) {
    if (!repeated(lines[i])) continue;
    let j = i;
    let gaps = 0;
    while (j + 1 < lines.length && (repeated(lines[j + 1]) || (gaps === 0 && j + 2 < lines.length && repeated(lines[j + 2]) && ++gaps))) j++;
    const block = lines.slice(i, j + 1).filter(repeated);
    // Puntuación: cuántas veces se repite el bloque × su tamaño; leve preferencia por que no sea el intro.
    blocks.push({ from: i, to: j, score: block.reduce((s, l) => s + (counts.get(l.key) ?? 0), 0) * (lines[i].t >= 10 ? 1 : 0.7) });
    i = j;
  }
  if (!blocks.length) return null;
  // Entre los bloques casi tan fuertes como el mejor, el que suena PRIMERO: es el que la gente reconoce.
  const top = Math.max(...blocks.map((b) => b.score));
  const best = blocks.find((b) => b.score >= top * 0.7)!;

  // Primera aparición del bloque ganador (lo que se usa en los videos).
  const firstKey = lines[best.from].key;
  const first = lines.findIndex((l) => l.key === firstKey);
  const fromIdx = first >= 0 ? first : best.from;
  const start = Math.max(0, lines[fromIdx].t - 0.3);
  // Fin: se avanza por líneas hasta cubrir el bloque, sin pasar de 30 s.
  const blockLen = best.to - best.from;
  let toIdx = Math.min(lines.length - 1, fromIdx + blockLen + 1);
  while (toIdx > fromIdx + 1 && lines[toIdx].t - start > MAX_LEN) toIdx--;
  let end = lines[toIdx].t;
  while (end - start < MIN_LEN && toIdx + 1 < lines.length) end = lines[++toIdx].t;
  end = Math.min(end, start + MAX_LEN, duration || Infinity);
  if (end - start < 6) return null;

  return { start: Math.round(start), end: Math.round(end), lyric: lines[fromIdx].text, durationSeconds: Math.round(duration) || undefined, method: 'coro' };
}
