import type { OfficialTrendSong } from '../../domain/ports/ITikTokTrendsStore';

/**
 * Convierte texto pegado (Creative Center de TikTok, listas, hojas de cálculo...) en canciones.
 * Formatos admitidos:
 *  - Una por línea: "Canción - Artista", "Canción — Artista", "Canción by Artista", "Canción | Artista",
 *    columnas separadas por tabulador, con o sin número delante ("1.", "#1", "1)").
 *  - Bloques del Creative Center: una línea con el número de puesto, seguida del título y del artista.
 * Las líneas de interfaz ("Ver análisis", "Nuevo", "1.2M", "Popular"...) se ignoran.
 */
export function parseTrendsText(text: string, max = 100): OfficialTrendSong[] {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.replace(/ /g, ' ').trim())
    .filter((l) => l && !isNoise(l));

  const songs: OfficialTrendSong[] = [];
  const seen = new Set<string>();
  const push = (title: string, artist: string, rank?: number) => {
    title = clean(title);
    artist = clean(artist);
    if (!title || !artist || title.length > 120 || artist.length > 120) return;
    if (HEADER.test(title) || HEADER.test(artist)) return; // fila de encabezados de una tabla
    if (title.startsWith('#') || artist.startsWith('#')) return; // es un hashtag, no una canción
    const key = `${title}|${artist}`.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    songs.push({ rank: rank ?? songs.length + 1, title, artist });
  };

  for (let i = 0; i < lines.length && songs.length < max; i++) {
    const line = lines[i];
    const rankOnly = /^#?(\d{1,3})[.)]?$/.exec(line);
    if (rankOnly) {
      // Bloque: número de puesto + título + artista en las líneas siguientes.
      const title = lines[i + 1];
      const artist = lines[i + 2];
      if (title && artist && !/^#?\d{1,3}[.)]?$/.test(title) && !/^#?\d{1,3}[.)]?$/.test(artist)) {
        const inline = splitInline(title);
        if (inline) push(inline[0], inline[1], Number(rankOnly[1]));
        else push(title, artist, Number(rankOnly[1]));
        i += inline ? 1 : 2;
      }
      continue;
    }
    const withRank = /^#?(\d{1,3})[.)\s-]+\s*(.+)$/.exec(line);
    const rank = withRank ? Number(withRank[1]) : undefined;
    const body = withRank ? withRank[2] : line;
    const inline = splitInline(body);
    if (inline) push(inline[0], inline[1], rank);
  }
  return songs.sort((a, b) => a.rank - b.rank);
}

function splitInline(s: string): [string, string] | null {
  const tab = s.split('\t').map((p) => p.trim()).filter(Boolean);
  if (tab.length >= 2) {
    // En hojas de cálculo puede venir "puesto<TAB>título<TAB>artista".
    const parts = /^\d+$/.test(tab[0]) ? tab.slice(1) : tab;
    if (parts.length >= 2) return [parts[0], parts[1]];
  }
  // Guiones y barras: se divide por el ÚLTIMO separador ("Canción - Remix - Artista").
  const dash = /^(.+)\s+(?:-|–|—|\||·)\s+(.+?)$/.exec(s);
  if (dash) return [dash[1], dash[2]];
  // "by" solo en inglés ("de" no: aparece dentro de muchos títulos en español).
  const by = /^(.+?)\s+by\s+(.+)$/i.exec(s);
  return by ? [by[1], by[2]] : null;
}

const NOISE = [
  /^(ver|see|view)\b/i,
  /an[aá]lisis|analytics|details?|detalles/i,
  /^(nuevo|new|popular|tendencia|trending|breakout|rising|en alza|sube|baja)$/i,
  // Contadores ("1.2M", "850K videos", "3 mil"); un número solo NO es ruido: es el puesto.
  /^[\d.,]+\s*([kmb]|mil|millones?)(\s*(videos?|vídeos?|posts?|publicaciones))?$/i,
  /^[\d.,]+\s*(videos?|vídeos?|posts?|publicaciones)$/i,
  /^[+-]?\d+([.,]\d+)?\s*%$/,
  /^https?:\/\//i,
  /^(canci[oó]n|song|artista|artist|puesto|rank(ing)?|t[ií]tulo|title)$/i,
  /^\d{1,2}:\d{2}$/, // duraciones "0:30"
];

const HEADER = /^(canci[oó]n|song|artista|artist|puesto|rank(ing)?|t[ií]tulo|title|#)$/i;

function isNoise(line: string) {
  return NOISE.some((r) => r.test(line));
}

function clean(s: string) {
  return s
    .replace(/^["'“”«»]+|["'“”«»]+$/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
}
