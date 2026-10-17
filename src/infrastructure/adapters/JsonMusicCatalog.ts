import fs from 'node:fs/promises';
import type { CatalogSong, IMusicCatalog, MusicCandidates, MusicCandidatesRequest } from '../../domain/ports/IMusicCatalog';

interface CatalogFile {
  genres: Array<{ genre: string; aliases?: string[]; songs: Array<{ title: string; artist: string; moods?: string[] }> }>;
}

interface Genre {
  name: string;
  keys: string[];
  songs: CatalogSong[];
}

const normalize = (s: string) =>
  s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    // Cualquier otro carácter (incluido texto con acentos mal codificados, ej. "reggaetÃ³n") se descarta
    // sin dejar hueco, para que "reggaet?n" siga pareciéndose a "reggaeton".
    .replace(/[^a-z0-9 ]+/g, '')
    .replace(/\s+/g, ' ')
    .trim();

const slug = (s: string) => normalize(s).replace(/\s+/g, '-');

/**
 * Catálogo local de data/music-catalog.json (editable). Se usa como respaldo cuando no hay
 * conexión para consultar tendencias en vivo.
 */
export class JsonMusicCatalog implements IMusicCatalog {
  private loaded?: Promise<Genre[]>;

  constructor(private readonly filePath: string) {}

  async candidates({ musicStyle, limit }: MusicCandidatesRequest): Promise<MusicCandidates> {
    const genres = await this.load();
    const wanted = musicStyle ? normalize(musicStyle) : '';
    // Puntuación por especificidad: exacta > el texto contiene la clave > la clave contiene el texto.
    const score = (g: Genre) =>
      Math.max(
        0,
        ...g.keys.map((k) => {
          if (k === wanted) return 1000;
          if (wanted.includes(k)) return k.length * 2;
          if (k.includes(wanted)) return wanted.length;
          // Erratas o acentos mal codificados ("regeton", "reggaetan"): coincidencia aproximada.
          if (k.length >= 5 && editDistance(k, wanted) <= 2) return k.length;
          return 0;
        }),
      );
    const best = wanted ? Math.max(0, ...genres.map(score)) : 0;
    const matched = best > 0 ? genres.filter((g) => score(g) === best) : [];

    if (matched.length) {
      const songs = dedupe(matched.flatMap((g) => g.songs));
      return { songs: songs.slice(0, limit), live: false };
    }
    // Sin género concreto: muestra variada (unas cuantas de cada género) para que la IA elija por contexto.
    const perGenre = Math.max(2, Math.floor(limit / genres.length));
    const songs = dedupe(genres.flatMap((g) => g.songs.slice(0, perGenre)));
    return { songs: songs.slice(0, limit), live: false };
  }

  private load(): Promise<Genre[]> {
    this.loaded ??= fs.readFile(this.filePath, 'utf8').then((raw) => {
      const data = JSON.parse(raw) as CatalogFile;
      return data.genres.map((g) => ({
        name: g.genre,
        keys: [g.genre, ...(g.aliases ?? [])].map(normalize).filter(Boolean),
        songs: g.songs.map((s) => ({
          id: slug(`${s.title} ${s.artist}`),
          title: s.title,
          artist: s.artist,
          source: 'Catálogo local',
          genre: g.genre,
          moods: s.moods ?? [],
        })),
      }));
    });
    return this.loaded;
  }
}

function dedupe(songs: CatalogSong[]) {
  const seen = new Set<string>();
  return songs.filter((s) => !seen.has(s.id) && seen.add(s.id));
}

function editDistance(a: string, b: string): number {
  if (Math.abs(a.length - b.length) > 2) return 99;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
}
