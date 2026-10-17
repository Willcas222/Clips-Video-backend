import type { CatalogSong, IMusicCatalog, MusicCandidates, MusicCandidatesRequest } from '../../domain/ports/IMusicCatalog';
import type { ISongLookup } from '../../domain/ports/ITikTokTrendsStore';

interface DeezerPlaylist {
  id: number;
  title: string;
  nb_tracks: number;
  user?: { name?: string };
}

interface DeezerTrack {
  id: number;
  title: string;
  duration: number;
  rank: number;
  preview?: string;
  link?: string;
  time_add?: number;
  artist: { name: string };
}

interface AppleSong {
  name: string;
  artistName: string;
  url?: string;
  genres?: Array<{ name: string }>;
}

const COUNTRIES: Record<string, { code: string; name: string }> = {
  colombia: { code: 'co', name: 'Colombia' },
  mexico: { code: 'mx', name: 'México' },
  espana: { code: 'es', name: 'España' },
  argentina: { code: 'ar', name: 'Argentina' },
  chile: { code: 'cl', name: 'Chile' },
  peru: { code: 'pe', name: 'Perú' },
  ecuador: { code: 'ec', name: 'Ecuador' },
  venezuela: { code: 've', name: 'Venezuela' },
  'estados unidos': { code: 'us', name: 'Estados Unidos' },
};

const CACHE_MS = 6 * 60 * 60 * 1000;
/** Una lista sin canciones nuevas en este tiempo ya no es tendencia. */
const FRESH_DAYS = 120;

const normalize = (s: string) =>
  s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

/** Clave para cruzar la misma canción entre Apple Music y Deezer. */
const songKey = (title: string, artist: string) =>
  `${normalize(title).replace(/\(.*?\)|\[.*?\]|feat\..*$/g, '').replace(/[^a-z0-9]/g, '').slice(0, 14)}|${normalize(artist.split(/,|&| x | ft\.?| feat/i)[0]).replace(/[^a-z0-9]/g, '')}`;

/**
 * Música en tendencia, priorizando lo que MÁS SE ESCUCHA EN EL PAÍS esta semana:
 *  1. Top del país: Apple Music (más reproducidas, con género) + "Top <país>" de Deezer Charts.
 *  2. Estilo pedido: listas recientes de ese estilo.
 *  3. Viral en TikTok: unas pocas de listas editoriales recientes.
 * Solo se consultan rankings públicos (sin enviar fotos ni videos). Sin conexión, catálogo local.
 */
export class TrendingMusicCatalog implements IMusicCatalog, ISongLookup {
  private readonly cache = new Map<string, { at: number; value: unknown }>();

  constructor(
    private readonly fallback: IMusicCatalog,
    private readonly defaultCountry = 'colombia',
    private readonly timeoutMs = 8000,
  ) {}

  async candidates(req: MusicCandidatesRequest): Promise<MusicCandidates> {
    try {
      const songs = await this.trending(req);
      if (songs.length >= 5) return { songs, live: true };
    } catch (err) {
      console.warn('[music] tendencias no disponibles:', (err as Error).message);
    }
    const local = await this.fallback.candidates(req);
    return { ...local, note: 'No se pudieron consultar los rankings de música (¿sin internet?). Se usó el catálogo local.' };
  }

  /** Preview de 30 s y enlace de una canción concreta. */
  async lookup(title: string, artist: string) {
    const strict = await this.deezer<{ data?: DeezerTrack[] }>(`/search?q=${encodeURIComponent(`artist:"${artist}" track:"${title}"`)}&limit=1`);
    const track =
      strict.data?.[0] ?? (await this.deezer<{ data?: DeezerTrack[] }>(`/search?q=${encodeURIComponent(`${title} ${artist}`)}&limit=1`)).data?.[0];
    if (!track) return null;
    // Evita asociar otra canción: el título encontrado debe parecerse al buscado.
    const a = normalize(track.title).replace(/\(.*?\)/g, '').trim();
    const b = normalize(title).replace(/\(.*?\)/g, '').trim();
    if (!a.includes(b.slice(0, 8)) && !b.includes(a.slice(0, 8))) return null;
    return { previewUrl: track.preview || undefined, link: track.link };
  }

  private async trending(req: MusicCandidatesRequest): Promise<CatalogSong[]> {
    const country = COUNTRIES[normalize(req.country ?? '')] ?? Object.values(COUNTRIES).find((c) => c.code === req.country) ?? COUNTRIES[this.defaultCountry];
    const style = req.musicStyle?.trim();

    const [top, styled, viral] = await Promise.all([
      this.countryTop(country).catch((err) => {
        console.warn('[music] top del país no disponible:', (err as Error).message);
        return [] as CatalogSong[];
      }),
      style ? this.styleTrend(style, country).catch(() => [] as CatalogSong[]) : Promise.resolve([] as CatalogSong[]),
      this.tiktokViral(req.language).catch(() => [] as CatalogSong[]),
    ]);

    // Cupo: el top del país es la base; el estilo pedido gana espacio si existe.
    const quota = style ? { top: 16, styled: 9, viral: 5 } : { top: 22, styled: 0, viral: 8 };
    const seen = new Set<string>();
    const perArtist = new Map<string, number>();
    const out: CatalogSong[] = [];
    const take = (list: CatalogSong[], n: number) => {
      let taken = 0;
      for (const s of list) {
        if (taken >= n) break;
        const key = songKey(s.title, s.artist);
        // Variedad: máximo 2 canciones del mismo artista principal.
        const artist = key.split('|')[1];
        if (seen.has(key) || (perArtist.get(artist) ?? 0) >= 2) continue;
        seen.add(key);
        perArtist.set(artist, (perArtist.get(artist) ?? 0) + 1);
        out.push(s);
        taken++;
      }
    };
    take(top, quota.top);
    take(styled, quota.styled);
    take(viral, quota.viral);
    return out.slice(0, req.limit);
  }

  /** Lo más escuchado esta semana en el país: Apple Music (orden y género) + Deezer Charts (preview). */
  private async countryTop(country: { code: string; name: string }): Promise<CatalogSong[]> {
    const [apple, deezerChart] = await Promise.all([
      // Host definitivo (el antiguo redirige y a veces supera el tiempo límite).
      this.json<{ feed?: { results?: AppleSong[] } }>(`https://rss.marketingtools.apple.com/api/v2/${country.code}/music/most-played/50/songs.json`, 15_000)
        .then((r) => r.feed?.results ?? [])
        .catch((err) => {
          console.warn("[music] Apple Music no disponible:", (err as Error).message);
          return [] as AppleSong[];
        }),
      this.deezerChart(country.name).catch(() => [] as DeezerTrack[]),
    ]);
    const byKey = new Map(deezerChart.map((t) => [songKey(t.title, t.artist.name), t]));

    const songs: CatalogSong[] = apple.map((s, i) => {
      const dz = byKey.get(songKey(s.name, s.artistName));
      const genre = (s.genres ?? []).map((g) => g.name).filter((g) => !/^m[uú]sica$|^music$/i.test(g)).join(', ');
      return {
        id: dz ? `dz${dz.id}` : `am${i + 1}`,
        title: s.name,
        artist: s.artistName,
        source: `Top ${country.name} #${i + 1}`,
        genre: genre || undefined,
        previewUrl: dz?.preview || undefined,
        link: dz?.link ?? s.url,
      };
    });
    // Si Apple Music no respondió, el ranking de Deezer basta por sí solo.
    if (songs.length === 0) {
      deezerChart.forEach((t, i) =>
        songs.push({ id: `dz${t.id}`, title: t.title, artist: t.artist.name, source: `Top ${country.name} #${i + 1}`, previewUrl: t.preview, link: t.link }),
      );
    }
    // Completa el preview de las primeras que no lo tengan (para poder escucharlas en la app).
    await Promise.all(
      songs.slice(0, 22).map(async (s) => {
        if (s.previewUrl) return;
        const found = await this.lookup(s.title, s.artist).catch(() => null);
        if (found) Object.assign(s, { previewUrl: found.previewUrl, link: found.link ?? s.link });
      }),
    );
    return songs;
  }

  private async deezerChart(countryName: string): Promise<DeezerTrack[]> {
    const wanted = normalize(`top ${countryName}`);
    const lists = await this.deezer<{ data?: DeezerPlaylist[] }>(`/search/playlist?q=${encodeURIComponent(`top ${countryName}`)}&limit=15`);
    const chart = (lists.data ?? []).find((p) => /deezer charts/i.test(p.user?.name ?? '') && normalize(p.title) === wanted);
    if (!chart) return [];
    const tracks = await this.deezer<{ data?: DeezerTrack[] }>(`/playlist/${chart.id}/tracks?limit=100`);
    return (tracks.data ?? []).filter((t) => t.title && t.artist?.name);
  }

  /** Canciones recientes del estilo pedido (primero listas del país, luego de TikTok). */
  private async styleTrend(style: string, country: { name: string }): Promise<CatalogSong[]> {
    for (const query of [`${style} ${country.name}`, `${style} hits`, `tiktok ${style}`]) {
      const { tracks, fresh } = await this.bestPlaylist(query, style);
      if (fresh && tracks.length >= 8) return tracks.map((t) => toSong(t, `Tendencia · ${style}`));
    }
    return [];
  }

  private async tiktokViral(language: string): Promise<CatalogSong[]> {
    for (const query of language === 'es' ? ['salidas de tiktok', 'tiktok hits'] : ['tiktok hits', 'tiktok viral']) {
      const { tracks, fresh } = await this.bestPlaylist(query, 'tiktok');
      if (fresh && tracks.length) return tracks.map((t) => toSong(t, 'Viral en TikTok'));
    }
    return [];
  }

  /** Mejor lista para la búsqueda; sus canciones de la más reciente a la más antigua. */
  private async bestPlaylist(query: string, term: string): Promise<{ tracks: DeezerTrack[]; fresh: boolean }> {
    const lists = await this.deezer<{ data?: DeezerPlaylist[] }>(`/search/playlist?q=${encodeURIComponent(query)}&limit=15`);
    const ranked = (lists.data ?? []).map((p) => ({ p, score: scorePlaylist(p, term) })).sort((a, b) => b.score - a.score);
    if (!ranked[0] || ranked[0].score < 2) return { tracks: [], fresh: false };
    const tracks = await this.deezer<{ data?: DeezerTrack[] }>(`/playlist/${ranked[0].p.id}/tracks?limit=100`);
    const list = (tracks.data ?? [])
      .filter((t) => t.duration >= 45 && t.title && t.artist?.name)
      .sort((a, b) => (b.time_add ?? 0) - (a.time_add ?? 0) || b.rank - a.rank);
    const newest = (list[0]?.time_add ?? 0) * 1000;
    return { tracks: list, fresh: Date.now() - newest < FRESH_DAYS * 86_400_000 };
  }

  private deezer<T>(path: string): Promise<T> {
    return this.json<T>(`https://api.deezer.com${path}`);
  }

  private async json<T>(url: string, timeoutMs = this.timeoutMs): Promise<T> {
    const hit = this.cache.get(url);
    if (hit && Date.now() - hit.at < CACHE_MS) return hit.value as T;
    const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs), redirect: 'follow' });
    if (!res.ok) throw new Error(`HTTP ${res.status} en ${new URL(url).host}`);
    const value = (await res.json()) as T & { error?: { message?: string } };
    if (value && typeof value === 'object' && 'error' in value && value.error) throw new Error(`Deezer: ${value.error.message ?? 'error'}`);
    this.cache.set(url, { at: Date.now(), value });
    return value;
  }
}

function toSong(t: DeezerTrack, source: string): CatalogSong {
  return {
    id: `dz${t.id}`,
    title: t.title,
    artist: t.artist.name,
    source,
    previewUrl: t.preview || undefined,
    link: t.link,
    addedAt: t.time_add ? new Date(t.time_add * 1000).toISOString().slice(0, 10) : undefined,
  };
}

/** Prioriza listas de editores y del año en curso que mencionan el término buscado. */
function scorePlaylist(p: DeezerPlaylist, term: string): number {
  const title = normalize(p.title);
  const year = new Date().getFullYear();
  let score = 0;
  if (/deezer|editor/i.test(p.user?.name ?? '')) score += 4;
  if (title.includes(normalize(term))) score += 3;
  if (title.includes(String(year))) score += 2;
  else if (title.includes(String(year - 1))) score += 1;
  else if (/20\d\d/.test(title)) score -= 3; // lista de años anteriores
  if (p.nb_tracks >= 25 && p.nb_tracks <= 300) score += 1;
  else score -= 1;
  return score;
}
