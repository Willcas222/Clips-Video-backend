import type { CatalogSong, IMusicCatalog, MusicCandidates, MusicCandidatesRequest } from '../../domain/ports/IMusicCatalog';
import type { ITikTokTrendsStore } from '../../domain/ports/ITikTokTrendsStore';

export const OFFICIAL_SOURCE = 'Tendencia oficial TikTok';

/** Pone primero el ranking oficial que importó el usuario y completa con la fuente interna. */
export class OfficialTrendsCatalog implements IMusicCatalog {
  constructor(
    private readonly store: ITikTokTrendsStore,
    private readonly inner: IMusicCatalog,
    /** Máximo de canciones oficiales que se ofrecen a la IA por petición. */
    private readonly maxOfficial = 15,
  ) {}

  async candidates(req: MusicCandidatesRequest): Promise<MusicCandidates> {
    const official = await this.store.get();
    const songs: CatalogSong[] = (official?.songs ?? []).slice(0, this.maxOfficial).map((s) => ({
      id: `tt${s.rank}`,
      title: s.title,
      artist: s.artist,
      source: `${OFFICIAL_SOURCE}${official?.region ? ` · ${official.region}` : ''} #${s.rank}`,
      previewUrl: s.previewUrl,
      link: s.link,
      addedAt: official?.updatedAt.slice(0, 10),
    }));

    const rest = await this.inner.candidates({ ...req, limit: Math.max(10, req.limit - songs.length) });
    const seen = new Set(songs.map((s) => key(s)));
    const merged = [...songs, ...rest.songs.filter((s) => !seen.has(key(s)))];

    let note = rest.note;
    if (official?.songs.length) {
      const days = (Date.now() - new Date(official.updatedAt).getTime()) / 86_400_000;
      if (days > 7) note = `La lista oficial de TikTok tiene ${Math.floor(days)} días: actualízala en “Tendencias oficiales” para sugerencias más actuales.`;
    }
    return { songs: merged, live: rest.live || songs.length > 0, note };
  }
}

const key = (s: { title: string; artist: string }) => `${s.title}|${s.artist}`.toLowerCase().replace(/\s+/g, ' ');
