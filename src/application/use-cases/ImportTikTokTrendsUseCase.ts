import { AppError } from '../../domain/errors/AppError';
import type { ISongLookup, ITikTokTrendsStore, OfficialTrends } from '../../domain/ports/ITikTokTrendsStore';
import { parseTrendsText } from '../services/parseTrendsText';
import { parseHashtagsText } from '../services/parseHashtagsText';

export interface ImportResult extends OfficialTrends {
  /** Qué se actualizó con este texto. */
  imported: 'songs' | 'hashtags';
}

/**
 * Importa lo que el usuario copia del Creative Center de TikTok (con su propia sesión):
 * la tabla de Hashtags o, si su país la tiene, la de Música. Detecta cuál es por el contenido.
 */
export class ImportTikTokTrendsUseCase {
  constructor(
    private readonly store: ITikTokTrendsStore,
    private readonly lookup: ISongLookup,
  ) {}

  async execute(input: { text: string; region: string }): Promise<ImportResult> {
    const now = new Date().toISOString();
    const current: OfficialTrends = (await this.store.get()) ?? { region: input.region, updatedAt: now, songs: [] };

    const hashtags = parseHashtagsText(input.text, 60);
    if (hashtags.length >= 1) {
      const trends: OfficialTrends = { ...current, region: input.region, hashtags, hashtagsUpdatedAt: now };
      await this.store.save(trends);
      return { ...trends, imported: 'hashtags' };
    }

    const songs = parseTrendsText(input.text, 60);
    if (songs.length === 0) {
      throw AppError.validation(
        'No se reconocieron hashtags ni canciones. Copia la tabla de Hashtags (o de Música) del Creative Center tal cual, o escribe una canción por línea como "Canción - Artista".',
      );
    }
    // Preview de 30 s y enlace, por lotes; si una búsqueda falla, la canción se guarda igual.
    for (let i = 0; i < songs.length; i += 6) {
      await Promise.all(
        songs.slice(i, i + 6).map(async (song) => {
          const found = await this.lookup.lookup(song.title, song.artist).catch(() => null);
          if (found) Object.assign(song, found);
        }),
      );
    }
    const trends: OfficialTrends = { ...current, region: input.region, updatedAt: now, songs };
    await this.store.save(trends);
    return { ...trends, imported: 'songs' };
  }
}
