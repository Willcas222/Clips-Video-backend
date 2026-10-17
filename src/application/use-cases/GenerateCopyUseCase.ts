import type { ContentStrategy } from '../../domain/entities/ContentStrategy';
import type { CreativeBrief, SocialPlatform } from '../../domain/entities/Media';
import { silentReporter, type ProgressReporter } from '../../domain/entities/Job';
import type { IAIAssistant } from '../../domain/ports/IAIAssistant';
import type { IMusicCatalog } from '../../domain/ports/IMusicCatalog';
import type { ITikTokTrendsStore } from '../../domain/ports/ITikTokTrendsStore';
import type { IMusicHookFinder } from '../../domain/ports/IMusicHookFinder';
import { addSongHooks } from '../services/musicHooks';
import type { IRecentSuggestions } from '../../domain/ports/IRecentSuggestions';
import { withoutRecent } from '../services/musicVariety';
import { applyTrendingHashtags, loadTrendingHashtags, relevantHashtags } from '../services/trendingHashtags';

export interface GenerateCopyInput extends CreativeBrief {
  imagePath: string;
  platform: SocialPlatform;
  language: string;
}

export interface GenerateCopyResult {
  visualDescription: string;
  strategy: ContentStrategy;
}

/** Flujo A (parte 2): analizar la foto con visión y redactar copy, hashtags y música. */
export class GenerateCopyUseCase {
  constructor(
    private readonly ai: IAIAssistant,
    private readonly music: IMusicCatalog,
    private readonly trends: ITikTokTrendsStore,
    private readonly hooks: IMusicHookFinder,
    private readonly recentMusic: IRecentSuggestions,
  ) {}

  async execute(input: GenerateCopyInput, progress: ProgressReporter = silentReporter): Promise<GenerateCopyResult> {
    progress.step('Analizando la imagen con IA de visión', 75);
    const visualDescription = await this.ai.describeImage(input.imagePath, input.language, input.context);

    progress.step('Buscando música en tendencia para este contenido', 85);
    const region = (await this.trends.get())?.region;
    const music = await this.music.candidates({
      country: region || undefined,
      musicStyle: input.musicStyle,
      themeText: `${input.context ?? ''} ${visualDescription}`,
      language: input.language,
      limit: 40,
    });
    // Variedad: no se vuelven a ofrecer las canciones sugeridas en los últimos trabajos.
    const candidates = withoutRecent(music.songs, await this.recentMusic.recent()).slice(0, 30);
    if (music.note) progress.warn(music.note);
    progress.step('Redactando copy, hashtags y eligiendo música', 90);
    const trending = await loadTrendingHashtags(this.trends);
    const relevant = relevantHashtags(trending.hashtags, `${input.context ?? ''} ${visualDescription}`);
    const strategy = await this.ai.generateStrategy({
      mediaKind: 'image',
      platform: input.platform,
      visualDescription,
      language: input.language,
      tone: input.tone,
      context: input.context,
      musicStyle: input.musicStyle,
      musicCandidates: candidates,
      trendingHashtags: relevant,
      trendsRegion: trending.region,
    });
    return { visualDescription, strategy: await this.finish(applyTrendingHashtags(strategy, trending.hashtags, relevant)) };
  }

  /** Añade el tramo exacto de cada canción y la recuerda para no repetirla en los próximos trabajos. */
  private async finish(strategy: ContentStrategy): Promise<ContentStrategy> {
    const withHooks = await addSongHooks(strategy, this.hooks);
    await this.recentMusic.remember(withHooks.music).catch(() => undefined);
    return withHooks;
  }
}
