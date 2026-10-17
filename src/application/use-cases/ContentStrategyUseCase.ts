import type { ContentStrategy } from '../../domain/entities/ContentStrategy';
import type { CreativeBrief, SocialPlatform } from '../../domain/entities/Media';
import type { Transcript } from '../../domain/entities/ProcessedContent';
import { silentReporter, type ProgressReporter } from '../../domain/entities/Job';
import type { FrameAnalysis, IAIAssistant } from '../../domain/ports/IAIAssistant';
import type { IMusicCatalog } from '../../domain/ports/IMusicCatalog';
import type { ITikTokTrendsStore } from '../../domain/ports/ITikTokTrendsStore';
import type { IMusicHookFinder } from '../../domain/ports/IMusicHookFinder';
import { addSongHooks } from '../services/musicHooks';
import type { IRecentSuggestions } from '../../domain/ports/IRecentSuggestions';
import { withoutRecent } from '../services/musicVariety';
import { applyTrendingHashtags, loadTrendingHashtags, relevantHashtags } from '../services/trendingHashtags';

export interface ContentStrategyInput extends CreativeBrief {
  platform: SocialPlatform;
  language: string;
  durationSeconds: number;
  frameAnalyses: FrameAnalysis[];
  transcript?: Transcript;
}

export interface ContentStrategyResult {
  visualDescription: string;
  strategy: ContentStrategy;
}

/** Flujo B (parte 2): música, tramo de la canción y textos a partir de lo que ocurre en el video. */
export class ContentStrategyUseCase {
  constructor(
    private readonly ai: IAIAssistant,
    private readonly music: IMusicCatalog,
    private readonly trends: ITikTokTrendsStore,
    private readonly hooks: IMusicHookFinder,
    private readonly recentMusic: IRecentSuggestions,
  ) {}

  async execute(input: ContentStrategyInput, progress: ProgressReporter = silentReporter): Promise<ContentStrategyResult> {
    progress.step('Generando estrategia: música, copy y textos', 88);
    const visualDescription = input.frameAnalyses
      .filter((a) => a.description)
      .map((a) => `[${formatTime(a.timestamp)}] ${a.description}`)
      .join('\n');

    const region = (await this.trends.get())?.region;
    const music = await this.music.candidates({
      country: region || undefined,
      musicStyle: input.musicStyle,
      themeText: `${input.context ?? ''} ${visualDescription} ${input.transcript?.text.slice(0, 500) ?? ''}`,
      language: input.language,
      limit: 40,
    });
    // Variedad: no se vuelven a ofrecer las canciones sugeridas en los últimos trabajos.
    const candidates = withoutRecent(music.songs, await this.recentMusic.recent()).slice(0, 30);
    if (music.note) progress.warn(music.note);
    const trending = await loadTrendingHashtags(this.trends);
    const relevant = relevantHashtags(trending.hashtags, `${input.context ?? ''} ${visualDescription} ${input.transcript?.text.slice(0, 500) ?? ''}`);
    const strategy = await this.ai.generateStrategy({
      mediaKind: 'video',
      platform: input.platform,
      visualDescription: visualDescription || 'Video sin descripción disponible.',
      language: input.language,
      tone: input.tone,
      context: input.context,
      musicStyle: input.musicStyle,
      musicCandidates: candidates,
      trendingHashtags: relevant,
      trendsRegion: trending.region,
      transcript: input.transcript,
      durationSeconds: input.durationSeconds,
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

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}
