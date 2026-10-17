import type { Media, VideoProcessingOptions } from '../../domain/entities/Media';
import type { Job } from '../../domain/entities/Job';
import type { ProcessedVideoContent } from '../../domain/entities/ProcessedContent';
import type { IStorageService } from '../../domain/ports/IStorageService';
import type { ProcessVideoUseCase } from '../use-cases/ProcessVideoUseCase';
import type { ContentStrategyUseCase } from '../use-cases/ContentStrategyUseCase';
import type { JobRunner } from '../services/JobRunner';

/** Orquesta el Flujo B completo como trabajo asíncrono. */
export class VideoContentPipeline {
  constructor(
    private readonly processVideo: ProcessVideoUseCase,
    private readonly contentStrategy: ContentStrategyUseCase,
    private readonly storage: IStorageService,
    private readonly runner: JobRunner,
  ) {}

  start(media: Media, options: VideoProcessingOptions): Job {
    return this.runner.enqueue('video', async (progress): Promise<ProcessedVideoContent> => {
      const video = await this.processVideo.execute(media, options, progress);
      const { visualDescription, strategy } = await this.contentStrategy.execute(
        {
          platform: options.platform,
          language: options.language,
          tone: options.tone,
          context: options.context,
          musicStyle: options.musicStyle,
          durationSeconds: video.durationSeconds,
          frameAnalyses: video.frameAnalyses,
          transcript: video.transcript,
        },
        progress,
      );

      return {
        kind: 'video',
        originalUrl: this.storage.toPublicUrl(media.path),
        durationSeconds: video.durationSeconds,
        clips: video.clips,
        visualDescription,
        transcript: video.transcript,
        strategy,
      };
    });
  }
}
