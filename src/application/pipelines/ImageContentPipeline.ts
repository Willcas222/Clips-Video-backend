import type { ImageProcessingOptions, Media } from '../../domain/entities/Media';
import type { Job } from '../../domain/entities/Job';
import type { ProcessedImageContent } from '../../domain/entities/ProcessedContent';
import type { IStorageService } from '../../domain/ports/IStorageService';
import type { ProcessImageUseCase } from '../use-cases/ProcessImageUseCase';
import type { GenerateCopyUseCase } from '../use-cases/GenerateCopyUseCase';
import type { JobRunner } from '../services/JobRunner';

/** Orquesta el Flujo A completo como trabajo asíncrono. */
export class ImageContentPipeline {
  constructor(
    private readonly processImage: ProcessImageUseCase,
    private readonly generateCopy: GenerateCopyUseCase,
    private readonly storage: IStorageService,
    private readonly runner: JobRunner,
  ) {}

  start(media: Media, options: ImageProcessingOptions): Job {
    return this.runner.enqueue('image', async (progress): Promise<ProcessedImageContent> => {
      const { processedPath, appliedSteps, quality } = await this.processImage.execute(media, options, progress);
      const { visualDescription, strategy } = await this.generateCopy.execute(
        {
          imagePath: processedPath,
          platform: options.platform,
          language: options.language,
          tone: options.tone,
          context: options.context,
          musicStyle: options.musicStyle,
        },
        progress,
      );
      if (options.maskPath) await this.storage.remove(options.maskPath);

      return {
        kind: 'image',
        originalUrl: this.storage.toPublicUrl(media.path),
        processedUrl: this.storage.toPublicUrl(processedPath),
        appliedSteps,
        quality,
        visualDescription,
        strategy,
      };
    });
  }
}
