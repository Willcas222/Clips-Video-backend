import fs from 'node:fs/promises';
import type { Request, Response } from 'express';
import { z } from 'zod';
import { AppError } from '../../../domain/errors/AppError';
import type { Media } from '../../../domain/entities/Media';
import type { ImageContentPipeline } from '../../../application/pipelines/ImageContentPipeline';
import type { VideoContentPipeline } from '../../../application/pipelines/VideoContentPipeline';

// multipart/form-data solo trae strings: se convierten aquí.
const booleanish = z
  .union([z.boolean(), z.enum(['true', 'false', '1', '0', 'on', 'off'])])
  .transform((v) => v === true || v === 'true' || v === '1' || v === 'on');

const common = {
  platform: z.enum(['tiktok', 'reels', 'stories', 'post']).default('tiktok'),
  language: z.string().min(2).max(5).default('es'),
  tone: z.string().trim().max(120).optional().transform((v) => v || undefined),
  context: z.string().trim().max(600).optional().transform((v) => v || undefined),
  musicStyle: z.string().trim().max(120).optional().transform((v) => v || undefined),
};

const imageBodySchema = z.object({
  ...common,
  removeObject: z.string().trim().max(200).optional().transform((v) => v || undefined),
  upscale: booleanish.default(false),
  removeReflections: booleanish.default(true),
  enhance: booleanish.default(true),
  proFinish: booleanish.default(true),
  portraitBlur: z.enum(['off', 'suave', 'medio', 'fuerte']).default('suave'),
  straighten: booleanish.default(true),
  upscaleFactor: z.coerce.number().pipe(z.union([z.literal(2), z.literal(4)])).default(2),
});

const videoBodySchema = z.object({
  ...common,
  maxClips: z.coerce.number().int().min(1).max(10).default(3),
  clipDuration: z.coerce.number().min(3).max(180).default(15),
  verticalCrop: booleanish.default(true),
  transcribe: booleanish.default(false),
  audioMode: z.enum(['mute', 'voice', 'original']).default('mute'),
});

function toMedia(file: Express.Multer.File, kind: Media['kind']): Media {
  return {
    id: file.filename,
    kind,
    originalName: file.originalname,
    mimeType: file.mimetype,
    sizeBytes: file.size,
    path: file.path,
  };
}

async function discard(files: Array<Express.Multer.File | undefined>) {
  await Promise.all(files.filter(Boolean).map((f) => fs.rm(f!.path, { force: true })));
}

export class ContentController {
  constructor(
    private readonly imagePipeline: ImageContentPipeline,
    private readonly videoPipeline: VideoContentPipeline,
  ) {}

  /** POST /api/v1/content/image */
  processImage = async (req: Request, res: Response) => {
    const files = req.files as Record<string, Express.Multer.File[]> | undefined;
    const image = files?.file?.[0];
    const mask = files?.mask?.[0];
    if (!image) {
      await discard([mask]);
      throw AppError.validation('Falta el archivo de imagen (campo "file")');
    }

    const parsed = imageBodySchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      await discard([image, mask]);
      throw parsed.error;
    }
    const job = this.imagePipeline.start(toMedia(image, 'image'), { ...parsed.data, maskPath: mask?.path });
    res.status(202).json({ jobId: job.id, status: job.status, statusUrl: `/api/v1/jobs/${job.id}` });
  };

  /** POST /api/v1/content/video */
  processVideo = async (req: Request, res: Response) => {
    const video = req.file;
    if (!video) throw AppError.validation('Falta el archivo de video (campo "file")');

    const parsed = videoBodySchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      await discard([video]);
      throw parsed.error;
    }
    const job = this.videoPipeline.start(toMedia(video, 'video'), parsed.data);
    res.status(202).json({ jobId: job.id, status: job.status, statusUrl: `/api/v1/jobs/${job.id}` });
  };
}
