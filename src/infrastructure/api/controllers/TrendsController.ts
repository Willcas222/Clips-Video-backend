import type { Request, Response } from 'express';
import { z } from 'zod';
import type { ITikTokTrendsStore } from '../../../domain/ports/ITikTokTrendsStore';
import type { ImportTikTokTrendsUseCase } from '../../../application/use-cases/ImportTikTokTrendsUseCase';

const importSchema = z.object({
  text: z.string().min(3).max(50_000),
  region: z.string().trim().min(2).max(60).default('Colombia'),
});

export class TrendsController {
  constructor(
    private readonly store: ITikTokTrendsStore,
    private readonly importTrends: ImportTikTokTrendsUseCase,
  ) {}

  /** GET /api/v1/music/tiktok-trends */
  get = async (_req: Request, res: Response) => {
    res.json((await this.store.get()) ?? { region: '', updatedAt: null, songs: [] });
  };

  /** PUT /api/v1/music/tiktok-trends  { text, region } */
  import = async (req: Request, res: Response) => {
    const input = importSchema.parse(req.body ?? {});
    res.json(await this.importTrends.execute(input));
  };

  /** DELETE /api/v1/music/tiktok-trends */
  clear = async (_req: Request, res: Response) => {
    await this.store.clear();
    res.status(204).end();
  };
}
