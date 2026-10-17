import fs from 'node:fs/promises';
import path from 'node:path';
import type { ITikTokTrendsStore, OfficialTrends } from '../../domain/ports/ITikTokTrendsStore';

/** Guarda la lista oficial de tendencias en data/tiktok-trends.json (sobrevive a reinicios). */
export class JsonTikTokTrendsStore implements ITikTokTrendsStore {
  constructor(private readonly filePath: string) {}

  async get(): Promise<OfficialTrends | null> {
    try {
      return JSON.parse(await fs.readFile(this.filePath, 'utf8')) as OfficialTrends;
    } catch {
      return null;
    }
  }

  async save(trends: OfficialTrends): Promise<void> {
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    const tmp = `${this.filePath}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(trends, null, 2), 'utf8');
    await fs.rename(tmp, this.filePath);
  }

  async clear(): Promise<void> {
    await fs.rm(this.filePath, { force: true });
  }
}
