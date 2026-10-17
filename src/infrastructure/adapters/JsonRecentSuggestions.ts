import fs from 'node:fs/promises';
import path from 'node:path';
import { songKey, type IRecentSuggestions } from '../../domain/ports/IRecentSuggestions';

/** Guarda las canciones sugeridas en los últimos trabajos (storage/recent-music.json). */
export class JsonRecentSuggestions implements IRecentSuggestions {
  constructor(
    private readonly filePath: string,
    /** Cuántos trabajos recientes se recuerdan (3 trabajos × 3 canciones). */
    private readonly keepJobs = 3,
  ) {}

  async recent(): Promise<string[]> {
    return (await this.read()).flat();
  }

  async remember(songs: Array<{ title: string; artist: string }>): Promise<void> {
    const jobs = await this.read();
    jobs.unshift(songs.map((s) => songKey(s.title, s.artist)));
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    await fs.writeFile(this.filePath, JSON.stringify(jobs.slice(0, this.keepJobs)), 'utf8');
  }

  private async read(): Promise<string[][]> {
    try {
      return JSON.parse(await fs.readFile(this.filePath, 'utf8')) as string[][];
    } catch {
      return [];
    }
  }
}
