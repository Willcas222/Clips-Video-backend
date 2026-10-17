import fs from 'node:fs';
import path from 'node:path';
import type { Job } from '../../domain/entities/Job';
import type { MediaKind } from '../../domain/entities/Media';
import type { IJobRepository } from '../../domain/ports/IJobRepository';
import { InMemoryJobRepository } from './InMemoryJobRepository';

/**
 * Historial de trabajos persistido en un JSON: sobrevive a reinicios del servidor.
 * Lee/escribe en memoria y guarda en disco de forma diferida (agrupando cambios seguidos).
 */
export class FileJobRepository implements IJobRepository {
  private readonly memory: InMemoryJobRepository;
  private saveTimer?: NodeJS.Timeout;

  constructor(
    private readonly filePath: string,
    maxJobs = 200,
  ) {
    this.memory = new InMemoryJobRepository(maxJobs);
    this.load();
    // Garantiza el último guardado al cerrar el proceso.
    process.once('exit', () => this.flush());
  }

  create(kind: MediaKind): Job {
    const job = this.memory.create(kind);
    this.scheduleSave();
    return job;
  }

  get(id: string): Job | undefined {
    return this.memory.get(id);
  }

  update(id: string, patch: Partial<Omit<Job, 'id' | 'createdAt'>>): Job | undefined {
    const job = this.memory.update(id, patch);
    // El progreso cambia muy seguido: se guarda agrupado; los estados finales, al momento.
    if (patch.status === 'completed' || patch.status === 'failed') this.flush();
    else this.scheduleSave();
    return job;
  }

  list(limit?: number): Job[] {
    return this.memory.list(limit);
  }

  private load() {
    try {
      const jobs = JSON.parse(fs.readFileSync(this.filePath, 'utf8')) as Job[];
      // Orden cronológico para que el repositorio en memoria conserve los más recientes.
      for (const job of jobs.sort((a, b) => a.createdAt.localeCompare(b.createdAt))) {
        const interrupted = job.status === 'processing' || job.status === 'queued';
        this.memory.restore(
          interrupted ? { ...job, status: 'failed', step: 'Interrumpido', error: 'El servidor se reinició mientras se procesaba. Vuelve a enviarlo.' } : job,
        );
      }
      console.log(`[jobs] historial cargado: ${jobs.length} trabajos`);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') console.warn('[jobs] no se pudo leer el historial:', (err as Error).message);
    }
  }

  private scheduleSave() {
    if (this.saveTimer) return;
    this.saveTimer = setTimeout(() => this.flush(), 1500);
  }

  private flush() {
    clearTimeout(this.saveTimer);
    this.saveTimer = undefined;
    try {
      fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
      const tmp = `${this.filePath}.tmp`;
      fs.writeFileSync(tmp, JSON.stringify(this.memory.list(Number.MAX_SAFE_INTEGER)), 'utf8');
      fs.renameSync(tmp, this.filePath);
    } catch (err) {
      console.warn('[jobs] no se pudo guardar el historial:', (err as Error).message);
    }
  }
}
