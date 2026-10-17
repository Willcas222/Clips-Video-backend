import crypto from 'node:crypto';
import type { Job } from '../../domain/entities/Job';
import type { MediaKind } from '../../domain/entities/Media';
import type { IJobRepository } from '../../domain/ports/IJobRepository';

/** Repositorio de trabajos en memoria (se pierde al reiniciar; suficiente para uso local). */
export class InMemoryJobRepository implements IJobRepository {
  private readonly jobs = new Map<string, Job>();

  constructor(private readonly maxJobs = 200) {}

  create(kind: MediaKind): Job {
    const now = new Date().toISOString();
    const job: Job = {
      id: crypto.randomUUID(),
      kind,
      status: 'queued',
      step: 'En cola',
      progress: 0,
      warnings: [],
      createdAt: now,
      updatedAt: now,
    };
    this.jobs.set(job.id, job);
    this.evictOld();
    return job;
  }

  /** Reinserta un trabajo existente (al cargar el historial guardado). */
  restore(job: Job): void {
    this.jobs.set(job.id, job);
    this.evictOld();
  }

  get(id: string): Job | undefined {
    return this.jobs.get(id);
  }

  update(id: string, patch: Partial<Omit<Job, 'id' | 'createdAt'>>): Job | undefined {
    const current = this.jobs.get(id);
    if (!current) return undefined;
    const next: Job = { ...current, ...patch, updatedAt: new Date().toISOString() };
    this.jobs.set(id, next);
    return next;
  }

  list(limit = 50): Job[] {
    return [...this.jobs.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, limit);
  }

  private evictOld() {
    while (this.jobs.size > this.maxJobs) {
      const oldest = this.jobs.keys().next().value;
      if (oldest === undefined) break;
      this.jobs.delete(oldest);
    }
  }
}
