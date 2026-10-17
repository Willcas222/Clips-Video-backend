import type { Job, ProgressReporter } from '../../domain/entities/Job';
import type { MediaKind } from '../../domain/entities/Media';
import type { ProcessedContent } from '../../domain/entities/ProcessedContent';
import type { IJobRepository } from '../../domain/ports/IJobRepository';

type Task = (progress: ProgressReporter) => Promise<ProcessedContent>;

/**
 * Ejecuta pipelines en segundo plano con concurrencia limitada (por defecto 1:
 * la IA local satura la CPU/GPU y procesar en paralelo solo lo haría todo más lento).
 */
export class JobRunner {
  private readonly queue: Array<() => Promise<void>> = [];
  private running = 0;

  constructor(
    private readonly jobs: IJobRepository,
    private readonly concurrency = 1,
  ) {}

  enqueue(kind: MediaKind, task: Task): Job {
    const job = this.jobs.create(kind);
    this.queue.push(() => this.run(job.id, task));
    this.drain();
    return job;
  }

  private drain() {
    while (this.running < this.concurrency && this.queue.length > 0) {
      const next = this.queue.shift()!;
      this.running++;
      next().finally(() => {
        this.running--;
        this.drain();
      });
    }
  }

  private async run(jobId: string, task: Task) {
    const progress: ProgressReporter = {
      step: (label, value) => {
        this.jobs.update(jobId, { step: label, progress: Math.max(0, Math.min(99, value)) });
      },
      warn: (message) => {
        const job = this.jobs.get(jobId);
        if (job) this.jobs.update(jobId, { warnings: [...job.warnings, message] });
      },
    };

    this.jobs.update(jobId, { status: 'processing', step: 'Iniciando', progress: 1 });
    const started = Date.now();
    try {
      const result = await task(progress);
      this.jobs.update(jobId, { status: 'completed', step: 'Finalizado', progress: 100, result });
      console.log(`[job ${jobId.slice(0, 8)}] completado en ${((Date.now() - started) / 1000).toFixed(1)} s`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.jobs.update(jobId, { status: 'failed', step: 'Error', error: message });
      console.error(`[job ${jobId.slice(0, 8)}] falló:`, message);
    }
  }
}
