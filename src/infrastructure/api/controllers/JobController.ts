import type { Request, Response } from 'express';
import { AppError } from '../../../domain/errors/AppError';
import type { IJobRepository } from '../../../domain/ports/IJobRepository';

export class JobController {
  constructor(private readonly jobs: IJobRepository) {}

  /** GET /api/v1/jobs/:id */
  getById = (req: Request, res: Response) => {
    const job = this.jobs.get(String(req.params.id));
    if (!job) throw AppError.notFound('Trabajo no encontrado (¿se reinició el servidor?)');
    res.json(job);
  };

  /** GET /api/v1/jobs */
  list = (_req: Request, res: Response) => {
    res.json(this.jobs.list(30).map(({ result, ...summary }) => ({ ...summary, hasResult: Boolean(result) })));
  };
}
