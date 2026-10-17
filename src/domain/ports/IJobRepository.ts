import type { Job } from '../entities/Job';
import type { MediaKind } from '../entities/Media';

export interface IJobRepository {
  create(kind: MediaKind): Job;
  get(id: string): Job | undefined;
  update(id: string, patch: Partial<Omit<Job, 'id' | 'createdAt'>>): Job | undefined;
  list(limit?: number): Job[];
}
