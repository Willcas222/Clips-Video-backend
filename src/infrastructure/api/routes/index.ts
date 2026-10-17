import { Router } from 'express';
import type { ContentController } from '../controllers/ContentController';
import type { JobController } from '../controllers/JobController';
import type { HealthController } from '../controllers/HealthController';
import type { TrendsController } from '../controllers/TrendsController';
import { imageUpload, videoUpload } from '../middlewares/upload';

export interface RouteDeps {
  content: ContentController;
  jobs: JobController;
  health: HealthController;
  trends: TrendsController;
  inputsDir: string;
  maxUploadBytes: number;
}

export function createApiRouter(deps: RouteDeps): Router {
  const router = Router();

  router.get('/health', deps.health.check);

  router.post('/content/image', imageUpload(deps.inputsDir, deps.maxUploadBytes), deps.content.processImage);
  router.post('/content/video', videoUpload(deps.inputsDir, deps.maxUploadBytes), deps.content.processVideo);

  router.get('/music/tiktok-trends', deps.trends.get);
  router.put('/music/tiktok-trends', deps.trends.import);
  router.delete('/music/tiktok-trends', deps.trends.clear);

  router.get('/jobs', deps.jobs.list);
  router.get('/jobs/:id', deps.jobs.getById);

  return router;
}
