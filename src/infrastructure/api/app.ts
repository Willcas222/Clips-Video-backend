import path from 'node:path';
import express from 'express';
import { AppError } from '../../domain/errors/AppError';
import cors from 'cors';
import { createApiRouter, type RouteDeps } from './routes';
import { errorHandler, notFoundHandler } from './middlewares/errorHandler';

export interface AppOptions extends RouteDeps {
  corsOrigin: string[];
  outputsDir: string;
}

export function createApp(opts: AppOptions) {
  const app = express();
  app.disable('x-powered-by');
  app.use(cors({ origin: opts.corsOrigin }));
  app.use(express.json({ limit: '1mb' }));

  // Descarga directa (Content-Disposition: attachment): el atributo `download` de un enlace no
  // funciona entre orígenes distintos (frontend :3000, API :3001).
  const areas: Record<string, string> = { inputs: opts.inputsDir, outputs: opts.outputsDir };
  app.get('/files/download/:area/:name', (req, res, next) => {
    const dir = areas[req.params.area];
    const name = path.basename(req.params.name);
    if (!dir || name !== req.params.name) return next(AppError.notFound('Archivo no encontrado'));
    res.download(path.join(dir, name), name, (err) => {
      if (err && !res.headersSent) next(AppError.notFound('Archivo no encontrado'));
    });
  });

  // Archivos locales accesibles para previsualizar desde el frontend.
  const staticOpts = { fallthrough: false, maxAge: '1h' };
  app.use('/files/inputs', express.static(opts.inputsDir, staticOpts));
  app.use('/files/outputs', express.static(opts.outputsDir, staticOpts));

  app.get('/', (_req, res) => {
    res.json({ name: 'Asistente de contenido IA (local)', api: '/api/v1', health: '/api/v1/health' });
  });
  app.use('/api/v1', createApiRouter(opts));

  app.use(notFoundHandler);
  app.use(errorHandler);
  return app;
}
