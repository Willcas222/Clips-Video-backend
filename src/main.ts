import { config } from './infrastructure/config/env';
import { buildContainer } from './infrastructure/config/container';
import { createApp } from './infrastructure/api/app';

async function bootstrap() {
  const { storage, controllers } = buildContainer();
  await storage.init();

  const app = createApp({
    ...controllers,
    inputsDir: config.paths.inputs,
    outputsDir: config.paths.outputs,
    maxUploadBytes: config.maxUploadBytes,
    corsOrigin: config.corsOrigin,
  });

  const server = app.listen(config.port, () => {
    console.log(`API lista en ${config.publicBaseUrl}  (health: ${config.publicBaseUrl}/api/v1/health)`);
    console.log(`Ollama: ${config.ollama.url}  texto=${config.ollama.textModel}  visión=${config.ollama.visionModel}`);
    console.log(`Python: ${config.python.bin}`);
  });
  // Subidas grandes por red local lenta: no cortar la petición a los 5 min por defecto.
  server.requestTimeout = 0;

  const shutdown = () => server.close(() => process.exit(0));
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

bootstrap().catch((err) => {
  console.error('No se pudo iniciar el servidor:', err);
  process.exit(1);
});
