import { config } from './env';
import { FFmpegAdapter } from '../adapters/FFmpegAdapter';
import { OllamaAdapter } from '../adapters/OllamaAdapter';
import { PythonWorkerAdapter } from '../adapters/PythonWorkerAdapter';
import { LocalStorageService } from '../adapters/LocalStorageService';
import { FileJobRepository } from '../adapters/FileJobRepository';
import { JsonMusicCatalog } from '../adapters/JsonMusicCatalog';
import { TrendingMusicCatalog } from '../adapters/TrendingMusicCatalog';
import { LrclibHookFinder } from '../adapters/LrclibHookFinder';
import { JsonRecentSuggestions } from '../adapters/JsonRecentSuggestions';
import { OfficialTrendsCatalog } from '../adapters/OfficialTrendsCatalog';
import { JsonTikTokTrendsStore } from '../adapters/JsonTikTokTrendsStore';
import { ImportTikTokTrendsUseCase } from '../../application/use-cases/ImportTikTokTrendsUseCase';
import { TrendsController } from '../api/controllers/TrendsController';
import path from 'node:path';
import { ProcessImageUseCase } from '../../application/use-cases/ProcessImageUseCase';
import { GenerateCopyUseCase } from '../../application/use-cases/GenerateCopyUseCase';
import { ProcessVideoUseCase } from '../../application/use-cases/ProcessVideoUseCase';
import { ContentStrategyUseCase } from '../../application/use-cases/ContentStrategyUseCase';
import { JobRunner } from '../../application/services/JobRunner';
import { ImageContentPipeline } from '../../application/pipelines/ImageContentPipeline';
import { VideoContentPipeline } from '../../application/pipelines/VideoContentPipeline';
import { ContentController } from '../api/controllers/ContentController';
import { JobController } from '../api/controllers/JobController';
import { HealthController } from '../api/controllers/HealthController';

/** Raíz de composición: el único lugar donde se conectan adaptadores concretos con los puertos. */
export function buildContainer() {
  const storage = new LocalStorageService(
    { inputs: config.paths.inputs, outputs: config.paths.outputs, temp: config.paths.temp },
    config.publicBaseUrl,
  );
  const ffmpeg = new FFmpegAdapter(config.ffmpeg.ffmpegPath, config.ffmpeg.ffprobePath);
  const python = new PythonWorkerAdapter({
    pythonBin: config.python.bin,
    workersDir: config.paths.workers,
    timeoutMs: config.python.timeoutMs,
    whisperModel: config.python.whisperModel,
    ffmpegPath: config.ffmpeg.ffmpegPath,
  });
  const ollama = new OllamaAdapter({ ...config.ollama, tempDir: config.paths.temp }, ffmpeg);
  const localMusic = new JsonMusicCatalog(path.join(config.paths.root, 'data', 'music-catalog.json'));
  // Tendencias de TikTok en vivo (Deezer) con el catálogo local como respaldo sin conexión.
  const deezer = config.musicTrends ? new TrendingMusicCatalog(localMusic, config.musicCountry) : null;
  // Ranking oficial de TikTok pegado por el usuario (prioridad) + Deezer/catálogo local.
  const officialTrends = new JsonTikTokTrendsStore(path.join(config.paths.root, 'data', 'tiktok-trends.json'));
  const music = new OfficialTrendsCatalog(officialTrends, deezer ?? localMusic);
  const importTrends = new ImportTikTokTrendsUseCase(officialTrends, deezer ?? { lookup: async () => null });
  // Historial persistido en disco: sobrevive a reinicios del servidor.
  // Tramo exacto del coro de cada canción sugerida (letras sincronizadas de LRCLIB).
  const hooks = config.musicTrends ? new LrclibHookFinder() : { findHook: async () => null };
  const recentMusic = new JsonRecentSuggestions(path.join(config.paths.root, 'storage', 'recent-music.json'));
  const jobs = new FileJobRepository(path.join(config.paths.root, 'storage', 'jobs.json'));
  const runner = new JobRunner(jobs);

  const imagePipeline = new ImageContentPipeline(
    new ProcessImageUseCase(python, ollama, storage),
    new GenerateCopyUseCase(ollama, music, officialTrends, hooks, recentMusic),
    storage,
    runner,
  );
  const videoPipeline = new VideoContentPipeline(
    new ProcessVideoUseCase(ffmpeg, python, python, ollama, storage),
    new ContentStrategyUseCase(ollama, music, officialTrends, hooks, recentMusic),
    storage,
    runner,
  );

  return {
    storage,
    controllers: {
      content: new ContentController(imagePipeline, videoPipeline),
      jobs: new JobController(jobs),
      trends: new TrendsController(officialTrends, importTrends),
      health: new HealthController({ ollama, python, ffmpeg, models: { text: config.ollama.textModel, vision: config.ollama.visionModel } }),
    },
  };
}
