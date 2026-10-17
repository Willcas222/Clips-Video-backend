import type { Request, Response } from 'express';
import type { OllamaAdapter } from '../../adapters/OllamaAdapter';
import type { PythonWorkerAdapter } from '../../adapters/PythonWorkerAdapter';
import type { FFmpegAdapter } from '../../adapters/FFmpegAdapter';

interface HealthDeps {
  ollama: OllamaAdapter;
  python: PythonWorkerAdapter;
  ffmpeg: FFmpegAdapter;
  models: { text: string; vision: string };
}

interface HealthReport {
  ok: boolean;
  services: Record<string, { ok: boolean; [key: string]: unknown }>;
}

/** Las dependencias locales no cambian cada segundo: evita lanzar procesos en cada consulta. */
const CACHE_MS = 30_000;

/** GET /api/v1/health: estado de cada dependencia local (Ollama, Python, FFmpeg). */
export class HealthController {
  private cached?: { at: number; report: HealthReport };
  private inFlight?: Promise<HealthReport>;

  constructor(private readonly deps: HealthDeps) {}

  check = async (req: Request, res: Response) => {
    const fresh = req.query.fresh === '1';
    let report: HealthReport;
    if (!fresh && this.cached && Date.now() - this.cached.at < CACHE_MS) {
      report = this.cached.report;
    } else {
      // Peticiones simultáneas comparten la misma comprobación.
      this.inFlight ??= this.compute().finally(() => (this.inFlight = undefined));
      report = await this.inFlight;
      this.cached = { at: Date.now(), report };
    }
    res.status(report.ok ? 200 : 503).json(report);
  };

  private async compute(): Promise<HealthReport> {
    const [ollama, python, ffmpeg] = await Promise.all([this.checkOllama(), this.deps.python.healthCheck(), this.deps.ffmpeg.isAvailable()]);
    return {
      ok: ollama.ok && python.ok && ffmpeg,
      services: { ollama, python, ffmpeg: { ok: ffmpeg } },
    };
  }

  private async checkOllama() {
    try {
      const installed = await this.deps.ollama.listModels();
      const has = (name: string) => installed.some((m) => m === name || m === `${name}:latest`);
      const missing = [this.deps.models.text, this.deps.models.vision].filter((m) => !has(m));
      return {
        ok: missing.length === 0,
        models: this.deps.models,
        detail: missing.length ? `Faltan modelos: ${missing.join(', ')} (ollama pull <modelo>)` : 'OK',
      };
    } catch (err) {
      return { ok: false, models: this.deps.models, detail: `Ollama no responde: ${(err as Error).message}` };
    }
  }
}
