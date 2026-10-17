import path from 'node:path';
import { AppError } from '../../domain/errors/AppError';
import type { BoundingBox } from '../../domain/entities/Media';
import type { ImageQualityReport, QualityScore, Transcript } from '../../domain/entities/ProcessedContent';
import type { IImageProcessor, ITranscriber, IVoiceIsolator } from '../../domain/ports/IMediaProcessor';
import { runProcess } from './shared/runProcess';

interface WorkerResponse {
  ok: boolean;
  error?: string;
  [key: string]: unknown;
}

export interface PythonWorkerConfig {
  pythonBin: string;
  workersDir: string;
  timeoutMs: number;
  whisperModel: string;
  /** Binario de FFmpeg que whisper_sync.py usa para decodificar el audio. */
  ffmpegPath: string;
}

/**
 * Ejecuta los scripts de /ai-workers como procesos hijos.
 * Protocolo: el worker escribe logs en stderr y, al final, UNA línea JSON en stdout
 * con `{ "ok": true, ... }` o `{ "ok": false, "error": "..." }`.
 */
export class PythonWorkerAdapter implements IImageProcessor, ITranscriber, IVoiceIsolator {
  constructor(private readonly cfg: PythonWorkerConfig) {}

  async inpaint({ inputPath, outputPath, maskPath, box, reflections = true }: { inputPath: string; outputPath: string; maskPath?: string; box?: BoundingBox; reflections?: boolean }) {
    if (!maskPath && !box) throw AppError.validation('Inpainting requiere una máscara o un bounding box');
    const args = ['--input', inputPath, '--output', outputPath];
    if (maskPath) args.push('--mask', maskPath);
    if (box) args.push('--bbox', [box.x1, box.y1, box.x2, box.y2].map(Math.round).join(','));
    if (!reflections) args.push('--no-reflections');
    const res = await this.run('inpainting.py', args);
    return { engine: String(res.engine ?? 'unknown'), reflectionRemoved: res.reflectionRemoved === true };
  }

  async enhance({ inputPath, outputPath }: { inputPath: string; outputPath?: string }): Promise<ImageQualityReport> {
    const args = ['--input', inputPath];
    if (outputPath) args.push('--output', outputPath);
    else args.push('--analyze-only');
    const res = await this.run('enhance.py', args);
    const strings = (v: unknown) => (Array.isArray(v) ? v.map(String) : []);
    return {
      issues: strings(res.issues),
      adjustments: strings(res.adjustments),
      metrics: (res.metrics as Record<string, number>) ?? {},
    };
  }

  async proFinish({ inputPath, outputPath, referencePath, blur, straighten }: { inputPath: string; outputPath: string; referencePath?: string; blur: 'off' | 'suave' | 'medio' | 'fuerte'; straighten: boolean }) {
    const args = ['--input', inputPath, '--output', outputPath];
    if (referencePath) args.push('--reference', referencePath);
    if (blur === 'off') args.push('--no-blur');
    else args.push('--blur', blur);
    if (!straighten) args.push('--no-straighten');
    const res = await this.run('pro_finish.py', args);
    const strings = (v: unknown) => (Array.isArray(v) ? v.map(String) : []);
    return {
      adjustments: strings(res.adjustments),
      notes: strings(res.notes),
      scoreBefore: res.scoreBefore as QualityScore,
      scoreAfter: res.scoreAfter as QualityScore,
    };
  }

  async upscale({ inputPath, outputPath, factor, onProgress }: { inputPath: string; outputPath: string; factor: 2 | 4; onProgress?: (percent: number) => void }) {
    const res = await this.run('upscaler.py', ['--input', inputPath, '--output', outputPath, '--scale', String(factor)], this.cfg.timeoutMs, onProgress);
    return { engine: String(res.engine ?? 'unknown') };
  }

  async isolateVoice({ inputPath, outputPath }: { inputPath: string; outputPath: string }) {
    const res = await this.run('voice_isolate.py', ['--input', inputPath, '--output', outputPath, '--ffmpeg', this.cfg.ffmpegPath]);
    return { engine: String(res.engine ?? 'demucs'), hasVoice: res.hasVoice !== false };
  }

  async transcribe({ inputPath, language }: { inputPath: string; language?: string }): Promise<Transcript> {
    const args = ['--input', inputPath, '--model', this.cfg.whisperModel, '--ffmpeg', this.cfg.ffmpegPath];
    if (language) args.push('--language', language);
    const res = await this.run('whisper_sync.py', args);
    return {
      language: String(res.language ?? language ?? ''),
      text: String(res.text ?? ''),
      segments: Array.isArray(res.segments) ? (res.segments as Transcript['segments']) : [],
    };
  }

  /** Comprueba que el intérprete arranca y que las dependencias básicas están instaladas. */
  async healthCheck(): Promise<{ ok: boolean; detail: string }> {
    try {
      const res = await this.run('healthcheck.py', [], 20_000);
      return { ok: true, detail: JSON.stringify(res.packages ?? {}) };
    } catch (err) {
      return { ok: false, detail: (err as Error).message };
    }
  }

  private async run(script: string, args: string[], timeoutMs = this.cfg.timeoutMs, onProgress?: (percent: number) => void): Promise<WorkerResponse> {
    const scriptPath = path.join(this.cfg.workersDir, script);
    let stdout = '';
    let lastLogged = -1;
    try {
      ({ stdout } = await runProcess(this.cfg.pythonBin, [scriptPath, ...args], {
        cwd: this.cfg.workersDir,
        timeoutMs,
        onStderrLine: (line) => {
          const progress = /^PROGRESS (\d+)$/.exec(line.trim());
          if (!progress) return console.log(`[py:${script}] ${line}`);
          const percent = Number(progress[1]);
          onProgress?.(percent);
          // En consola solo cada 10% para no inundarla.
          if (percent >= lastLogged + 10 || percent === 100) {
            lastLogged = percent;
            console.log(`[py:${script}] ${percent}%`);
          }
        },
      }));
    } catch (err) {
      // Aun con código de salida != 0 el worker puede haber explicado el error en stdout.
      const partial = (err as { result?: { stdout?: string } }).result?.stdout;
      const parsed = partial ? parseLastJsonLine(partial) : undefined;
      throw new AppError(`Worker ${script} falló: ${parsed?.error ?? (err as Error).message}`, 'WORKER_ERROR', 500);
    }

    const response = parseLastJsonLine(stdout);
    if (!response) throw new AppError(`Worker ${script} no devolvió JSON válido`, 'WORKER_ERROR', 500, stdout.slice(-500));
    if (!response.ok) throw new AppError(`Worker ${script}: ${response.error ?? 'error desconocido'}`, 'WORKER_ERROR', 500);
    return response;
  }
}

function parseLastJsonLine(stdout: string): WorkerResponse | undefined {
  const lines = stdout.trim().split(/\r?\n/).reverse();
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{')) continue;
    try {
      return JSON.parse(trimmed) as WorkerResponse;
    } catch {
      /* línea no JSON, seguir buscando */
    }
  }
  return undefined;
}
