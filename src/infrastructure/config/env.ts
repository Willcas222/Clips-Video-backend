import path from 'node:path';
import fs from 'node:fs';
import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config({ quiet: true });

const ROOT_DIR = path.resolve(__dirname, '..', '..', '..');

const emptyToUndefined = (value: unknown) => (value === '' ? undefined : value);

const envSchema = z.object({
  PORT: z.coerce.number().int().positive().default(3001),
  CORS_ORIGIN: z.string().default('http://localhost:3000,http://127.0.0.1:3000'),
  PUBLIC_BASE_URL: z.preprocess(emptyToUndefined, z.string().url().optional()),
  MAX_UPLOAD_MB: z.coerce.number().positive().default(1024),

  OLLAMA_URL: z.string().url().default('http://localhost:11434'),
  OLLAMA_TEXT_MODEL: z.string().default('qwen2.5:7b'),
  OLLAMA_VISION_MODEL: z.string().default('qwen2.5vl:3b'),
  OLLAMA_TIMEOUT_MS: z.coerce.number().int().positive().default(600_000),
  // '0' descarga el modelo tras cada llamada: evita tener visión + texto + Whisper en RAM a la vez.
  OLLAMA_KEEP_ALIVE: z.string().default('0'),

  PYTHON_BIN: z.preprocess(emptyToUndefined, z.string().optional()),
  WORKER_TIMEOUT_MS: z.coerce.number().int().positive().default(1_800_000),
  WHISPER_MODEL: z.string().default('small'),
  // 'on': consulta tendencias de TikTok en internet (solo títulos de canciones; nunca tus archivos).
  MUSIC_TRENDS: z.enum(['on', 'off']).default('on'),
  // País cuyo ranking de música se usa si no se indicó otro en “Tendencias oficiales”.
  MUSIC_COUNTRY: z.string().default('colombia'),

  FFMPEG_PATH: z.preprocess(emptyToUndefined, z.string().optional()),
  FFPROBE_PATH: z.preprocess(emptyToUndefined, z.string().optional()),
});

const parsed = envSchema.safeParse(process.env);
if (!parsed.success) {
  console.error('Variables de entorno inválidas:', z.treeifyError(parsed.error));
  process.exit(1);
}
const env = parsed.data;

function resolvePythonBin(): string {
  if (env.PYTHON_BIN) return env.PYTHON_BIN;
  const venvPython =
    process.platform === 'win32'
      ? path.join(ROOT_DIR, 'ai-workers', '.venv', 'Scripts', 'python.exe')
      : path.join(ROOT_DIR, 'ai-workers', '.venv', 'bin', 'python');
  return fs.existsSync(venvPython) ? venvPython : 'python';
}

function resolveFfmpeg(): string {
  if (env.FFMPEG_PATH) return env.FFMPEG_PATH;
  const bundled = require('ffmpeg-static') as string | null;
  return bundled ?? 'ffmpeg';
}

function resolveFfprobe(): string {
  if (env.FFPROBE_PATH) return env.FFPROBE_PATH;
  const bundled = require('ffprobe-static') as { path?: string };
  return bundled.path ?? 'ffprobe';
}

export const config = {
  port: env.PORT,
  corsOrigin: env.CORS_ORIGIN.split(',').map((o) => o.trim()),
  publicBaseUrl: env.PUBLIC_BASE_URL ?? `http://localhost:${env.PORT}`,
  maxUploadBytes: env.MAX_UPLOAD_MB * 1024 * 1024,
  musicTrends: env.MUSIC_TRENDS === 'on',
  musicCountry: env.MUSIC_COUNTRY.toLowerCase(),
  paths: {
    root: ROOT_DIR,
    inputs: path.join(ROOT_DIR, 'storage', 'inputs'),
    outputs: path.join(ROOT_DIR, 'storage', 'outputs'),
    temp: path.join(ROOT_DIR, 'storage', 'temp'),
    workers: path.join(ROOT_DIR, 'ai-workers'),
  },
  ollama: {
    url: env.OLLAMA_URL,
    textModel: env.OLLAMA_TEXT_MODEL,
    visionModel: env.OLLAMA_VISION_MODEL,
    timeoutMs: env.OLLAMA_TIMEOUT_MS,
    keepAlive: env.OLLAMA_KEEP_ALIVE,
  },
  python: {
    bin: resolvePythonBin(),
    timeoutMs: env.WORKER_TIMEOUT_MS,
    whisperModel: env.WHISPER_MODEL,
  },
  ffmpeg: {
    ffmpegPath: resolveFfmpeg(),
    ffprobePath: resolveFfprobe(),
  },
} as const;

export type AppConfig = typeof config;
