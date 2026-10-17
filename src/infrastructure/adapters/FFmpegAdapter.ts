import path from 'node:path';
import fs from 'node:fs/promises';
import { AppError } from '../../domain/errors/AppError';
import type { KeyFrame } from '../../domain/entities/ProcessedContent';
import type { IVideoProcessor, VideoInfo } from '../../domain/ports/IMediaProcessor';
import { runProcess, ProcessError } from './shared/runProcess';

interface FfprobeOutput {
  format?: { duration?: string };
  streams?: Array<{ codec_type?: string; width?: number; height?: number; duration?: string; tags?: { rotate?: string }; side_data_list?: Array<{ rotation?: number }> }>;
}

export interface ImageDimensions {
  width: number;
  height: number;
}

/** Adaptador de video/imagen basado en el CLI de FFmpeg ejecutado como proceso hijo. */
export class FFmpegAdapter implements IVideoProcessor {
  constructor(
    private readonly ffmpegPath: string,
    private readonly ffprobePath: string,
    private readonly timeoutMs = 30 * 60 * 1000,
  ) {}

  async probe(inputPath: string): Promise<VideoInfo> {
    const data = await this.ffprobe(inputPath);
    const video = data.streams?.find((s) => s.codec_type === 'video');
    if (!video?.width || !video.height) {
      throw new AppError('El archivo no contiene una pista de video válida', 'UNSUPPORTED_MEDIA', 422);
    }
    const duration = Number(data.format?.duration ?? video.duration ?? 0);
    const rotation = Math.abs(Number(video.tags?.rotate ?? video.side_data_list?.find((d) => d.rotation !== undefined)?.rotation ?? 0));
    const rotated = rotation === 90 || rotation === 270;
    return {
      durationSeconds: Number.isFinite(duration) ? duration : 0,
      width: rotated ? video.height : video.width,
      height: rotated ? video.width : video.height,
      hasAudio: Boolean(data.streams?.some((s) => s.codec_type === 'audio')),
    };
  }

  async getImageDimensions(inputPath: string): Promise<ImageDimensions> {
    const data = await this.ffprobe(inputPath);
    const stream = data.streams?.find((s) => s.width && s.height);
    if (!stream?.width || !stream.height) {
      throw new AppError('No se pudieron leer las dimensiones de la imagen', 'UNSUPPORTED_MEDIA', 422);
    }
    return { width: stream.width, height: stream.height };
  }

  async extractKeyFrames({ inputPath, outputDir, count }: { inputPath: string; outputDir: string; count: number }): Promise<KeyFrame[]> {
    const { durationSeconds } = await this.probe(inputPath);
    if (durationSeconds <= 0) throw new AppError('No se pudo determinar la duración del video', 'UNSUPPORTED_MEDIA', 422);

    await fs.mkdir(outputDir, { recursive: true });
    // Muestreo uniforme evitando el primer y último 3% (suelen ser negros o transiciones).
    const margin = durationSeconds * 0.03;
    const usable = durationSeconds - margin * 2;
    const timestamps = Array.from({ length: count }, (_, i) => margin + (usable * (i + 0.5)) / count);

    const frames: KeyFrame[] = [];
    for (const [i, timestamp] of timestamps.entries()) {
      const framePath = path.join(outputDir, `frame_${String(i).padStart(3, '0')}.jpg`);
      await this.extractFrame({ inputPath, outputPath: framePath, timestamp, maxWidth: 640 });
      frames.push({ path: framePath, timestamp: round(timestamp) });
    }
    return frames;
  }

  async extractFrame({ inputPath, outputPath, timestamp, maxWidth = 1080 }: { inputPath: string; outputPath: string; timestamp: number; maxWidth?: number }): Promise<void> {
    await this.ffmpeg([
      '-ss', timestamp.toFixed(3),
      '-i', inputPath,
      '-frames:v', '1',
      '-vf', `scale='min(${maxWidth},iw)':-2`,
      '-q:v', '3',
      outputPath,
    ]);
  }

  async cutClip({ inputPath, outputPath, start, end, verticalCrop, withAudio = true }: { inputPath: string; outputPath: string; start: number; end: number; verticalCrop: boolean; withAudio?: boolean }): Promise<void> {
    const duration = Math.max(0.5, end - start);
    // Auto-crop centrado a 9:16 y escalado a 1080x1920; si no, se limita el lado mayor a 1920.
    const videoFilter = verticalCrop
      ? "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=1080:1920,setsar=1"
      : "scale='if(gt(iw,ih),min(1920,iw),-2)':'if(gt(iw,ih),-2,min(1920,ih))',setsar=1";

    await this.ffmpeg([
      '-ss', start.toFixed(3),
      '-i', inputPath,
      '-t', duration.toFixed(3),
      '-vf', videoFilter,
      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
      ...(withAudio ? ['-c:a', 'aac', '-b:a', '160k'] : ['-an']),
      '-movflags', '+faststart',
      outputPath,
    ]);
  }

  async replaceAudio({ videoPath, audioPath, outputPath }: { videoPath: string; audioPath: string; outputPath: string }): Promise<void> {
    await this.ffmpeg([
      '-i', videoPath,
      '-i', audioPath,
      '-map', '0:v:0', '-map', '1:a:0',
      '-c:v', 'copy',
      // Limpieza ligera de la voz aislada: corta graves de ruido y normaliza el volumen.
      '-af', 'highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11',
      '-c:a', 'aac', '-b:a', '160k',
      '-shortest', '-movflags', '+faststart',
      outputPath,
    ]);
  }

  async createContactSheet({ frames, outputPath }: { frames: KeyFrame[]; outputPath: string }): Promise<void> {
    if (frames.length === 0) throw new AppError('No hay fotogramas para la hoja de contactos', 'MEDIA_PROCESSING_ERROR', 500);
    // Se apoya en la secuencia frame_%03d.jpg que genera extractKeyFrames.
    const dir = path.dirname(frames[0].path);
    const expected = frames.map((_, i) => path.join(dir, `frame_${String(i).padStart(3, '0')}.jpg`));
    if (frames.some((f, i) => path.resolve(f.path) !== path.resolve(expected[i]))) {
      throw new AppError('Los fotogramas deben provenir de extractKeyFrames', 'MEDIA_PROCESSING_ERROR', 500);
    }

    const { width, height } = await this.getImageDimensions(frames[0].path);
    const ratio = width / height;
    // Celdas y columnas según orientación, para que la hoja quede aproximadamente apaisada.
    const [cellW, cellH, cols] = ratio > 1.2 ? [448, 252, 3] : ratio < 0.8 ? [252, 448, 5] : [336, 336, 4];
    const rows = Math.ceil(frames.length / cols);

    const filters = [
      `scale=${cellW}:${cellH}:force_original_aspect_ratio=decrease`,
      `pad=${cellW}:${cellH}:(ow-iw)/2:(oh-ih)/2:black`,
    ];
    const font = await findFont();
    if (font) {
      const escaped = font.replace(/\\/g, '/').replace(/:/g, '\\:');
      filters.push(`drawtext=fontfile='${escaped}':text='%{eif\\:n+1\\:d}':x=12:y=10:fontsize=44:fontcolor=yellow:borderw=3:bordercolor=black`);
    }
    filters.push(`tile=${cols}x${rows}:padding=8:color=0x18181b`);

    await this.ffmpeg(['-framerate', '1', '-i', path.join(dir, 'frame_%03d.jpg'), '-vf', filters.join(','), '-frames:v', '1', '-q:v', '3', outputPath]);
  }

  /** Convierte una imagen a JPEG con dimensiones exactas (usado para preparar entradas del modelo de visión). */
  async resizeImage(inputPath: string, outputPath: string, width: number, height: number): Promise<void> {
    await this.ffmpeg(['-i', inputPath, '-vf', `scale=${width}:${height}:flags=lanczos`, '-frames:v', '1', '-q:v', '2', outputPath]);
  }

  async isAvailable(): Promise<boolean> {
    try {
      await runProcess(this.ffmpegPath, ['-version'], { timeoutMs: 10_000 });
      return true;
    } catch {
      return false;
    }
  }

  private async ffprobe(inputPath: string): Promise<FfprobeOutput> {
    try {
      const { stdout } = await runProcess(
        this.ffprobePath,
        ['-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', inputPath],
        { timeoutMs: 60_000 },
      );
      return JSON.parse(stdout) as FfprobeOutput;
    } catch (err) {
      throw new AppError(`No se pudo analizar el archivo: ${(err as Error).message}`, 'MEDIA_PROCESSING_ERROR', 422);
    }
  }

  private async ffmpeg(args: string[]): Promise<void> {
    try {
      await runProcess(this.ffmpegPath, ['-hide_banner', '-loglevel', 'error', '-y', ...args], { timeoutMs: this.timeoutMs });
    } catch (err) {
      const message = err instanceof ProcessError ? err.message : String(err);
      throw new AppError(`FFmpeg falló: ${message}`, 'MEDIA_PROCESSING_ERROR', 500);
    }
  }
}

const round = (n: number) => Math.round(n * 100) / 100;

const FONT_CANDIDATES = [
  'C:/Windows/Fonts/arialbd.ttf',
  'C:/Windows/Fonts/arial.ttf',
  '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
  '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
  '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
  '/Library/Fonts/Arial.ttf',
];

let fontLookup: Promise<string | null> | undefined;

/** Fuente para numerar la hoja de contactos; sin fuente, la cuadrícula se genera sin números. */
function findFont(): Promise<string | null> {
  fontLookup ??= (async () => {
    for (const candidate of FONT_CANDIDATES) {
      if (await fs.access(candidate).then(() => true, () => false)) return candidate;
    }
    return null;
  })();
  return fontLookup;
}
