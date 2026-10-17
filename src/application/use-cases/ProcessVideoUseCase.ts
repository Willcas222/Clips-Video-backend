import path from 'node:path';
import type { ClipAudioMode, Media, VideoProcessingOptions } from '../../domain/entities/Media';
import type { GeneratedClip, Transcript } from '../../domain/entities/ProcessedContent';
import { silentReporter, type ProgressReporter } from '../../domain/entities/Job';
import type { ITranscriber, IVideoProcessor, IVoiceIsolator } from '../../domain/ports/IMediaProcessor';
import type { FrameAnalysis, IAIAssistant } from '../../domain/ports/IAIAssistant';
import type { IStorageService } from '../../domain/ports/IStorageService';
import { selectBestMoments } from '../../domain/services/MomentSelector';

export interface ProcessVideoResult {
  durationSeconds: number;
  clips: GeneratedClip[];
  frameAnalyses: FrameAnalysis[];
  transcript?: Transcript;
}

/** Flujo B (parte 1): fotogramas clave -> visión -> mejores momentos -> recortes (+ transcripción). */
export class ProcessVideoUseCase {
  constructor(
    private readonly video: IVideoProcessor,
    private readonly transcriber: ITranscriber,
    private readonly voice: IVoiceIsolator,
    private readonly ai: IAIAssistant,
    private readonly storage: IStorageService,
  ) {}

  async execute(media: Media, options: VideoProcessingOptions, progress: ProgressReporter = silentReporter): Promise<ProcessVideoResult> {
    progress.step('Analizando el video', 5);
    const info = await this.video.probe(media.path);

    const tempDir = await this.storage.createTempDir('frames');
    try {
      // ~1 fotograma cada 4 s, entre 4 y 12: más celdas en la hoja de contactos las haría ilegibles.
      const frameCount = Math.min(12, Math.max(4, Math.round(info.durationSeconds / 4)));
      progress.step(`Extrayendo ${frameCount} fotogramas clave`, 10);
      const frames = await this.video.extractKeyFrames({ inputPath: media.path, outputDir: tempDir, count: frameCount });
      const sheetPath = path.join(tempDir, 'contact_sheet.jpg');
      await this.video.createContactSheet({ frames, outputPath: sheetPath });

      progress.step('Buscando los mejores momentos con IA de visión', 20);
      const frameAnalyses = await this.ai.analyzeFrameSheet({ sheetPath, frames, language: options.language, context: options.context });

      const moments = selectBestMoments({
        analyses: frameAnalyses,
        durationSeconds: info.durationSeconds,
        clipDuration: options.clipDuration,
        maxClips: options.maxClips,
      });

      const baseName = path.parse(media.originalName).name;
      const clips: GeneratedClip[] = [];
      for (const [i, moment] of moments.entries()) {
        const clipPath = this.storage.createOutputPath(`${baseName}_clip${i + 1}`, '.mp4');
        const audio = await this.cutWithAudio({
          inputPath: media.path,
          clipPath,
          tempDir,
          start: moment.start,
          end: moment.end,
          verticalCrop: options.verticalCrop,
          mode: info.hasAudio ? options.audioMode : 'mute',
          label: `clip ${i + 1} de ${moments.length}`,
          progress,
          progressAt: 50 + Math.round((i / moments.length) * 20),
        });
        const thumbPath = this.storage.createOutputPath(`${baseName}_clip${i + 1}_thumb`, '.jpg');
        await this.video.extractFrame({ inputPath: clipPath, outputPath: thumbPath, timestamp: Math.min(1, (moment.end - moment.start) / 2) });
        clips.push({
          url: this.storage.toPublicUrl(clipPath),
          thumbnailUrl: this.storage.toPublicUrl(thumbPath),
          start: moment.start,
          end: moment.end,
          score: moment.score,
          reason: moment.reason,
          audio,
        });
      }

      let transcript: Transcript | undefined;
      if (options.transcribe) {
        if (!info.hasAudio) {
          progress.warn('El video no tiene audio; se omitió la transcripción.');
        } else {
          progress.step('Transcribiendo audio con Whisper', 72);
          try {
            transcript = await this.transcriber.transcribe({ inputPath: media.path, language: options.language });
          } catch (err) {
            // La transcripción es opcional: no debe tumbar todo el pipeline.
            progress.warn(`No se pudo transcribir el audio: ${(err as Error).message}`);
          }
        }
      }

      return { durationSeconds: info.durationSeconds, clips, frameAnalyses, transcript };
    } finally {
      await this.storage.remove(tempDir);
    }
  }

  /**
   * Corta el clip con el audio pedido:
   *  - mute: sin pista de audio, listo para añadir la canción en TikTok/Reels.
   *  - voice: solo la voz (Demucs quita música y ruido de fondo); si no hay voz, queda sin audio.
   *  - original: el audio tal cual.
   */
  private async cutWithAudio(p: {
    inputPath: string;
    clipPath: string;
    tempDir: string;
    start: number;
    end: number;
    verticalCrop: boolean;
    mode: ClipAudioMode;
    label: string;
    progress: ProgressReporter;
    progressAt: number;
  }): Promise<ClipAudioMode> {
    const cut = (outputPath: string, withAudio: boolean) =>
      this.video.cutClip({ inputPath: p.inputPath, outputPath, start: p.start, end: p.end, verticalCrop: p.verticalCrop, withAudio });

    if (p.mode !== 'voice') {
      p.progress.step(`Recortando ${p.label}${p.mode === 'mute' ? ' (sin audio)' : ''}`, p.progressAt);
      await cut(p.clipPath, p.mode === 'original');
      return p.mode;
    }

    p.progress.step(`Recortando ${p.label} y aislando la voz`, p.progressAt);
    const base = path.basename(p.clipPath, '.mp4');
    const withSound = path.join(p.tempDir, `${base}_raw.mp4`);
    await cut(withSound, true);
    try {
      const voicePath = path.join(p.tempDir, `${base}_voz.wav`);
      const { hasVoice } = await this.voice.isolateVoice({ inputPath: withSound, outputPath: voicePath });
      if (!hasVoice) {
        p.progress.warn(`En el ${p.label} no se detectó voz: quedó sin audio para que pongas la canción.`);
        await cut(p.clipPath, false);
        return 'mute';
      }
      await this.video.replaceAudio({ videoPath: withSound, audioPath: voicePath, outputPath: p.clipPath });
      return 'voice';
    } catch (err) {
      // Si la separación falla, mejor un clip sin audio que uno con la música de fondo.
      p.progress.warn(`No se pudo aislar la voz del ${p.label} (${(err as Error).message}); quedó sin audio.`);
      await cut(p.clipPath, false);
      return 'mute';
    }
  }
}
