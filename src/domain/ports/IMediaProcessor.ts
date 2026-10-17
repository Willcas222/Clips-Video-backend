import type { BoundingBox } from '../entities/Media';
import type { ImageQualityReport, KeyFrame, QualityScore, Transcript } from '../entities/ProcessedContent';

/**
 * Puertos de procesamiento multimedia, separados por responsabilidad (ISP):
 * imágenes y transcripción los implementan workers de Python; video, FFmpeg.
 */
export interface IImageProcessor {
  /** Borra la zona indicada por una máscara o un bounding box. Devuelve el motor usado. */
  inpaint(params: {
    inputPath: string;
    outputPath: string;
    maskPath?: string;
    box?: BoundingBox;
    /** Borrar también el reflejo del objeto (por defecto true). */
    reflections?: boolean;
  }): Promise<{ engine: string; reflectionRemoved?: boolean }>;

  /**
   * Analiza la foto (exposición, luces quemadas, dominante de color, contraste, ruido) y aplica
   * correcciones tonales globales. Nunca altera la geometría: personas y objetos no se deforman.
   */
  enhance(params: {
    inputPath: string;
    /** Sin ruta de salida solo se hace el diagnóstico. */
    outputPath?: string;
  }): Promise<ImageQualityReport>;

  /** Acabado profesional (modo retrato, nitidez, color, enderezado leve). */
  proFinish(params: {
    inputPath: string;
    outputPath: string;
    /** Foto original, para la puntuación "antes". */
    referencePath?: string;
    blur: 'off' | 'suave' | 'medio' | 'fuerte';
    straighten: boolean;
  }): Promise<{ adjustments: string[]; notes: string[]; scoreBefore: QualityScore; scoreAfter: QualityScore }>;

  upscale(params: {
    inputPath: string;
    outputPath: string;
    factor: 2 | 4;
    /** Avance 0-100 mientras el modelo procesa. */
    onProgress?: (percent: number) => void;
  }): Promise<{ engine: string }>;
}

/** Separa la voz del resto del audio (música, ruido de fondo). */
export interface IVoiceIsolator {
  /** Devuelve un audio solo con la voz del video/audio de entrada. */
  isolateVoice(params: { inputPath: string; outputPath: string }): Promise<{ engine: string; hasVoice: boolean }>;
}

export interface ITranscriber {
  transcribe(params: { inputPath: string; language?: string }): Promise<Transcript>;
}

export interface VideoInfo {
  durationSeconds: number;
  width: number;
  height: number;
  hasAudio: boolean;
}

export interface IVideoProcessor {
  probe(inputPath: string): Promise<VideoInfo>;
  extractKeyFrames(params: { inputPath: string; outputDir: string; count: number }): Promise<KeyFrame[]>;
  cutClip(params: {
    inputPath: string;
    outputPath: string;
    start: number;
    end: number;
    verticalCrop: boolean;
    /** false = clip sin pista de audio (listo para poner la canción en la app). */
    withAudio?: boolean;
  }): Promise<void>;
  /** Sustituye el audio de un video por otro archivo de audio (sin recodificar el video). */
  replaceAudio(params: { videoPath: string; audioPath: string; outputPath: string }): Promise<void>;
  extractFrame(params: { inputPath: string; outputPath: string; timestamp: number }): Promise<void>;
  /** Une fotogramas (de extractKeyFrames) en una cuadrícula numerada 1..N, en orden cronológico. */
  createContactSheet(params: { frames: KeyFrame[]; outputPath: string }): Promise<void>;
}

/** Agregado: procesador multimedia completo que los casos de uso pueden recibir. */
export interface IMediaProcessor {
  image: IImageProcessor;
  video: IVideoProcessor;
  transcriber: ITranscriber;
}
