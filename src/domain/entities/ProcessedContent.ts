import type { ContentStrategy } from './ContentStrategy';

export interface KeyFrame {
  path: string;
  timestamp: number;
}

export interface Moment {
  start: number;
  end: number;
  score: number;
  reason: string;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

export interface Transcript {
  language: string;
  text: string;
  segments: TranscriptSegment[];
}

export interface GeneratedClip {
  url: string;
  thumbnailUrl: string;
  start: number;
  end: number;
  score: number;
  reason: string;
  /** Qué audio lleva el clip. */
  audio: 'mute' | 'voice' | 'original';
}

/** Diagnóstico técnico de la foto y correcciones aplicadas. */
export interface ImageQualityReport {
  /** Problemas detectados, en lenguaje claro (ej. "Luces quemadas en el 6% de la imagen"). */
  issues: string[];
  /** Correcciones aplicadas (ej. "Recuperación de luces altas"). */
  adjustments: string[];
  metrics: Record<string, number>;
  /** Puntuación técnica estimada 0-100 de la foto original y de la final, con su desglose. */
  score?: { before: QualityScore; after: QualityScore };
  /** Avisos (ej. inclinación que no se corrigió para no recortar). */
  notes?: string[];
}

export interface QualityScore {
  total: number;
  [part: string]: number;
}

export interface ProcessedImageContent {
  kind: 'image';
  originalUrl: string;
  processedUrl: string;
  appliedSteps: string[];
  quality?: ImageQualityReport;
  visualDescription: string;
  strategy: ContentStrategy;
}

export interface ProcessedVideoContent {
  kind: 'video';
  originalUrl: string;
  durationSeconds: number;
  clips: GeneratedClip[];
  visualDescription: string;
  transcript?: Transcript;
  strategy: ContentStrategy;
}

export type ProcessedContent = ProcessedImageContent | ProcessedVideoContent;
