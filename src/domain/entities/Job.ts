import type { MediaKind } from './Media';
import type { ProcessedContent } from './ProcessedContent';

export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed';

/** Trabajo asíncrono de procesamiento que el frontend consulta por polling. */
export interface Job {
  id: string;
  kind: MediaKind;
  status: JobStatus;
  /** Paso actual legible para humanos (ej. "Mejorando calidad"). */
  step: string;
  /** Progreso aproximado 0-100. */
  progress: number;
  warnings: string[];
  result?: ProcessedContent;
  error?: string;
  createdAt: string;
  updatedAt: string;
}

/** Contexto que los casos de uso reciben para informar avance sin conocer la persistencia. */
export interface ProgressReporter {
  step(label: string, progress: number): void;
  warn(message: string): void;
}

export const silentReporter: ProgressReporter = {
  step: () => {},
  warn: () => {},
};
