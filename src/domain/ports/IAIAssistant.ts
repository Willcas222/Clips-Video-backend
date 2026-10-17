import type { BoundingBox, CreativeBrief, SocialPlatform } from '../entities/Media';
import type { ContentStrategy } from '../entities/ContentStrategy';
import type { KeyFrame, Transcript } from '../entities/ProcessedContent';
import type { CatalogSong } from './IMusicCatalog';

export interface StrategyRequest extends CreativeBrief {
  mediaKind: 'image' | 'video';
  platform: SocialPlatform;
  visualDescription: string;
  language: string;
  transcript?: Transcript;
  durationSeconds?: number;
  /** Canciones reales entre las que la IA debe elegir; sin lista, sugiere libremente. */
  musicCandidates?: CatalogSong[];
  /** Hashtags en tendencia oficial (Creative Center) para incluir los que encajen. */
  trendingHashtags?: Array<{ tag: string; category?: string }>;
  trendsRegion?: string;
}

export interface FrameAnalysis {
  timestamp: number;
  /** Potencial viral 0-10. */
  score: number;
  description: string;
}

/** Puerto hacia el motor de IA (LLM de texto + modelo de visión). */
export interface IAIAssistant {
  /** Describe una imagen con el modelo de visión; el contexto del creador orienta la descripción. */
  describeImage(imagePath: string, language: string, context?: string): Promise<string>;

  /** Localiza un objeto descrito en lenguaje natural; null si no lo encuentra. */
  locateObject(imagePath: string, objectDescription: string): Promise<BoundingBox | null>;

  /**
   * Analiza una hoja de contactos (fotogramas numerados 1..N) en una sola pasada de visión
   * y puntúa cada fotograma según su potencial para un clip corto.
   */
  analyzeFrameSheet(params: { sheetPath: string; frames: KeyFrame[]; language: string; context?: string }): Promise<FrameAnalysis[]>;

  generateStrategy(request: StrategyRequest): Promise<ContentStrategy>;
}
