export type MediaKind = 'image' | 'video';

/** Archivo multimedia que ya existe en el almacenamiento local. */
export interface Media {
  id: string;
  kind: MediaKind;
  originalName: string;
  mimeType: string;
  sizeBytes: number;
  /** Ruta absoluta en disco. */
  path: string;
}

export type SocialPlatform = 'tiktok' | 'reels' | 'stories' | 'post';

/** Caja en coordenadas de píxel absolutas sobre la imagen original. */
export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Indicaciones del creador que orientan el copy y la música. */
export interface CreativeBrief {
  tone?: string;
  /** Qué está pasando / qué se quiere transmitir (ej. "salida con amigas, quiero algo empoderador"). */
  context?: string;
  /** Tipo o género de música preferido (ej. "reggaetón", "pop en inglés"). */
  musicStyle?: string;
}

export type ClipAudioMode = 'mute' | 'voice' | 'original';

export type PortraitBlur = 'off' | 'suave' | 'medio' | 'fuerte';

export interface ImageProcessingOptions extends CreativeBrief {
  /** Descripción del objeto a borrar (ej. "la bolsa del fondo"); la IA lo localiza. */
  removeObject?: string;
  /** Ruta a una máscara en blanco/negro (blanco = zona a borrar). Tiene prioridad sobre removeObject. */
  maskPath?: string;
  /** Detectar y borrar también el reflejo del objeto sobre superficies brillantes. */
  removeReflections: boolean;
  /** Corrección de aspecto: exposición, luces quemadas, balance de blancos, contraste y color. */
  enhance: boolean;
  /** Acabado profesional: modo retrato, nitidez del sujeto, acabado de color (nunca deforma a la persona). */
  proFinish: boolean;
  portraitBlur: PortraitBlur;
  /** Enderezar inclinaciones leves (giro rígido de toda la foto). */
  straighten: boolean;
  upscale: boolean;
  upscaleFactor: 2 | 4;
  platform: SocialPlatform;
  language: string;
}

export interface VideoProcessingOptions extends CreativeBrief {
  platform: SocialPlatform;
  /** Número máximo de clips a generar. */
  maxClips: number;
  /** Duración objetivo de cada clip en segundos. */
  clipDuration: number;
  /** Recortar al formato vertical 9:16. */
  verticalCrop: boolean;
  /** Extraer texto del audio con timestamps (Whisper). */
  transcribe: boolean;
  /** Audio de los clips: sin audio (para poner la canción en la app), solo la voz, o el original. */
  audioMode: ClipAudioMode;
  language: string;
}
