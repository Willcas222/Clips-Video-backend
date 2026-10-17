export interface MusicSuggestion {
  title: string;
  artist: string;
  reason: string;
  /** Tramo a usar, ej. "0:37 – 0:53" (el coro, detectado en la letra sincronizada). */
  segment?: string;
  segmentStart?: number;
  segmentEnd?: number;
  /** Primera línea de letra del tramo, para reconocerlo al escuchar. */
  hookLyric?: string;
  /** Duración total de la canción (segundos). */
  durationSeconds?: number;
  /** Lista de origen, ej. "Tendencia TikTok" o "TikTok · reggaetón". */
  source?: string;
  /** Fragmento de ~30 s para escucharla. */
  previewUrl?: string;
  link?: string;
  /** Cuándo entró a la lista de tendencias (YYYY-MM-DD). */
  addedAt?: string;
}

/** Estrategia de contenido generada por la IA para una publicación. */
export interface ContentStrategy {
  copy: string;
  hashtags: string[];
  /** Los hashtags sugeridos que están en el ranking oficial de TikTok importado. */
  trendingHashtags?: string[];
  music: MusicSuggestion[];
  /** Texto en pantalla sugerido (overlays) para el video/historia. */
  onScreenText: string[];
  /** Consejos de publicación (hora, gancho, CTA...). */
  tips: string[];
}
