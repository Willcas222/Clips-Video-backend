/** Tramo de la canción que conviene usar en el video (normalmente el coro). */
export interface SongHook {
  /** Segundos desde el inicio de la canción. */
  start: number;
  end: number;
  /** Primera línea de letra del tramo, para reconocerlo al escuchar. */
  lyric?: string;
  /** Duración total de la canción en segundos. */
  durationSeconds?: number;
  /** Cómo se obtuvo: "coro" (líneas que se repiten en la letra sincronizada). */
  method: 'coro';
}

/** Encuentra el tramo más reconocible (coro) de una canción real. */
export interface IMusicHookFinder {
  findHook(title: string, artist: string): Promise<SongHook | null>;
}
