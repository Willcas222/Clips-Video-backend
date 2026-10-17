/** Canciones sugeridas recientemente, para no repetir las mismas en cada trabajo. */
export interface IRecentSuggestions {
  /** Claves (título|artista normalizados) de las canciones sugeridas en los últimos trabajos. */
  recent(): Promise<string[]>;
  /** Registra las canciones sugeridas en un trabajo. */
  remember(songs: Array<{ title: string; artist: string }>): Promise<void>;
}

export const songKey = (title: string, artist: string) =>
  `${title}|${artist.split(/,|&| x | ft\.?| feat/i)[0]}`
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9|]/g, '');
