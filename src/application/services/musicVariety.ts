import type { CatalogSong } from '../../domain/ports/IMusicCatalog';
import { songKey } from '../../domain/ports/IRecentSuggestions';

/**
 * Quita de las candidatas las canciones sugeridas en los últimos trabajos, para que cada
 * video/foto reciba sugerencias distintas. Si quedaran muy pocas, las recientes van al final.
 */
export function withoutRecent(songs: CatalogSong[], recent: string[], minimum = 12): CatalogSong[] {
  if (!recent.length) return songs;
  const recentSet = new Set(recent);
  const fresh = songs.filter((s) => !recentSet.has(songKey(s.title, s.artist)));
  if (fresh.length >= minimum) return fresh;
  return [...fresh, ...songs.filter((s) => recentSet.has(songKey(s.title, s.artist)))];
}
