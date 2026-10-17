import type { ContentStrategy } from '../../domain/entities/ContentStrategy';
import type { IMusicHookFinder } from '../../domain/ports/IMusicHookFinder';

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;

/**
 * Sustituye la "parte a usar" que escribe la IA (que no escucha la canción) por el tramo real
 * del coro, con minuto de inicio y fin. Si no hay letra sincronizada, se deja sin tramo.
 */
export async function addSongHooks(strategy: ContentStrategy, finder: IMusicHookFinder): Promise<ContentStrategy> {
  const music = await Promise.all(
    strategy.music.map(async (song) => {
      const hook = await finder.findHook(song.title, song.artist).catch(() => null);
      if (!hook) return { ...song, segment: undefined };
      return {
        ...song,
        segment: `${fmt(hook.start)} – ${fmt(hook.end)}`,
        segmentStart: hook.start,
        segmentEnd: hook.end,
        hookLyric: hook.lyric,
        durationSeconds: hook.durationSeconds,
      };
    }),
  );
  return { ...strategy, music };
}
