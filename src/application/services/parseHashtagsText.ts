import type { OfficialTrendHashtag } from '../../domain/ports/ITikTokTrendsStore';

/**
 * Extrae hashtags en orden de ranking de texto pegado (tabla "Hashtag" del Creative Center:
 * "1 / #rockalparque / Noticias y entretenimiento / 10.7K Publicaciones / 65.6M ..."),
 * o de una lista simple. La categoría es la línea de texto que sigue al hashtag, si existe.
 */
export function parseHashtagsText(text: string, max = 100): OfficialTrendHashtag[] {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.replace(/ /g, ' ').trim())
    .filter(Boolean);

  const tags: OfficialTrendHashtag[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < lines.length && tags.length < max; i++) {
    for (const match of lines[i].matchAll(/#([\p{L}\p{N}_]{2,60})/gu)) {
      const tag = `#${match[1]}`;
      const key = tag.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      const next = lines[i + 1] ?? '';
      const category = isCategory(next) ? next : undefined;
      tags.push({ rank: tags.length + 1, tag, category });
    }
  }
  return tags;
}

/** Una categoría es texto corto sin números ni '#' (ej. "Noticias y entretenimiento"). */
function isCategory(line: string) {
  return line.length >= 3 && line.length <= 40 && !line.includes('#') && !/\d/.test(line) && !/^(ver|see)\b/i.test(line);
}
