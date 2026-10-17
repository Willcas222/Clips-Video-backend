import type { ContentStrategy } from '../../domain/entities/ContentStrategy';
import type { ITikTokTrendsStore, OfficialTrendHashtag } from '../../domain/ports/ITikTokTrendsStore';
import { detectThemes, normalizeText } from './themes';

/** Pasadas tres semanas, un ranking de hashtags ya no refleja la tendencia. */
const MAX_AGE_DAYS = 21;
/** Hashtags de alcance general que encajan con casi cualquier publicación. */
const GENERIC = new Set(['fyp', 'foryou', 'foryoupage', 'parati', 'paratii', 'viral', 'tendencia', 'trending', 'xyzbca', 'fypシ']);

export async function loadTrendingHashtags(store: ITikTokTrendsStore, limit = 30): Promise<{ hashtags: OfficialTrendHashtag[]; region?: string }> {
  const trends = await store.get();
  if (!trends?.hashtags?.length || !trends.hashtagsUpdatedAt) return { hashtags: [] };
  const days = (Date.now() - new Date(trends.hashtagsUpdatedAt).getTime()) / 86_400_000;
  if (days > MAX_AGE_DAYS) return { hashtags: [] };
  return { hashtags: trends.hashtags.slice(0, limit), region: trends.region };
}

/**
 * Solo los hashtags oficiales que tienen relación real con el contenido: por la categoría que
 * asigna TikTok (outfit → "Ropa y accesorios") o porque el hashtag y el contexto comparten
 * palabras ("viernes"). Así no se cuela un festival de rock en una selfie.
 */
export function relevantHashtags(hashtags: OfficialTrendHashtag[], text: string): OfficialTrendHashtag[] {
  const themes = detectThemes(text);
  const normalized = normalizeText(text);
  const tokens = new Set(normalized.split(/[^a-z0-9]+/).filter((w) => w.length >= 4));
  const themeWords = themes.flatMap((t) => [t.term, ...t.words.map((w) => w.trim())]).map(normalizeText);

  return hashtags.filter((h) => {
    const body = normalizeText(h.tag.replace(/^#/, ''));
    if (GENERIC.has(body)) return true;
    const category = normalizeText(h.category ?? '');
    if (category && themes.some((t) => t.categories.some((c) => category.includes(c)))) return true;
    if ([...tokens].some((w) => body.includes(w) || (body.length >= 4 && w.includes(body)))) return true;
    return themeWords.some((w) => w.length >= 4 && body.includes(w.replace(/\s+/g, '')));
  });
}

/**
 * Quita de la estrategia los hashtags oficiales sin relación con el contenido (si el modelo
 * los añadió igual) y marca los que sí están en tendencia.
 */
export function applyTrendingHashtags(strategy: ContentStrategy, all: OfficialTrendHashtag[], relevant: OfficialTrendHashtag[]): ContentStrategy {
  const officialSet = new Set(all.map((t) => t.tag.toLowerCase()));
  const relevantSet = new Set(relevant.map((t) => t.tag.toLowerCase()));
  const hashtags = strategy.hashtags.filter((h) => !officialSet.has(h.toLowerCase()) || relevantSet.has(h.toLowerCase()));
  return { ...strategy, hashtags, trendingHashtags: hashtags.filter((h) => relevantSet.has(h.toLowerCase())) };
}
