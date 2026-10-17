import type { Moment } from '../entities/ProcessedContent';
import type { FrameAnalysis } from '../ports/IAIAssistant';

/**
 * Regla de negocio pura: a partir de fotogramas puntuados elige ventanas de
 * `clipDuration` segundos que no se solapen, priorizando las de mayor puntuación.
 */
export function selectBestMoments(params: {
  analyses: FrameAnalysis[];
  durationSeconds: number;
  clipDuration: number;
  maxClips: number;
}): Moment[] {
  const { durationSeconds, maxClips } = params;
  const clipDuration = Math.min(params.clipDuration, durationSeconds);
  const analyses = [...params.analyses].sort((a, b) => a.timestamp - b.timestamp);

  if (durationSeconds <= 0 || maxClips <= 0) return [];
  if (analyses.length === 0 || durationSeconds <= clipDuration * 1.1) {
    return [{ start: 0, end: round(durationSeconds), score: avg(analyses.map((a) => a.score)) || 5, reason: analyses[0]?.description || 'Video completo' }];
  }

  const candidates = analyses.map((center) => {
    const start = clampWindowStart(center.timestamp - clipDuration / 2, clipDuration, durationSeconds);
    const end = start + clipDuration;
    const inside = analyses.filter((a) => a.timestamp >= start && a.timestamp <= end);
    // El fotograma central pesa doble: es el "momento" que motivó el clip.
    const score = (avg(inside.map((a) => a.score)) + center.score) / 2;
    return { start, end, score, reason: center.description };
  });

  const selected: Moment[] = [];
  for (const candidate of candidates.sort((a, b) => b.score - a.score)) {
    if (selected.length >= maxClips) break;
    const overlaps = selected.some((s) => candidate.start < s.end && candidate.end > s.start);
    if (!overlaps) {
      selected.push({ start: round(candidate.start), end: round(candidate.end), score: round(candidate.score), reason: candidate.reason });
    }
  }
  return selected.sort((a, b) => a.start - b.start);
}

function clampWindowStart(start: number, length: number, total: number) {
  return Math.max(0, Math.min(start, total - length));
}

const avg = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
const round = (n: number) => Math.round(n * 100) / 100;
