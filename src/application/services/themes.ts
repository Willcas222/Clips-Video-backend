/** Temas de contenido que se detectan en el contexto/descripción (para música y hashtags). */
export interface Theme {
  term: string;
  words: string[];
  /** Palabras de las categorías del Creative Center de TikTok que corresponden a este tema. */
  categories: string[];
}

export const THEMES: Theme[] = [
  { term: 'fiesta', words: ['fiesta', 'discoteca', 'antro', 'rumba', 'party', 'farra', 'parche', 'salir de noche', 'perreo'], categories: [] },
  { term: 'gym', words: ['gym', 'gimnasio', 'entreno', 'entrenamiento', 'fitness', 'pesas', 'correr', 'deporte', 'rutina'], categories: ['deporte', 'salud', 'fitness'] },
  { term: 'viajes', words: ['viaje', 'viajar', 'vacaciones', 'aeropuerto', 'turismo', 'roadtrip', 'mochilero'], categories: ['viaje', 'turismo'] },
  { term: 'verano', words: ['playa', 'verano', 'piscina', 'mar ', 'sol ', 'bikini', 'tropical'], categories: ['viaje'] },
  { term: 'amor', words: ['amor', 'pareja', 'novio', 'novia', 'enamorad', 'romantic', 'aniversario', 'san valentin', 'beso'], categories: [] },
  { term: 'boda', words: ['boda', 'matrimonio', 'novios', 'casamiento'], categories: ['boda'] },
  { term: 'cumpleaños', words: ['cumpleanos', 'cumple', 'pastel', 'torta'], categories: [] },
  { term: 'comida', words: ['comida', 'receta', 'cocina', 'restaurante', 'cafe', 'postre', 'tacos', 'cocinar'], categories: ['comida', 'bebida', 'alimento'] },
  { term: 'outfit', words: ['outfit', 'moda', 'ropa', 'look', 'estilo', 'vestido', 'selfie', 'espejo', 'glow', 'falda', 'top '], categories: ['ropa', 'accesorio', 'moda'] },
  { term: 'makeup', words: ['maquillaje', 'makeup', 'belleza', 'skincare', 'unas', 'peinado', 'cabello', 'labial', 'antes de salir', 'arreglarme', 'arreglandome', 'alistandome', 'grwm'], categories: ['belleza', 'cuidado personal', 'cosmetic'] },
  { term: 'mascotas', words: ['mascota', 'perro', 'gato', 'perrito', 'gatito', 'cachorro'], categories: ['mascota', 'animal'] },
  { term: 'motivacion', words: ['motivacion', 'exito', 'emprend', 'negocio', 'metas', 'disciplina', 'superacion'], categories: ['negocio', 'finanza', 'educacion'] },
  { term: 'desamor', words: ['triste', 'desamor', 'ruptura', 'llorar', 'extrano', 'soledad', 'indirecta'], categories: [] },
  { term: 'baile', words: ['baile', 'bailar', 'coreografia', 'dance', 'trend de baile'], categories: [] },
  { term: 'humor', words: ['humor', 'gracioso', 'comedia', 'broma', 'chiste', 'divertid'], categories: [] },
];

export const normalizeText = (s: string) =>
  s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/\s+/g, ' ');

/** Temas presentes en el texto, del más mencionado al menos. */
export function detectThemes(text: string): Theme[] {
  const t = ` ${normalizeText(text)} `;
  return THEMES.map((theme) => ({ theme, hits: theme.words.filter((w) => t.includes(w)).length }))
    .filter((x) => x.hits > 0)
    .sort((a, b) => b.hits - a.hits)
    .map((x) => x.theme);
}
