export interface CatalogSong {
  id: string;
  title: string;
  artist: string;
  /** De dónde sale: "Tendencia TikTok", "TikTok · reggaetón", "TikTok · fiesta", "Catálogo local"... */
  source: string;
  /** Género conocido (catálogo local) o vacío si la fuente no lo indica. */
  genre?: string;
  moods?: string[];
  /** Fragmento de audio de ~30 s para escucharla en la interfaz. */
  previewUrl?: string;
  link?: string;
  /** Fecha (ISO) en que entró a la lista de tendencias: más reciente = más actual. */
  addedAt?: string;
}

export interface MusicCandidatesRequest {
  musicStyle?: string;
  /** Texto libre del que se deducen temas (contexto del creador + descripción visual). */
  themeText?: string;
  language: string;
  /** País para el ranking (nombre o código ISO, ej. "Colombia" / "co"). */
  country?: string;
  limit: number;
}

export interface MusicCandidates {
  songs: CatalogSong[];
  /** true si son tendencias en vivo; false si se usó el catálogo local de respaldo. */
  live: boolean;
  /** Aviso para el usuario (ej. "Sin conexión: se usó el catálogo local"). */
  note?: string;
}

/** Fuente de canciones reales entre las que la IA elige (nunca inventa títulos ni artistas). */
export interface IMusicCatalog {
  candidates(request: MusicCandidatesRequest): Promise<MusicCandidates>;
}
