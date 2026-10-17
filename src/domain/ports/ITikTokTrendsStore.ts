/** Canción del ranking oficial de TikTok que el usuario pegó desde el Creative Center. */
export interface OfficialTrendSong {
  rank: number;
  title: string;
  artist: string;
  previewUrl?: string;
  link?: string;
}

/** Hashtag del ranking oficial de TikTok (Creative Center → Hashtag). */
export interface OfficialTrendHashtag {
  rank: number;
  tag: string;
  /** Sector que indica TikTok (ej. "Noticias y entretenimiento"). */
  category?: string;
}

export interface OfficialTrends {
  /** País o región del ranking (ej. "Colombia"). */
  region: string;
  /** Fecha ISO en que se pegó la lista. */
  updatedAt: string;
  songs: OfficialTrendSong[];
  hashtags?: OfficialTrendHashtag[];
  /** Fecha ISO en que se pegaron los hashtags (pueden actualizarse por separado). */
  hashtagsUpdatedAt?: string;
}

/** Persistencia de la lista oficial de tendencias que importa el usuario. */
export interface ITikTokTrendsStore {
  get(): Promise<OfficialTrends | null>;
  save(trends: OfficialTrends): Promise<void>;
  clear(): Promise<void>;
}

/** Busca el preview y el enlace de una canción concreta (opcional, mejor esfuerzo). */
export interface ISongLookup {
  lookup(title: string, artist: string): Promise<{ previewUrl?: string; link?: string } | null>;
}
