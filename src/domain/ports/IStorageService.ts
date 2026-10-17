/** Puerto para el manejo de archivos locales. */
export interface IStorageService {
  /** Ruta absoluta para un nuevo archivo de salida. */
  createOutputPath(baseName: string, extension: string): string;
  /** Crea un directorio temporal de trabajo y devuelve su ruta. */
  createTempDir(prefix: string): Promise<string>;
  /** Convierte una ruta de inputs/outputs en URL pública servida por la API. */
  toPublicUrl(absolutePath: string): string;
  exists(absolutePath: string): Promise<boolean>;
  remove(absolutePath: string): Promise<void>;
}
