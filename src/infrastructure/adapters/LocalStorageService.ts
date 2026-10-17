import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import type { IStorageService } from '../../domain/ports/IStorageService';

export interface StoragePaths {
  inputs: string;
  outputs: string;
  temp: string;
}

/** Almacenamiento en disco local; los archivos se exponen vía /files/inputs y /files/outputs. */
export class LocalStorageService implements IStorageService {
  constructor(
    private readonly paths: StoragePaths,
    private readonly publicBaseUrl: string,
  ) {}

  async init(): Promise<void> {
    await Promise.all(Object.values(this.paths).map((dir) => fs.mkdir(dir, { recursive: true })));
  }

  createOutputPath(baseName: string, extension: string): string {
    const safe = baseName.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 60) || 'output';
    const ext = extension.startsWith('.') ? extension : `.${extension}`;
    return path.join(this.paths.outputs, `${safe}_${crypto.randomUUID().slice(0, 8)}${ext}`);
  }

  async createTempDir(prefix: string): Promise<string> {
    return fs.mkdtemp(path.join(this.paths.temp, `${prefix}-`));
  }

  toPublicUrl(absolutePath: string): string {
    for (const area of ['inputs', 'outputs'] as const) {
      const relative = path.relative(this.paths[area], absolutePath);
      if (relative && !relative.startsWith('..') && !path.isAbsolute(relative)) {
        const urlPath = relative.split(path.sep).map(encodeURIComponent).join('/');
        return `${this.publicBaseUrl}/files/${area}/${urlPath}`;
      }
    }
    throw new Error(`La ruta no pertenece al almacenamiento público: ${absolutePath}`);
  }

  async exists(absolutePath: string): Promise<boolean> {
    return fs.access(absolutePath).then(
      () => true,
      () => false,
    );
  }

  async remove(absolutePath: string): Promise<void> {
    await fs.rm(absolutePath, { recursive: true, force: true });
  }
}
