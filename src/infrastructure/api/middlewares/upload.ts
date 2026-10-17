import path from 'node:path';
import crypto from 'node:crypto';
import multer from 'multer';
import { AppError } from '../../../domain/errors/AppError';

const IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff']);
const VIDEO_TYPES = new Set(['video/mp4', 'video/quicktime', 'video/webm', 'video/x-matroska', 'video/x-msvideo', 'video/mpeg']);

function createStorage(inputsDir: string) {
  return multer.diskStorage({
    destination: inputsDir,
    filename: (_req, file, cb) => {
      const ext = path.extname(file.originalname).toLowerCase().replace(/[^.a-z0-9]/g, '') || '';
      cb(null, `${Date.now()}_${crypto.randomUUID().slice(0, 8)}${ext}`);
    },
  });
}

/** Multer para imágenes: campo `file` (imagen) y `mask` opcional (PNG blanco/negro). */
export function imageUpload(inputsDir: string, maxBytes: number) {
  return multer({
    storage: createStorage(inputsDir),
    limits: { fileSize: maxBytes, files: 2 },
    fileFilter: (_req, file, cb) => {
      if (IMAGE_TYPES.has(file.mimetype)) return cb(null, true);
      cb(new AppError(`Tipo de imagen no soportado: ${file.mimetype}`, 'UNSUPPORTED_MEDIA', 415));
    },
  }).fields([
    { name: 'file', maxCount: 1 },
    { name: 'mask', maxCount: 1 },
  ]);
}

/** Multer para videos: campo `file`. Soporta archivos grandes porque escribe directo a disco. */
export function videoUpload(inputsDir: string, maxBytes: number) {
  return multer({
    storage: createStorage(inputsDir),
    limits: { fileSize: maxBytes, files: 1 },
    fileFilter: (_req, file, cb) => {
      if (VIDEO_TYPES.has(file.mimetype)) return cb(null, true);
      cb(new AppError(`Tipo de video no soportado: ${file.mimetype}`, 'UNSUPPORTED_MEDIA', 415));
    },
  }).single('file');
}
