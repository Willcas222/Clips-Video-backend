export type ErrorCode =
  | 'VALIDATION_ERROR'
  | 'NOT_FOUND'
  | 'UNSUPPORTED_MEDIA'
  | 'AI_SERVICE_ERROR'
  | 'WORKER_ERROR'
  | 'MEDIA_PROCESSING_ERROR'
  | 'INTERNAL_ERROR';

/** Error de dominio/aplicación con un código estable y un status HTTP sugerido. */
export class AppError extends Error {
  constructor(
    message: string,
    public readonly code: ErrorCode = 'INTERNAL_ERROR',
    public readonly statusCode = 500,
    public readonly details?: unknown,
  ) {
    super(message);
    this.name = 'AppError';
  }

  static validation(message: string, details?: unknown) {
    return new AppError(message, 'VALIDATION_ERROR', 400, details);
  }

  static notFound(message: string) {
    return new AppError(message, 'NOT_FOUND', 404);
  }
}
