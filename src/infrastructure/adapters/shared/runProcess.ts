import { spawn } from 'node:child_process';

export interface RunProcessOptions {
  cwd?: string;
  timeoutMs?: number;
  env?: NodeJS.ProcessEnv;
  /** Recibe cada línea de stderr en vivo (logs/progreso de workers). */
  onStderrLine?: (line: string) => void;
}

export interface RunProcessResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

/** Límite de bytes retenidos por stream, para no acumular logs gigantes en memoria. */
const MAX_BUFFER = 5 * 1024 * 1024;

export class ProcessError extends Error {
  constructor(
    message: string,
    public readonly result: Partial<RunProcessResult>,
  ) {
    super(message);
    this.name = 'ProcessError';
  }
}

/**
 * Ejecuta un binario con `spawn` (sin shell) consumiendo stdout/stderr como streams,
 * de modo que procesos largos nunca bloquean el event loop ni llenan el pipe.
 */
export function runProcess(command: string, args: string[], options: RunProcessOptions = {}): Promise<RunProcessResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1', ...options.env },
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let stderrPartial = '';
    let timedOut = false;

    const append = (current: string, chunk: string) => {
      const next = current + chunk;
      return next.length > MAX_BUFFER ? next.slice(next.length - MAX_BUFFER) : next;
    };

    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk: string) => {
      stdout = append(stdout, chunk);
    });
    child.stderr.on('data', (chunk: string) => {
      stderr = append(stderr, chunk);
      if (options.onStderrLine) {
        const lines = (stderrPartial + chunk).split(/\r?\n/);
        stderrPartial = lines.pop() ?? '';
        lines.filter(Boolean).forEach((line) => options.onStderrLine!(line));
      }
    });

    const timer = options.timeoutMs
      ? setTimeout(() => {
          timedOut = true;
          child.kill('SIGKILL');
        }, options.timeoutMs)
      : undefined;

    child.on('error', (err) => {
      clearTimeout(timer);
      reject(new ProcessError(`No se pudo ejecutar "${command}": ${err.message}`, { stdout, stderr }));
    });

    child.on('close', (code) => {
      clearTimeout(timer);
      const exitCode = code ?? -1;
      if (timedOut) {
        reject(new ProcessError(`"${command}" superó el tiempo límite (${options.timeoutMs} ms)`, { stdout, stderr, exitCode }));
      } else if (exitCode !== 0) {
        const tail = stderr.trim().split(/\r?\n/).slice(-8).join('\n');
        reject(new ProcessError(`"${command}" terminó con código ${exitCode}: ${tail}`, { stdout, stderr, exitCode }));
      } else {
        resolve({ stdout, stderr, exitCode });
      }
    });
  });
}
