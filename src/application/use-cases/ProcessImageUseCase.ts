import path from 'node:path';
import fs from 'node:fs/promises';
import type { ImageProcessingOptions, Media } from '../../domain/entities/Media';
import type { ImageQualityReport } from '../../domain/entities/ProcessedContent';
import { silentReporter, type ProgressReporter } from '../../domain/entities/Job';
import type { IImageProcessor } from '../../domain/ports/IMediaProcessor';
import type { IAIAssistant } from '../../domain/ports/IAIAssistant';
import type { IStorageService } from '../../domain/ports/IStorageService';

export interface ProcessImageResult {
  processedPath: string;
  appliedSteps: string[];
  quality?: ImageQualityReport;
}

/**
 * Flujo A (parte 1): borrar objetos (y su reflejo), corregir el aspecto (luz y color) y
 * mejorar la resolución. El orden importa: la corrección tonal se hace antes del upscaling
 * (más barata y sin amplificar defectos).
 */
export class ProcessImageUseCase {
  constructor(
    private readonly imageProcessor: IImageProcessor,
    private readonly ai: IAIAssistant,
    private readonly storage: IStorageService,
  ) {}

  async execute(media: Media, options: ImageProcessingOptions, progress: ProgressReporter = silentReporter): Promise<ProcessImageResult> {
    const baseName = path.parse(media.originalName).name;
    const appliedSteps: string[] = [];
    let current = media.path;

    const replaceCurrent = async (next: string) => {
      if (current !== media.path) await this.storage.remove(current);
      current = next;
    };

    if (options.maskPath || options.removeObject) {
      progress.step('Preparando borrado de objeto', 10);
      let box;
      if (!options.maskPath && options.removeObject) {
        progress.step(`Buscando "${options.removeObject}" con IA de visión`, 15);
        box = await this.ai.locateObject(current, options.removeObject);
        if (!box) progress.warn(`No se encontró "${options.removeObject}" en la imagen; se omitió el borrado.`);
      }

      if (options.maskPath || box) {
        progress.step(options.removeReflections ? 'Borrando objeto y su reflejo (inpainting)' : 'Borrando objeto (inpainting)', 30);
        const output = this.storage.createOutputPath(`${baseName}_clean`, '.png');
        const { engine, reflectionRemoved } = await this.imageProcessor.inpaint({
          inputPath: current,
          outputPath: output,
          maskPath: options.maskPath,
          box: box ?? undefined,
          reflections: options.removeReflections,
        });
        appliedSteps.push(`Objeto borrado (${engine})`);
        if (reflectionRemoved) appliedSteps.push('Reflejo del objeto borrado');
        await replaceCurrent(output);
      }
    }

    // Diagnóstico siempre (es rápido); la corrección solo si se pidió.
    progress.step(options.enhance ? 'Analizando y corrigiendo luz y color' : 'Analizando luz y color', 42);
    let quality: ImageQualityReport;
    if (options.enhance) {
      const output = this.storage.createOutputPath(`${baseName}_look`, '.png');
      quality = await this.imageProcessor.enhance({ inputPath: current, outputPath: output });
      appliedSteps.push('Aspecto mejorado (luz, color y contraste)');
      await replaceCurrent(output);
    } else {
      quality = await this.imageProcessor.enhance({ inputPath: current });
    }

    // La resolución se mejora ANTES del acabado profesional: Real-ESRGAN está entrenado para
    // volver nítido lo borroso, así que aplicado después "re-enfocaría" el fondo desenfocado
    // inventando bordes y dejando parches.
    if (options.upscale) {
      const label = `Mejorando calidad x${options.upscaleFactor}`;
      progress.step(label, 45);
      const output = this.storage.createOutputPath(`${baseName}_hd`, '.png');
      const { engine } = await this.imageProcessor.upscale({
        inputPath: current,
        outputPath: output,
        factor: options.upscaleFactor,
        onProgress: (percent) => progress.step(`${label} (${percent}%)`, 45 + Math.round(percent * 0.2)),
      });
      appliedSteps.push(`Calidad mejorada x${options.upscaleFactor} (${engine})`);
      await replaceCurrent(output);
    }

    if (options.proFinish) {
      progress.step('Acabado profesional: modo retrato, nitidez y color', 67);
      const output = this.storage.createOutputPath(`${baseName}_pro`, '.png');
      const pro = await this.imageProcessor.proFinish({
        inputPath: current,
        outputPath: output,
        referencePath: media.path,
        blur: options.portraitBlur,
        straighten: options.straighten,
      });
      appliedSteps.push('Acabado profesional');
      quality = {
        ...quality,
        adjustments: [...quality.adjustments, ...pro.adjustments],
        notes: pro.notes,
        score: { before: pro.scoreBefore, after: pro.scoreAfter },
      };
      await replaceCurrent(output);
    }

    if (current === media.path) {
      // Sin transformaciones: se publica una copia para mantener inputs/outputs separados.
      const output = this.storage.createOutputPath(baseName, path.extname(media.originalName) || '.jpg');
      await fs.copyFile(media.path, output);
      current = output;
    }

    return { processedPath: current, appliedSteps, quality };
  }
}
