import path from 'path';

// Process types whose Python constructor derives its writer directory as:
//   BaseProcess.name = process.name ?? config.name
//   BaseTrainProcess.training_folder = process.training_folder ?? job.training_folder
//   BaseTrainProcess.save_root = training_folder / name
// The launch helper rewrites both inputs so save_root is exactly the folder admitted/claimed by the UI.
export const TRAINING_PROCESS_TYPES = new Set([
  // extension trainers
  'sd_trainer',
  'ui_trainer',
  'diffusion_trainer',
  'textual_inversion_trainer',
  'concept_replacer',
  'concept_slider',
  'ultimate_slider_trainer',
  'image_reference_slider_trainer',
  // built-in TrainJob process types
  'vae',
  'slider',
  'slider_old',
  'lora_hack',
  'rescale_sd',
  'esrgan',
  'reference',
]);

export class LaunchConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LaunchConfigError';
  }
}

/**
 * Bind a saved job config to the one run/folder claimed by the UI. Pure: returns a new parsed object and never writes.
 * Throws LaunchConfigError for layouts that cannot be represented by one claimed writer folder.
 */
export const bindLaunchConfig = (rawJobConfig: string | object, jobName: string, runDir: string): any => {
  let config: any;
  try {
    config = typeof rawJobConfig === 'string' ? JSON.parse(rawJobConfig) : JSON.parse(JSON.stringify(rawJobConfig));
  } catch (e: any) {
    throw new LaunchConfigError(`The saved job config is not valid JSON (${e?.message ?? e}). Re-save the job and retry.`);
  }

  if (!config || typeof config !== 'object' || !config.config || typeof config.config !== 'object') {
    throw new LaunchConfigError('The saved job config has no config object. Re-save the job and retry.');
  }
  const processes = config.config.process;
  if (!Array.isArray(processes) || processes.length === 0) {
    throw new LaunchConfigError('The saved job config has no processes. Re-save the job and retry.');
  }
  if (processes.length !== 1) {
    throw new LaunchConfigError(
      `This job has ${processes.length} processes, but a UI run claims exactly one output folder. ` +
        'Split it into one process per job (Advanced editor), then save and start again.',
    );
  }
  const process = processes[0];
  if (!process || typeof process !== 'object' || typeof process.type !== 'string' || process.type.trim() === '') {
    throw new LaunchConfigError('The saved job process has no type. Re-save the job and retry.');
  }

  const folder = path.resolve(runDir);
  const parent = path.dirname(folder);
  const folderName = path.basename(folder);
  if (parent === folder || !folderName) {
    throw new LaunchConfigError(`The admitted run folder "${runDir}" is not a job folder. Save and start the job again.`);
  }
  const namesMatch = globalThis.process.platform === 'win32'
    ? jobName.toLowerCase() === folderName.toLowerCase()
    : jobName === folderName;
  if (jobName && !namesMatch) {
    throw new LaunchConfigError(
      `The admitted run folder "${folder}" does not match job name "${jobName}". Save and start the job again.`,
    );
  }

  // Canonical claims are case-folded on Windows. Preserve the saved job's spelling for checkpoint filenames;
  // changing MixedCase to mixedcase here would make existing case-sensitive checkpoint-name filters miss a resume.
  const name = jobName || folderName;

  // The job-level name is what Python uses for logging and as the process-name fallback.
  config.config.name = name;

  if (process.type === 'InferenceEngine') {
    // InferenceEngine uses an explicit engine.job_folder for engine.json, assets and default outputs.
    process.engine = { ...(process.engine || {}), job_folder: folder };
    process.name = name;
    return config;
  }

  if (TRAINING_PROCESS_TYPES.has(process.type)) {
    // process.training_folder wins over job.training_folder, and process.name wins over config.name. Binding both
    // prevents a stale Training Folder setting or an Advanced-editor name from making the trainer write elsewhere.
    config.config.training_folder = parent;
    process.training_folder = parent;
    process.name = name;
    const actual = path.resolve(process.training_folder, process.name);
    if (path.relative(folder, actual) !== '') {
      throw new LaunchConfigError(`Could not bind process output folder to the claimed run folder "${folder}".`);
    }
    return config;
  }

  // These registered dataset tools extend BaseExtensionProcess, not BaseTrainProcess, and intentionally operate on
  // caller-selected dataset paths. An arbitrary extension name is not evidence that it has no training writer.
  if (new Set(['dataset_tools', 'sync_from_collection', 'super_tagger']).has(process.type)) return config;
  throw new LaunchConfigError(
    `Process type "${process.type}" has no supported UI output-folder binding. ` +
      'Use a supported UI trainer or run this extension from the CLI with its own output folder.',
  );
};
