import path from 'path';

export class LaunchConfigError extends Error {}

const datasetTools = new Set(['dataset_tools', 'sync_from_collection', 'super_tagger']);

export const bindLaunchConfig = (rawJobConfig: string | object, jobName: string, runDir: string): any => {
  let config: any;
  try {
    config = typeof rawJobConfig === 'string' ? JSON.parse(rawJobConfig) : JSON.parse(JSON.stringify(rawJobConfig));
  } catch (e: any) {
    throw new LaunchConfigError(`The saved job config is not valid JSON (${e?.message ?? e}). Re-save the job and retry.`);
  }

  const processes = config?.config?.process;
  if (!Array.isArray(processes) || !processes.length) {
    throw new LaunchConfigError('The saved job config has no processes. Re-save the job and retry.');
  }
  const folder = path.resolve(runDir);
  const parent = path.dirname(folder);
  const baseName = path.basename(folder);
  const name = process.platform === 'win32' && jobName.toLowerCase() === baseName.toLowerCase() ? jobName : baseName;
  if (parent === folder || !name) throw new LaunchConfigError(`Invalid run folder "${runDir}".`);

  // Python runs these processes in sequence. Compare their original writer locations before rebinding:
  // collapsing two explicitly distinct model outputs could make the second resume the first one's weights.
  const writerFolders = new Set<string>();
  for (const proc of processes) {
    if (typeof proc?.type !== 'string' || !proc.type.trim()) {
      throw new LaunchConfigError('The saved job process has no type. Re-save the job and retry.');
    }
    // Captioners and dataset tools operate on explicit dataset paths. Their names are independent of trainer outputs.
    if (datasetTools.has(proc.type) || (proc.type.endsWith('Captioner') && proc.caption?.path_to_caption != null)) continue;
    if (proc.type === 'InferenceEngine') {
      proc.engine = { ...proc.engine, job_folder: folder };
    } else {
      const original = path.resolve(proc.training_folder ?? config.config.training_folder ?? parent, proc.name ?? config.config.name ?? jobName);
      writerFolders.add(process.platform === 'win32' ? original.toLowerCase() : original);
      proc.training_folder = parent;
    }
    // BaseTrainProcess reads name/training_folder. Captioners and other extensions keep their own
    // settings (e.g. the directory to caption). Python's extension registry resolves the type.
    proc.name = name;
  }
  if (writerFolders.size > 1) {
    throw new LaunchConfigError('The processes specify different output folders. Use a separate UI job for each output folder.');
  }
  config.config.name = name;
  config.config.training_folder = parent;
  return config;
};
