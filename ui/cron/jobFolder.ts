import path from 'path';

// The worker and delete route use the job name as one directory under the training root.
export const jobFolderError = (trainingRoot: string, name: unknown): string | null => {
  if (typeof name !== 'string' || name.trim() === '') return 'Job name is required';
  const root = path.resolve(trainingRoot);
  const folder = path.resolve(root, name);
  if (path.basename(name) !== name || path.dirname(folder) !== root || folder === root) return 'Invalid job name';
  const normalized = name.trim().replace(/[. ]+$/, '').toLowerCase();
  if (normalized === '.aitk_locks') return 'The training lock directory is reserved';
  if (process.platform === 'win32' && !normalized) return 'Invalid job name';
  return null;
};
