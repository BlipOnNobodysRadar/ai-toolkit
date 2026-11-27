import fs from 'fs';
import path from 'path';
import type { PrismaClient } from '@prisma/client';
import { pathWithinRoots } from '../safePath';
import { jobOutputFolder } from './resources';

type OutputJob = Parameters<typeof jobOutputFolder>[1];

// Historical directories can sit outside today's configured roots. Do not follow a
// replaced job directory, sample, thumbnail, or missing destination through a link.
export const regularJobPath = async (folder: string, candidate: string): Promise<boolean> => {
  const relative = path.relative(folder, candidate);
  if (path.isAbsolute(relative) || relative === '..' || relative.startsWith(`..${path.sep}`)) return false;
  let current = folder;
  for (const component of ['', ...relative.split(path.sep).filter(Boolean)]) {
    current = path.join(current, component);
    try {
      if ((await fs.promises.lstat(current)).isSymbolicLink()) return false;
    } catch (error: any) {
      if (error.code === 'ENOENT') return true;
      throw error;
    }
  }
  return true;
};

export const jobArtifactPath = async (trainingRoot: string, job: OutputJob, ...parts: string[]) => {
  const folder = jobOutputFolder(trainingRoot, job);
  const target = path.join(folder, ...parts);
  const historical = path.relative(path.resolve(trainingRoot, job.name), folder) !== '';
  const allowed = historical ? await regularJobPath(folder, target) : await pathWithinRoots([trainingRoot], target);
  if (!allowed) throw new Error('The output path is outside the job storage.');
  return target;
};

// Native file/media requests carry a path, not a job ID. Only extend the read roots for
// an actual server-recorded run directory and a file physically inside that directory.
export const historicalArtifactFolder = async (db: PrismaClient, trainingRoot: string, file: string) => {
  const ancestors: string[] = [];
  for (let dir = path.dirname(file); dir !== path.dirname(dir); dir = path.dirname(dir)) ancestors.push(dir);
  const rows = await db.job.findMany({
    where: { run_dir: { in: ancestors }, OR: [{ run_id: { not: null } }, { exit_verified_at: { not: null } }] },
    select: { job_type: true, name: true, run_dir: true, run_id: true, exit_verified_at: true },
  });
  for (const job of rows) {
    const folder = jobOutputFolder(trainingRoot, job);
    const relative = path.relative(folder, file);
    const dir = path.dirname(relative);
    const generated = job.job_type === 'inference' && dir.startsWith(`outputs${path.sep}`);
    const downloadable = dir === '.'
      ? /\.safetensors$/i.test(relative) || ['optimizer.pt', 'samples.zip'].includes(relative)
      : (dir === 'samples' || generated) && /\.(?:png|jpe?g|gif|webp|bmp|mp4|avi|mov|mkv|wmv|m4v|flv|mp3|wav|flac|ogg|txt)$/i.test(relative);
    if (downloadable && await regularJobPath(folder, file)) return folder;
  }
  return null;
};
