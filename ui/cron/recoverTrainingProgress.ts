import { createReadStream, promises as fs } from 'fs';
import path from 'path';
import { createInterface } from 'readline';
import type { PrismaClient } from '@prisma/client';
import { jobArtifactPath } from './ownership/jobArtifacts';

// Older UI trainers replaced Job.step with the last save's filename step on stop/error. The trainer's
// own named tqdm bar records completed iterations, including those after the last save. Logs rotate
// per launch. A different job, a sampling bar or a config's target step is not evidence of progress.
async function lastReportedStep(file: string, jobId: string) {
  // A row can be renamed after stopping. The training bar used the original folder name.
  const prefix = `${path.basename(path.dirname(file))}: `;
  const stream = createReadStream(file);
  const lines = createInterface({ input: stream, crlfDelay: Infinity });
  let ours = false;
  let step = 0;
  try {
    for await (const line of lines) {
      if (line.startsWith('Job ID: "')) {
        ours = line.trimEnd() === `Job ID: "${jobId}"`;
        step = 0;
      } else if (/^Process \d+\/\d+ · /.test(line)) {
        step = 0;
      } else if (ours && line.startsWith(prefix)) {
        const progress = /^\s*\d+%\|[^|]*\|\s*(\d+)\/(\d+)\s*\[/.exec(line.slice(prefix.length));
        if (progress && Number(progress[1]) <= Number(progress[2])) step = Number(progress[1]);
      }
    }
  } finally {
    lines.close();
    stream.destroy();
  }
  return step;
}

export async function recoverTrainingProgress(db: PrismaClient, trainingRoot: string) {
  const jobs = await db.job.findMany({
    where: {
      job_type: 'train', status: { in: ['stopped', 'error'] }, pid: null,
      OR: [{ launch_phase: null }, { launch_phase: 'exited' }],
    },
    select: { id: true, name: true, run_dir: true, run_id: true, status: true, step: true, updated_at: true, exit_verified_at: true },
  });
  for (const job of jobs) {
    try {
      const file = await jobArtifactPath(trainingRoot, job, 'log.txt');
      if (!(await fs.stat(file)).isFile()) continue;
      const step = await lastReportedStep(file, job.id);
      if (step <= job.step) continue;
      // A concurrent restart, edit, deletion or new step wins. Keep the job's original history ordering.
      await db.job.updateMany({
        where: { id: job.id, run_id: job.run_id, status: job.status, pid: null, step: job.step, updated_at: job.updated_at },
        data: { step, updated_at: job.updated_at },
      });
    } catch (e: any) {
      if (e?.code !== 'ENOENT') console.warn(`Could not recover training progress for ${job.name}:`, e);
    }
  }
}
