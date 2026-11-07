import prisma from '../prisma';

import { Job, Queue } from '@prisma/client';
import startJob, { launchesInFlight } from './startJob';
import { Db, Launcher, admit, isLegacyOccupant, requestQueueReturn, stopIdleQueue } from '../ownership/store';
import { getTrainingFolder } from '../paths';

export type QueueContext = {
  db: Db;
  trainingRoot: string;
  launcher: Launcher;
  launch: (job: Job) => void;
};

// One queue tick. Admission is the atomic claim in ownership/store.ts: a queued job starts only when its whole
// normalized GPU set and its folder are free of every other run's claims, deletion reservations and legacy
// occupancy, across all queues and workers.
export async function processQueueWith(ctx: QueueContext) {
  const { db } = ctx;
  const queues: Queue[] = await db.queue.findMany({
    orderBy: {
      id: 'asc',
    },
  });

  for (const queue of queues) {
    // the jobs of this queue that may still have a process (claims are checked at admission; this is the queue's
    // own view, so a paused queue only ever touches its own jobs)
    const ownJobs: Job[] = await db.job.findMany({
      where: {
        gpu_ids: queue.gpu_ids,
        OR: [{ status: { in: ['starting', 'running', 'stopping'] } }, { launch_phase: { in: ['claimed', 'spawn_requested', 'attached'] } }, { run_id: null, pid: { not: null } }],
      },
    });
    const active = ownJobs.filter(j => (j.run_id ? j.launch_phase !== 'exited' : isLegacyOccupant(j)));

    if (!queue.is_running) {
      // pause: active runs of this queue go back to 'queued' once their process has exited
      for (const job of active) {
        if (['starting', 'running'].includes(job.status) && !job.return_to_queue) {
          console.log(`Stopping job ${job.id} on GPU(s) ${job.gpu_ids}`);
          await requestQueueReturn(db, job);
        }
      }
      continue;
    }

    if (active.length) {
      // already running (or still exiting), nothing to do
      continue;
    }
    const nextJob: Job | null = await db.job.findFirst({
      where: {
        status: 'queued',
        gpu_ids: queue.gpu_ids,
      },
      orderBy: {
        queue_position: 'asc',
      },
    });
    if (nextJob) {
      const { admitted, reason } = await admit(db, nextJob.id, ctx.trainingRoot, ctx.launcher, { requireQueueRunning: true });
      if (admitted) {
        console.log(`Starting job ${admitted.id} on GPU(s) ${admitted.gpu_ids} (run ${admitted.run_id})`);
        ctx.launch(admitted);
      } else if (reason && reason !== 'not queued') {
        // stays queued; the reason is on the row (ownership_note) for the UI
      }
    } else {
      // no more jobs, stop the queue
      if (await stopIdleQueue(db, queue.id, queue.gpu_ids)) {
        console.log(`No more jobs in queue for GPU(s) ${queue.gpu_ids}, stopped queue`);
      }
    }
  }
}

export default async function processQueue(launcher: Launcher) {
  await processQueueWith({
    db: prisma,
    trainingRoot: await getTrainingFolder(),
    launcher,
    launch: job => {
      // registered synchronously, before any await: the reconciler treats an unregistered 'claimed' run of this
      // worker as abandoned before spawn
      launchesInFlight.add(job.run_id!);
      // starts and watches the job asynchronously so the cron can continue
      void startJob(job).catch(e => console.error(`Launch of job ${job.id} failed:`, e));
    },
  });
}
