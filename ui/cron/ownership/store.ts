// Run ownership transitions (docs/STATE_MACHINES.md M10). Every decision that depends on who owns a GPU or a job
// folder is made inside one interactive SQLite transaction. With Prisma 6.3.1 such a transaction holds the database
// write lock from BEGIN, so transactions from different workers/routes run one after another (one that waits past
// maxWait gets a busy error and is retried) and each sees everything committed before it. RunClaim's primary key
// additionally makes a duplicate claim on one resource impossible even if a check were ever stale.
//
// Display status is not ownership. Python may write completed/error/stopped/queued while its process is still
// running, so start/edit/delete/mark-stopped check claims (and legacy occupancy), not just the status.
import { randomUUID } from 'crypto';
import fs from 'fs';
import path from 'path';
import { OpLockBusy, withOpLock } from './opLock';
import { writerLeasePath } from './writerLease';
import type { Job, Prisma, PrismaClient, RunClaim } from '@prisma/client';
import {
  FolderIdentityError,
  ResourceSet,
  foldersOverlap,
  canonicalFolder,
  claimKeys,
  jobFolderError,
  overlap,
  parseGpuIds,
  resourceSet,
  setFromClaims,
} from './resources';

export type Db = PrismaClient;
type Tx = Prisma.TransactionClient;
export type Outcome = { status: number; body: any };

export const ACTIVE = ['starting', 'running', 'stopping'];
export const DELETING = ['deleting', 'delete_failed'];
export const EDITABLE = ['stopped', 'completed', 'error', 'queued'];
export const DELETABLE = ['stopped', 'completed', 'error', 'queued', 'delete_failed'];
export const deleteRunId = (jobId: string) => `delete:${jobId}`;

// ---------------------------------------------------------------- retry
class LostRace extends Error {}

const isBusy = (e: any) =>
  e instanceof LostRace ||
  ['P2034', 'P1008', 'P2028'].includes(e?.code) ||
  /database is locked|SQLITE_BUSY|busy|write conflict|deadlock/i.test(String(e?.message ?? ''));

// Retries a transaction that lost a race or hit a locked database. After the last attempt a busy error becomes a
// 503 outcome for routes (callers that need an exception pass rethrow=true).
export const withRetry = async <T>(fn: () => Promise<T>, attempts = 8): Promise<T> => {
  let last: any;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (e: any) {
      if (!isBusy(e)) throw e;
      last = e;
      await new Promise(r => setTimeout(r, 5 + Math.floor(Math.random() * 20) * (i + 1)));
    }
  }
  throw last;
};

const busyOutcome = (e: any): Outcome => ({
  status: 503,
  body: { error: 'The job database is busy; nothing was changed. Try again.', detail: String(e?.message ?? e) },
});

const routeTx = async (db: Db, fn: (tx: Tx) => Promise<Outcome>): Promise<Outcome> => {
  try {
    return await withRetry(() => db.$transaction(tx => fn(tx), { maxWait: 5000, timeout: 10000 }));
  } catch (e: any) {
    if (isBusy(e)) return busyOutcome(e);
    if (e instanceof FolderIdentityError) return { status: 500, body: { error: e.message } };
    throw e;
  }
};

// ---------------------------------------------------------------- ownership predicates
// A row last launched before run ownership existed (run_id null) still owns its GPUs and folder while its process
// may be alive: any active status, or any recorded pid (the original code never cleared pid on a normal exit).
// The reconciler clears it only on proof of exit.
export const isLegacyOccupant = (job: Pick<Job, 'run_id' | 'status' | 'pid'>) =>
  job.run_id == null && (ACTIVE.includes(job.status) || job.pid != null);

export type Occupant = {
  job_id: string;
  job_name: string;
  run_id: string | null;
  kind: 'run' | 'delete' | 'legacy';
  status: string;
  set: ResourceSet;
};

export const occupants = async (tx: Tx, trainingRoot: string): Promise<Occupant[]> => {
  const claims: RunClaim[] = await tx.runClaim.findMany();
  const byRun = new Map<string, RunClaim[]>();
  for (const c of claims) byRun.set(c.run_id, [...(byRun.get(c.run_id) ?? []), c]);
  const legacy = await tx.job.findMany({
    where: { run_id: null, OR: [{ status: { in: ACTIVE } }, { pid: { not: null } }] },
  });
  const ids = [...new Set(claims.map(c => c.job_id))];
  const named = ids.length ? await tx.job.findMany({ where: { id: { in: ids } }, select: { id: true, name: true, status: true } }) : [];
  const info = new Map(named.map(j => [j.id, j]));
  const out: Occupant[] = [];
  for (const [run_id, rows] of byRun) {
    const j = info.get(rows[0].job_id);
    out.push({
      job_id: rows[0].job_id,
      job_name: j?.name ?? rows[0].job_id,
      run_id,
      kind: rows[0].kind === 'delete' ? 'delete' : 'run',
      status: j?.status ?? 'missing',
      set: setFromClaims(rows.map(r => r.resource)),
    });
  }
  for (const j of legacy) {
    const gpu = parseGpuIds(j.gpu_ids, true);
    out.push({
      job_id: j.id,
      job_name: j.name,
      run_id: null,
      kind: 'legacy',
      status: j.status,
      set: resourceSet(gpu.ok ? gpu : { gpus: [], exclusive: true }, canonicalFolder(trainingRoot, j.name)),
    });
  }
  return out;
};

const describeOccupant = (o: Occupant) =>
  o.kind === 'delete'
    ? `job "${o.job_name}" (being deleted)`
    : o.kind === 'legacy'
      ? `job "${o.job_name}" (${o.status}; started before run ownership, process exit not yet verified)`
      : `job "${o.job_name}" (${o.status}${ACTIVE.includes(o.status) ? '' : ', process still exiting'})`;

export const firstConflict = (occ: Occupant[], set: ResourceSet, exceptJobId?: string) => {
  for (const o of occ) {
    if (o.job_id === exceptJobId && o.kind !== 'delete') continue;
    const shared = overlap(set, o.set);
    if (shared.length) return { occupant: o, shared, message: `${shared.join(' and ')} in use by ${describeOccupant(o)}` };
  }
  return null;
};

export type Ownership = { held: boolean; kind?: 'run' | 'legacy'; message?: string };

export const ownershipOf = async (tx: Tx, job: Job): Promise<Ownership> => {
  const claims = await tx.runClaim.count({ where: { job_id: job.id, kind: 'run' } });
  if (claims > 0) {
    const finished = !ACTIVE.includes(job.status);
    return {
      held: true,
      kind: 'run',
      message: finished
        ? `The job reported "${job.status}" but its process has not exited yet; this is available once the exit is verified.`
        : `The job is ${job.status}; stop it and wait for its process to exit first.`,
    };
  }
  if (isLegacyOccupant(job)) {
    return {
      held: true,
      kind: 'legacy',
      message:
        job.pid != null
          ? `This job's process (pid ${job.pid}, started before the run-ownership update) has not been verified as exited yet.`
          : `This job is marked ${job.status} from before the run-ownership update and its process has not been ruled out yet.`,
    };
  }
  return { held: false };
};

const conflict = (error: string, extra: any = {}): Outcome => ({ status: 409, body: { error, ...extra } });

// ---------------------------------------------------------------- route transitions
export const requestStart = (db: Db, jobId: string, trainingRoot: string): Promise<Outcome> =>
  routeTx(db, async tx => {
    const job = await tx.job.findUnique({ where: { id: jobId } });
    if (!job) return { status: 404, body: { error: 'Job not found' } };
    if (DELETING.includes(job.status)) return conflict('This job is being deleted.');
    if (job.status === 'queued') return { status: 200, body: job }; // a repeated Start is a no-op
    if (ACTIVE.includes(job.status)) return conflict(`The job is already ${job.status}.`);
    const own = await ownershipOf(tx, job);
    if (own.held) return conflict(own.message!, { ownership: own.kind });
    const gpu = parseGpuIds(job.gpu_ids);
    if (gpu.ok === false) return { status: 400, body: { error: gpu.error } };
    const nameError = jobFolderError(trainingRoot, job.name);
    if (nameError) return { status: 400, body: { error: nameError } };
    const top = await tx.job.aggregate({ _max: { queue_position: true } });
    if (!(await tx.queue.findUnique({ where: { gpu_ids: job.gpu_ids } }))) {
      await tx.queue.create({ data: { gpu_ids: job.gpu_ids, is_running: false } });
    }
    const r = await tx.job.updateMany({
      where: { id: jobId, status: job.status, run_id: job.run_id },
      data: {
        status: 'queued',
        queue_position: (top._max.queue_position || 0) + 1000,
        stop: false,
        return_to_queue: false,
        after_exit: null,
        info: 'Job queued',
        ownership_note: null,
      },
    });
    if (r.count !== 1) throw new LostRace();
    return { status: 200, body: job };
  });

// Stop is a request. It never releases ownership: claims stay until the process exit is verified.
export const requestStop = (db: Db, jobId: string): Promise<Outcome> =>
  routeTx(db, async tx => {
    const job = await tx.job.findUnique({ where: { id: jobId } });
    if (!job) return { status: 404, body: { error: 'Job not found' } };
    const own = await ownershipOf(tx, job);
    if (!own.held) {
      if (job.status === 'queued') {
        await tx.job.updateMany({
          where: { id: jobId, status: 'queued' },
          data: { status: 'stopped', info: 'Removed from queue' },
        });
      }
      return { status: 200, body: { ...job, stopping: false } };
    }
    const data: any = { stop: true, info: 'Stopping job...', ...keepLegacyTime(job) };
    if (own.kind === 'run') {
      data.stop_requested_at = job.stop_requested_at ?? new Date();
      data.after_exit = 'stopped';
    }
    const r = await tx.job.updateMany({ where: { id: jobId, run_id: job.run_id }, data });
    if (r.count !== 1) throw new LostRace();
    // status shows the request, but never overwrites an ending Python already wrote
    await tx.job.updateMany({
      where: { id: jobId, run_id: job.run_id, status: { in: ['starting', 'running'] } },
      data: { status: 'stopping', ...keepLegacyTime(job) },
    });
    return { status: 200, body: { ...job, stopping: true } };
  });

// Queue pause: only jobs of that queue (exact gpu_ids), and only while active. The run returns to 'queued' once its
// exit is verified.
export const requestQueueReturn = async (db: Db, job: Job) => {
  await withRetry(() =>
    db.$transaction(async tx => {
      // the queue may have been started again after the tick read it as paused
      const queue = await tx.queue.findUnique({ where: { gpu_ids: job.gpu_ids } });
      if (queue?.is_running) return;
      if (job.run_id == null) {
        // legacy process: it reads return_to_queue itself, as before
        await tx.job.updateMany({
          where: { id: job.id, run_id: null, status: 'running', return_to_queue: false },
          data: { return_to_queue: true, info: 'Stopping job...', updated_at: job.updated_at },
        });
        return;
      }
      await tx.job.updateMany({
        where: { id: job.id, run_id: job.run_id, status: { in: ['starting', 'running'] } },
        data: {
          return_to_queue: true,
          after_exit: 'queued',
          stop_requested_at: job.stop_requested_at ?? new Date(),
          status: 'stopping',
          info: 'Stopping job (queue paused)...',
        },
      });
    }),
  );
};

// Pure display reset for a row that owns nothing. Never an ownership bypass.
export const markStopped = (db: Db, jobId: string): Promise<Outcome> =>
  routeTx(db, async tx => {
    const job = await tx.job.findUnique({ where: { id: jobId } });
    if (!job) return { status: 404, body: { error: 'Job not found' } };
    if (DELETING.includes(job.status)) return conflict('This job is being deleted.');
    const own = await ownershipOf(tx, job);
    if (own.held) return conflict(`Not marked as stopped: ${own.message}`, { ownership: own.kind });
    await tx.job.updateMany({
      where: { id: jobId, run_id: job.run_id, status: job.status },
      data: { stop: true, status: 'stopped', info: 'Job stopped', pid: null },
    });
    return { status: 200, body: job };
  });

// Create (no id) or edit. Names are compared by canonical folder, so 'Foo' and 'foo' collide where the file system
// would put them in one directory, and a folder reserved by a deletion cannot be reused until it is gone.
export const saveJob = (db: Db, body: any, trainingRoot: string, forcedGpuIds?: string): Promise<Outcome> =>
  routeTx(db, async tx => {
    const { id, name, job_config } = body;
    const gpu_ids: string = forcedGpuIds ?? body.gpu_ids;
    const nameError = jobFolderError(trainingRoot, name);
    if (nameError) return { status: 400, body: { error: nameError } };
    const gpu = parseGpuIds(gpu_ids);
    if (gpu.ok === false) return { status: 400, body: { error: gpu.error } };
    const extra: any = {};
    if ('job_ref' in body) extra.job_ref = body.job_ref;
    if ('job_type' in body) extra.job_type = body.job_type;
    const folder = canonicalFolder(trainingRoot, name);

    let existing: Job | null = null;
    if (id) {
      existing = await tx.job.findUnique({ where: { id } });
      if (!existing) return { status: 404, body: { error: 'Job not found' } };
      if (!EDITABLE.includes(existing.status)) return conflict(`A ${existing.status} job cannot be edited.`);
      const own = await ownershipOf(tx, existing);
      if (own.held) return conflict(`Not saved: ${own.message}`, { ownership: own.kind });
    }
    const others = await tx.job.findMany({ select: { id: true, name: true } });
    const alias = others.find(o => o.id !== id && canonicalFolder(trainingRoot, o.name) === folder);
    if (alias) {
      return conflict(alias.name === name ? 'Job name already exists' : `Job name conflicts with existing job "${alias.name}" (same folder)`);
    }
    const holder = (await occupants(tx, trainingRoot)).find(o => o.job_id !== id && o.kind !== 'legacy' && foldersOverlap(o.set.folder, folder));
    if (holder) {
      return conflict(holder.kind === 'delete' ? 'A job with this name (or its folder) is still being deleted.' : 'This job folder is in use by another job.');
    }

    if (existing) {
      const r = await tx.job.updateMany({
        where: { id, status: existing.status, run_id: existing.run_id },
        data: { name, gpu_ids, job_config: JSON.stringify(job_config), ...extra },
      });
      if (r.count !== 1) throw new LostRace();
      return { status: 200, body: await tx.job.findUnique({ where: { id } }) };
    }
    const top = await tx.job.aggregate({ _max: { queue_position: true } });
    const created = await tx.job.create({
      data: {
        name,
        gpu_ids,
        job_config: JSON.stringify(job_config),
        queue_position: (top._max.queue_position || 0) + 1000,
        ...extra,
      },
    });
    return { status: 200, body: created };
  });

// Delete, step 1: claim-free -> deleting, and reserve the canonical folder so nothing can be created in it or
// admitted to it until removal finishes. The row (and its name) stay until the folder is gone.
// resume=true is only for a caller holding the job's deletion lock (deleteJob): a row left 'deleting' by a deleter
// that crashed, or whose final DB write failed, is then safely continued.
export const beginDelete = (db: Db, jobId: string, trainingRoot: string, opts: { resume?: boolean; expectedFolder?: string } = {}): Promise<Outcome> =>
  routeTx(db, async tx => {
    const job = await tx.job.findUnique({ where: { id: jobId } });
    if (!job) return { status: 404, body: { error: 'Job not found' } };
    if (job.status === 'deleting' && !opts.resume) return conflict('This job is already being deleted.');
    if (job.status !== 'deleting' && !DELETABLE.includes(job.status)) return conflict(`A ${job.status} job cannot be deleted; stop it first.`);
    const own = await ownershipOf(tx, job);
    if (own.held) return conflict(`Not deleted: ${own.message}`, { ownership: own.kind });
    const nameError = jobFolderError(trainingRoot, job.name);
    if (nameError) return { status: 400, body: { error: `${nameError}; its folder is not removed from here.` } };
    const folder = canonicalFolder(trainingRoot, job.name);
    if (opts.expectedFolder !== undefined && folder !== opts.expectedFolder) {
      return conflict('The job folder changed before deletion was reserved; retry Delete.');
    }
    const resource = `folder:${folder}`;
    const holder = await tx.runClaim.findUnique({ where: { resource } });
    // any other owner of this folder, of a folder inside it, or of one containing it (recursive removal would hit it)
    const other = (await occupants(tx, trainingRoot)).find(o => o.job_id !== jobId && foldersOverlap(o.set.folder, folder));
    if (other) return conflict(`Another job ("${other.job_name}") is using this folder or a folder inside it.`);
    if (!holder) {
      await tx.runClaim.create({ data: { resource, job_id: jobId, run_id: deleteRunId(jobId), kind: 'delete' } });
    }
    const r = await tx.job.updateMany({
      where: { id: jobId, status: job.status, run_id: job.run_id },
      data: { status: 'deleting', info: 'Deleting job...' },
    });
    if (r.count !== 1) throw new LostRace();
    return { status: 200, body: job };
  });

// Delete, whole operation. Serialized per job by a crash-released lock kept OUTSIDE the folder being removed
// (<training root>/.aitk_locks): begin (or resume), remove the folder, finish. The job DB transaction is never held
// across the removal. A process that dies anywhere in between leaves 'deleting' + the folder reservation, and the
// next Delete resumes it; a Delete while another one is running gets 409.
export const LOCK_DIR = '.aitk_locks';
export const deleteJob = async (
  db: Db,
  jobId: string,
  trainingRoot: string,
  deps: { rm?: (dir: string) => Promise<void>; afterBegin?: () => Promise<void>; waitMs?: number } = {},
): Promise<Outcome> => {
  const lockFile = path.join(trainingRoot, LOCK_DIR, `delete.${jobId.replace(/[^\w-]/g, '_')}.lock`);
  try {
    return await withOpLock(
      lockFile,
      async () => {
        const candidate = await db.job.findUnique({ where: { id: jobId } });
        if (!candidate) return { status: 404, body: { error: 'Job not found' } };
        const invalid = jobFolderError(trainingRoot, candidate.name);
        if (invalid) return { status: 400, body: { error: invalid } };
        const expectedFolder = canonicalFolder(trainingRoot, candidate.name);
        try {
          return await withOpLock(writerLeasePath(expectedFolder), async () => {
            const begun = await beginDelete(db, jobId, trainingRoot, { resume: true, expectedFolder });
            if (begun.status !== 200) return begun;
            if (deps.afterBegin) await deps.afterBegin();
            const folder = path.join(trainingRoot, begun.body.name);
            let error: string | null = null;
            try {
              // force:true makes this a no-op if the folder is already gone (a resumed deletion)
              await (deps.rm ?? (dir => fs.promises.rm(dir, { recursive: true, force: true })))(folder);
            } catch (e: any) {
              error = e?.message || String(e);
            }
            await finishDelete(db, jobId, error); // a DB failure here throws: still 'deleting', resumable
            if (error) return { status: 500, body: { error: `Could not delete the job folder: ${error}` } };
            return { status: 200, body: begun.body };
          });
        } catch (e) {
          if (e instanceof OpLockBusy) return conflict('A training run or another deletion owns this output folder; stop it or retry after it finishes.');
          throw e;
        }
      },
      deps.waitMs ?? 0,
    );
  } catch (e: any) {
    if (e instanceof OpLockBusy) return conflict('This job is being deleted by another request; try again once it has finished.');
    // e.g. the final DB write failed: the row stays 'deleting' (folder still reserved) and Delete can be retried
    return { status: 500, body: { error: `Deletion not finished (${e?.message ?? e}); retry Delete to complete it.` } };
  }
};

// Delete, step 2 (after the folder removal attempt).
export const finishDelete = async (db: Db, jobId: string, error: string | null) => {
  await withRetry(() =>
    db.$transaction(async tx => {
      if (error === null) {
        await tx.runClaim.deleteMany({ where: { run_id: deleteRunId(jobId) } });
        await tx.job.deleteMany({ where: { id: jobId, status: 'deleting' } });
      } else {
        await tx.job.updateMany({
          where: { id: jobId, status: 'deleting' },
          data: { status: 'delete_failed', info: `Could not delete the job folder: ${error}` },
        });
      }
    }),
  );
};

// ---------------------------------------------------------------- admission (worker)
export type Launcher = { pid: number; identity: string; boot_id?: string | null };

export type Admission = { admitted: Job | null; reason?: string };

export const admit = async (db: Db, jobId: string, trainingRoot: string, launcher: Launcher, hooks?: { afterRead?: (tx: Tx) => Promise<void>; requireQueueRunning?: boolean }): Promise<Admission> => {
  try {
    return await withRetry(() =>
      db.$transaction(
        async tx => {
          const job = await tx.job.findUnique({ where: { id: jobId } });
          if (!job || job.status !== 'queued') return { admitted: null, reason: 'not queued' };
          if (hooks?.requireQueueRunning) {
            // the queue may have been paused after the tick read it: decide on its committed state, here
            const queue = await tx.queue.findUnique({ where: { gpu_ids: job.gpu_ids } });
            if (!queue?.is_running) return { admitted: null, reason: 'queue not running' };
          }
          const gpu = parseGpuIds(job.gpu_ids);
          const nameError = jobFolderError(trainingRoot, job.name);
          if (gpu.ok === false || nameError) {
            const message = gpu.ok === false ? gpu.error : nameError!;
            await tx.job.updateMany({ where: { id: jobId, status: 'queued' }, data: { status: 'error', info: message } });
            return { admitted: null, reason: message };
          }
          const folderAbs = path.resolve(trainingRoot, job.name);
          const set = resourceSet(gpu, canonicalFolder(trainingRoot, job.name));
          const occ = await occupants(tx, trainingRoot);
          if (hooks?.afterRead) await hooks.afterRead(tx);
          const c = firstConflict(occ, set);
          if (c) {
            const note = `Waiting: ${c.message}`;
            if (job.ownership_note !== note) {
              await tx.job.updateMany({ where: { id: jobId, status: 'queued' }, data: { ownership_note: note } });
            }
            return { admitted: null, reason: note };
          }
          const runId = randomUUID();
          await tx.runClaim.createMany({
            data: claimKeys(set).map(resource => ({ resource, job_id: jobId, run_id: runId, kind: 'run' })),
          });
          const r = await tx.job.updateMany({
            where: { id: jobId, status: 'queued' },
            data: {
              status: 'starting',
              run_id: runId,
              launch_phase: 'claimed',
              launcher_pid: launcher.pid,
              launcher_identity: launcher.identity,
              boot_id: launcher.boot_id ?? null,
              spawn_requested_at: null,
              relay_pid: null,
              relay_exited_at: null,
              pid: null,
              pid_identity: null,
              run_members: null,
              stop: false,
              return_to_queue: false,
              stop_requested_at: null,
              after_exit: null,
              exit_verified_at: null,
              run_dir: folderAbs,
              info: 'Starting job...',
              ownership_note: null,
            },
          });
          if (r.count !== 1) throw new LostRace();
          return { admitted: await tx.job.findUnique({ where: { id: jobId } }) };
        },
        { maxWait: 5000, timeout: 10000 },
      ),
    );
  } catch (e: any) {
    // a duplicate claim (P2002) or a database that stayed busy: not admitted now; the next tick decides again
    if (e?.code === 'P2002' || isBusy(e)) return { admitted: null, reason: `not admitted (${e?.code ?? 'busy'})` };
    if (e instanceof FolderIdentityError) {
      // the folder's real identity is unknown, so its claim key would be a guess: stay queued and say why
      await db.job.updateMany({ where: { id: jobId, status: 'queued' }, data: { ownership_note: `Waiting: ${e.message}` } }).catch(() => {});
      return { admitted: null, reason: e.message };
    }
    throw e;
  }
};

// A running queue with nothing left to do is switched off, but only if, at commit time, it is still running and
// none of its jobs is queued or may still own a process: a job queued and the queue started after the tick's read,
// or a run another worker admitted meanwhile, keep it running (a stopped queue would pause that run next tick).
export const stopIdleQueue = (db: Db, queueId: number, gpuIds: string) =>
  withRetry(() =>
    db.$transaction(async tx => {
      const busy = await tx.job.count({
        where: {
          gpu_ids: gpuIds,
          OR: [
            { status: { in: ['queued', ...ACTIVE] } },
            { run_id: { not: null }, launch_phase: { in: ['claimed', 'spawn_requested', 'attached'] } },
            { run_id: null, pid: { not: null } },
          ],
        },
      });
      if (busy > 0) return false;
      const r = await tx.queue.updateMany({ where: { id: queueId, is_running: true }, data: { is_running: false } });
      return r.count === 1;
    }),
  );

// ---------------------------------------------------------------- launch phases (worker); all keyed on run_id
export const markSpawnRequested = async (db: Db, jobId: string, runId: string) =>
  (
    await withRetry(() =>
      db.job.updateMany({
        where: { id: jobId, run_id: runId, launch_phase: 'claimed', stop: false },
        data: { launch_phase: 'spawn_requested', spawn_requested_at: new Date() },
      }),
    )
  ).count === 1;

export const recordRelay = (db: Db, jobId: string, runId: string, relayPid: number) =>
  withRetry(() => db.job.updateMany({ where: { id: jobId, run_id: runId }, data: { relay_pid: relayPid } }));

export const markRelayExited = (db: Db, jobId: string, runId: string) =>
  withRetry(() =>
    db.job.updateMany({ where: { id: jobId, run_id: runId, relay_exited_at: null }, data: { relay_exited_at: new Date() } }),
  );

// Process identity is attached (from the spawn receipt only) without overwriting an ending Python may already have
// written.
export type SpawnReceipt = { identity: string; pid?: number; supervisor?: string | null; boundary?: string | null; boot_id?: string | null };
export const attachRun = async (db: Db, jobId: string, runId: string, receipt: SpawnReceipt) =>
  withRetry(() =>
    db.$transaction(async tx => {
      const pid = Number(receipt.identity.split('@')[0]);
      const r = await tx.job.updateMany({
        where: { id: jobId, run_id: runId, launch_phase: 'spawn_requested' },
        data: {
          launch_phase: 'attached',
          pid,
          pid_identity: receipt.identity,
          supervisor_identity: receipt.supervisor ?? null,
          run_boundary: receipt.boundary ?? null,
          ...(receipt.boot_id ? { boot_id: receipt.boot_id } : {}),
          run_members: JSON.stringify([receipt.identity]),
        },
      });
      if (r.count === 1) {
        await tx.job.updateMany({
          where: { id: jobId, run_id: runId, status: 'starting' },
          data: { status: 'running', info: 'Job running' },
        });
      }
      return r.count === 1;
    }),
  );

export const updateMembers = (db: Db, jobId: string, runId: string, members: string[]) =>
  withRetry(() => db.job.updateMany({ where: { id: jobId, run_id: runId }, data: { run_members: JSON.stringify(members) } }));

// For a legacy row updated_at is evidence (the old code's last write, an upper bound for its process start time),
// so writes to legacy rows keep it unchanged.
export const keepLegacyTime = (job: Pick<Job, 'run_id' | 'updated_at'>) => (job.run_id == null ? { updated_at: job.updated_at } : {});

export const noteOwnership = async (db: Db, job: Job, note: string | null) => {
  if (job.ownership_note === note) return;
  await withRetry(() =>
    db.job.updateMany({ where: { id: job.id, run_id: job.run_id }, data: { ownership_note: note, ...keepLegacyTime(job) } }),
  );
};

// The final status once the run's processes are verified gone. Python's own ending (completed/error) stands; a
// requested stop/queue return applies otherwise; an active status with no ending means the process died.
export const finalStatus = (job: Job): { status: string; unexpected: boolean } => {
  if (job.status === 'completed' || job.status === 'error') return { status: job.status, unexpected: false };
  if (job.after_exit === 'stopped' || job.after_exit === 'queued') return { status: job.after_exit, unexpected: false };
  if (job.status === 'stopped' || job.status === 'queued') return { status: job.status, unexpected: false };
  return { status: 'error', unexpected: true };
};

// Releases a run's claims: only for verified exit, or for a launch known never to have created a process. The
// evidence was gathered for one launch phase (and, when attached, one process identity); the transaction re-checks
// both, so evidence about an earlier phase can never release a run that has moved on (e.g. 'claimed' evidence
// arriving after the launcher committed spawn_requested and spawned).
export type ReleaseOutcome = { kind: 'exited'; unexpectedInfo?: string; evidence?: string } | { kind: 'not_spawned'; info: string };
export const releaseRun = async (
  db: Db,
  jobId: string,
  runId: string,
  expect: { phase: 'claimed' | 'spawn_requested' | 'attached'; identity?: string | null },
  outcome: ReleaseOutcome,
): Promise<{ released: boolean; status?: string; unexpected?: boolean; reason?: string }> =>
  withRetry(() =>
    db.$transaction(async tx => {
      const job = await tx.job.findUnique({ where: { id: jobId } });
      if (!job || job.run_id !== runId) return { released: false, reason: 'run superseded' };
      if (job.launch_phase !== expect.phase) return { released: false, reason: `phase is ${job.launch_phase}, evidence was for ${expect.phase}` };
      if (expect.phase === 'attached' && expect.identity !== undefined && job.pid_identity !== expect.identity) {
        return { released: false, reason: 'process identity changed' };
      }
      await tx.runClaim.deleteMany({ where: { run_id: runId, kind: 'run' } });
      let status: string;
      let info = job.info;
      let unexpected = false;
      if (outcome.kind === 'not_spawned') {
        status = job.stop && job.after_exit ? job.after_exit : 'error';
        info = outcome.info;
      } else {
        const f = finalStatus(job);
        status = f.status;
        unexpected = f.unexpected;
        if (unexpected) info = outcome.unexpectedInfo ?? 'Job process exited unexpectedly.';
      }
      await tx.job.updateMany({
        where: { id: jobId, run_id: runId, launch_phase: expect.phase },
        data: {
          status,
          info,
          launch_phase: 'exited',
          exit_verified_at: new Date(),
          pid: null,
          ownership_note: outcome.kind === 'exited' && outcome.evidence ? `Exit verified: ${outcome.evidence}` : null,
        },
      });
      return { released: true, status, unexpected };
    }),
  );

// Claims left by a run id that no job row carries any more (not produced by this code; kept for manual repair).
// Legacy rows: cleared only on proof of exit (never signalled).
export const clearLegacy = (db: Db, job: Job, evidence: string) =>
  withRetry(() =>
    db.job.updateMany({
      where: { id: job.id, run_id: null, pid: job.pid, status: job.status, updated_at: job.updated_at },
      data: {
        pid: null,
        ...(ACTIVE.includes(job.status)
          ? { status: job.stop ? 'stopped' : 'error', info: job.stop ? 'Job stopped' : `Job process exited (${evidence}).` }
          : {}),
        ownership_note: `Legacy process exit verified: ${evidence}`,
      },
    }),
  );
