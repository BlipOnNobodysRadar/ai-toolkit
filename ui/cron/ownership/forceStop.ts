// Force stop (the kill route). Termination goes only through the run's ownership boundary (Windows: the run's job
// object, so no pid is ever signalled); a failed, partial or unsupported termination keeps the claims and says so.
// Releasing still needs the evidence the reconciler uses (here: the supervisor's exit receipt after termination).
import { ProcCtl } from './procctl';
import { readExitReceipt, runFiles } from './launch';
import { Db, Outcome, isLegacyOccupant, releaseRun, withRetry } from './store';

export const forceStop = async (db: Db, procctl: ProcCtl, jobId: string, waitForReceiptMs = 5000): Promise<Outcome> => {
  const job = await db.job.findUnique({ where: { id: jobId } });
  if (!job) return { status: 404, body: { error: 'Job not found' } };
  const claims = await db.runClaim.count({ where: { job_id: jobId, kind: 'run' } });
  if (claims === 0) {
    if (isLegacyOccupant(job)) {
      return {
        status: 409,
        body: {
          error:
            'This job was started before the run-ownership update. Choose Stop if it is still training, or Mark as Stopped if it has finished.',
          ownership: 'legacy',
        },
      };
    }
    return { status: 200, body: { ok: true, killed: false, info: 'No process is owned by this job.' } };
  }
  const runId = job.run_id!;
  await withRetry(() =>
    db.$transaction(async tx => {
      await tx.job.updateMany({
        where: { id: jobId, run_id: runId },
        data: { stop: true, after_exit: 'stopped', stop_requested_at: job.stop_requested_at ?? new Date(), info: 'Force stopping...' },
      });
      await tx.job.updateMany({ where: { id: jobId, run_id: runId, status: { in: ['starting', 'running'] } }, data: { status: 'stopping' } });
    }),
  );
  if (job.launch_phase !== 'attached' || !job.supervisor_identity) {
    return {
      status: 409,
      body: { error: 'The job is still starting and its process boundary is not confirmed yet. It will stop as soon as it runs; try Force stop again in a moment if needed.' },
    };
  }
  const t = await procctl.terminate(runId, job.supervisor_identity);
  if (!t.supported) {
    await db.job.updateMany({ where: { id: jobId, run_id: runId }, data: { ownership_note: `Force stop unavailable: ${t.error}` } });
    return { status: 501, body: { error: t.error, killed: false } };
  }
  if (!t.verified) {
    const error = t.error ?? `Force stop did not finish: ${t.remaining} process(es) of this job are still running.`;
    await db.job.updateMany({ where: { id: jobId, run_id: runId }, data: { ownership_note: error, info: 'Force stop failed' } });
    return { status: 500, body: { error, killed: false } };
  }
  // The kernel reported the job object empty. The supervisor now writes its exit receipt and exits; release on that
  // receipt (the worker's reconciler does the same if this wait runs out).
  const files = runFiles(job.run_dir!, runId);
  const deadline = Date.now() + waitForReceiptMs;
  while (Date.now() < deadline && !readExitReceipt(files.exit, runId, job.supervisor_identity)) {
    await new Promise(r => setTimeout(r, 50));
  }
  if (readExitReceipt(files.exit, runId, job.supervisor_identity)) {
    const out = await releaseRun(db, jobId, runId, { phase: 'attached', identity: job.pid_identity }, { kind: 'exited', evidence: 'force stopped; supervisor exit receipt' });
    if (out.released) await db.job.updateMany({ where: { id: jobId, run_id: runId }, data: { info: 'Job force stopped' } });
    return { status: 200, body: { ok: true, killed: true, status: out.status } };
  }
  return { status: 200, body: { ok: true, killed: true, info: 'Terminated; waiting for exit confirmation.' } };
};
