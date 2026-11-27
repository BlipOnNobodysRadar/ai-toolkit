// Force stop targets the recorded launch. Current supervisors confirm exit with a receipt; older
// launchers identify the trainer and its workers through their inherited job ID.
import { ProcCtl } from './procctl';
import { readExitReceipt, runFiles } from './launch';
import { Db, Outcome, clearLegacy, isLegacyOccupant, releaseRun, withRetry } from './store';
import { legacyIdentity, legacyProbe, legacyProcess } from './legacy';

export const forceStop = async (db: Db, procctl: ProcCtl, jobId: string, waitForReceiptMs = 5000): Promise<Outcome> => {
  const job = await db.job.findUnique({ where: { id: jobId } });
  if (!job) return { status: 404, body: { error: 'Job not found' } };
  const claims = await db.runClaim.count({ where: { job_id: jobId, kind: 'run' } });
  if (claims === 0) {
    if (isLegacyOccupant(job)) {
      const probe = await procctl.probe(legacyProbe([job]));
      if (legacyProcess(job, probe) === 'gone') {
        await clearLegacy(db, job, 'previous process exited');
        return { status: 200, body: { ok: true, killed: false } };
      }
      const identity = legacyIdentity(job, probe);
      if (!identity) return { status: 409, body: { error: 'Could not identify this previous run. If it has finished, choose Mark as Stopped.' } };
      const result = await procctl.terminateLegacy(jobId, job.name, identity);
      if (!result.verified) return { status: 409, body: { error: result.error || `Force stop did not finish: ${result.remaining} processes remain.` } };
      const current = await db.job.findUnique({ where: { id: jobId } });
      if (current && current.run_id == null && current.pid === job.pid) await clearLegacy(db, current, 'force stopped', true);
      return { status: 200, body: { ok: true, killed: true } };
    }
    return { status: 200, body: { ok: true, killed: false, info: 'No process is owned by this job.' } };
  }
  const runId = job.run_id!;
  if (job.launch_phase === 'attached' && job.run_dir && !job.after_exit) {
    const exit = readExitReceipt(runFiles(job.run_dir, runId).exit, runId, job.supervisor_identity);
    if (exit) {
      // The run finished before this click. Its real outcome takes precedence.
      const out = await releaseRun(db, jobId, runId, { phase: 'attached', identity: job.pid_identity },
        { kind: 'exited', exitCode: exit.exit_code, evidence: exit.evidence });
      return { status: 200, body: { ok: true, killed: false, status: out.status } };
    }
  }
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
  const files = runFiles(job.run_dir!, runId);
  // Another click or the graceful stop can finish while the UI is sending Force Stop.
  const hasExited = () => !!readExitReceipt(files.exit, runId, job.supervisor_identity);
  const t = hasExited() ? { supported: true, verified: true } : await procctl.terminate(runId, job.supervisor_identity, job.run_dir);
  if (!t.supported) {
    await db.job.updateMany({ where: { id: jobId, run_id: runId }, data: { ownership_note: `Force stop unavailable: ${t.error}` } });
    return { status: 501, body: { error: t.error, killed: false } };
  }
  if (!t.verified && !t.requested && !hasExited()) {
    const error = t.error ?? `Force stop did not finish: ${t.remaining} process(es) of this job are still running.`;
    await db.job.updateMany({ where: { id: jobId, run_id: runId }, data: { ownership_note: error, info: 'Force stop failed' } });
    return { status: 500, body: { error, killed: false } };
  }
  // POSIX supervisors acknowledge the request before the children have exited.
  const deadline = Date.now() + waitForReceiptMs;
  while (Date.now() < deadline && !hasExited()) {
    await new Promise(r => setTimeout(r, 50));
  }
  const exit = readExitReceipt(files.exit, runId, job.supervisor_identity);
  if (exit) {
    const out = await releaseRun(db, jobId, runId, { phase: 'attached', identity: job.pid_identity }, { kind: 'exited', exitCode: exit.exit_code, evidence: 'force stopped; supervisor exit receipt' });
    if (out.released && out.status === 'stopped') {
      await db.job.updateMany({ where: { id: jobId, run_id: runId, status: 'stopped' }, data: { info: 'Job force stopped' } });
    }
    return { status: 200, body: { ok: true, killed: true, status: out.status } };
  }
  return t.verified
    ? { status: 200, body: { ok: true, killed: true, info: 'Terminated; waiting for exit confirmation.' } }
    : { status: 202, body: { ok: true, killed: false, info: 'Force stop requested; waiting for the process to exit.' } };
};
