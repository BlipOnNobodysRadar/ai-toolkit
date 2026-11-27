// The owner confirmed this job has ended. Refuse if we can identify a live
// trainer or supervisor; never signal a PID as part of a bookkeeping repair.
import { legacyProbe, legacyProcess } from './legacy';
import { knownMembers, readExitReceipt, runFiles, validIdentity } from './launch';
import type { ProcCtl } from './procctl';
import { Db, Outcome, RunSnapshot, isLegacyOccupant, sameRun } from './store';

export const isRunSnapshot = (v: any): v is RunSnapshot =>
  v != null && typeof v.status === 'string' &&
  (v.run_id === null || typeof v.run_id === 'string') &&
  (v.launch_phase === null || typeof v.launch_phase === 'string') &&
  (v.pid === null || Number.isSafeInteger(v.pid));

export async function checkStopped(db: Db, pc: ProcCtl, id: string, seen: RunSnapshot, action: 'mark' | 'delete' = 'mark'): Promise<Outcome> {
  const job = await db.job.findUnique({ where: { id } });
  if (!job) return { status: 404, body: { error: 'Job not found' } };
  if (!sameRun(job, seen)) return { status: 409, body: { error: 'The job changed. Refresh and try again.' } };
  const active = { status: 409, body: { error: 'This job still has a running process. Use Stop first.' } };
  const unknown = { status: 409, body: { error: 'Could not verify process exit. If you have checked that training has ended, choose Mark as Stopped, then Delete.' } };
  if (isLegacyOccupant(job)) {
    const result = await pc.probe(legacyProbe([job]));
    const state = legacyProcess(job, result);
    if (job.boot_id && result.boot_id && job.boot_id !== result.boot_id) return { status: 200, body: job };
    if (state === 'live') return active;
    if (action === 'delete' && state !== 'gone') return unknown;
  } else if (job.run_id && job.launch_phase !== 'exited') {
    let supervisor = job.supervisor_identity;
    if (job.run_dir && readExitReceipt(runFiles(job.run_dir, job.run_id).exit, job.run_id, supervisor)) {
      return { status: 200, body: job };
    }
    // Once spawn was requested, close the race with a delayed launcher before releasing its reservation.
    if (job.launch_phase === 'spawn_requested') {
      if (!job.run_dir) return { status: 409, body: { error: 'The pending launch has no recorded output folder.' } };
      // A delayed relay may still start a supervisor. Deleting the folder would remove
      // its cancellation fence before the supervisor sees it.
      if (action === 'delete' && !job.relay_exited_at) {
        return { status: 409, body: { error: 'The launch is pending. Use Stop before deleting its output folder.' } };
      }
      const claim = await pc.fence(job.run_id, job.run_dir, action);
      if (claim.fenced || (claim.boundary?.run_id === job.run_id && claim.boundary.kind === 'fence')) {
        return { status: 200, body: job };
      }
      if (claim.boundary?.run_id !== job.run_id || !validIdentity(claim.boundary?.supervisor)) {
        return { status: 409, body: { error: 'Could not cancel the pending launch. Try Stop or retry when it finishes starting.' } };
      }
      supervisor = claim.boundary.supervisor;
    }
    // A claimed launch cannot spawn after the transaction invalidates its run ID.
    if (job.launch_phase !== 'claimed') {
      const result = await pc.probe({ runs: { [job.run_id]: { run_id: job.run_id, supervisor, members: knownMembers(job) } } });
      if (job.boot_id && result.boot_id && job.boot_id !== result.boot_id) return { status: 200, body: job };
      const state = result.runs[job.run_id];
      if (state?.supervisor === 'alive' || (state?.active ?? 0) > 0 || state?.live.length || Object.values(state?.members ?? {}).includes('alive')) return active;
      if (action === 'delete' && (!result.ok || state?.supervisor !== 'exited' || state.error ||
          Object.values(state.members).includes('unknown'))) return unknown;
    }
  }
  return { status: 200, body: job };
}
