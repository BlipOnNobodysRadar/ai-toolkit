// Worker reconciliation, one pass per tick before admission (docs/STATE_MACHINES.md M10, "Reconciler").
// Automatic releases after spawn_requested use process evidence. Each release re-checks,
// inside its transaction, the launch phase (and identity) the evidence was gathered for.
//   any phase        the run was recorded under another boot id than the current one (a restart ended it)
//   attached         the supervisor's exit receipt (its job object reported 0 active processes), or the job object
//                    still open with 0 active processes and its supervisor (the only possible spawner) proved gone
//   spawn_requested  a spawn receipt (attach) or failed receipt (release); without a receipt: a successful FENCE of
//                    the run's boundary claim (no supervisor claimed it, and none can any more), or the claiming
//                    supervisor proved gone with its job object still open and empty
//   claimed          the admitting worker is gone, or is this worker with no launch registered (nothing spawned)
//   legacy (no run)  the previous trainer is gone, its PID has been reused, or its boot ended
// A job object that can no longer be opened is NOT evidence: its name disappears when the last handle closes, before
// its members have finished terminating (measured, docs Q4). Supervisor death without an exit receipt keeps
// the claims until the user confirms the job ended or the worker verifies exit. "unknown" (helper failure, wait failure, unsupported
// platform) always keeps the claims and says why. A scan of the old launch command is used only for pre-update jobs without a PID.
import fs from 'fs';
import type { Job } from '@prisma/client';
import { knownMembers, readExitReceipt, readReceipt, runFiles, savedMembers, validIdentity } from './launch';
import { legacyProbe, legacyProcess } from './legacy';
import { ProbeRequest, ProbeResult, ProcCtl, pidExists } from './procctl';
import { Db, ReleaseOutcome, attachRun, clearLegacy, isLegacyOccupant, keepLegacyTime, noteOwnership, releaseRun, updateMembers, withRetry } from './store';

export type WorkerSelf = { pid: number; identity: string; boot_id: string | null };

export type ReconcileContext = {
  db: Db;
  procctl: ProcCtl;
  self: WorkerSelf;
  inFlight: Set<string>;
  trainingRoot: string;
  tick: number;
  now?: () => number;
  graceMs?: number; // a stop request this old gets a force stop through the run boundary (the former Windows backstop)
  identityEveryTicks?: number; // full check cadence for runs that the cheap pid check says are alive
  backstopAttempts?: Map<string, number>;
  log?: (message: string) => void;
};

export type ReconcileReport = Record<string, string>;

const BACKSTOP_RETRY_MS = 30_000;

export const SUPERVISOR_LOST_NOTE =
  "The supervisor exited before reporting the final process state. If the job has finished, choose Mark as Stopped.";
const readJson = (file: string): any => {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (e: any) {
    return e?.code === 'ENOENT' ? undefined : null; // undefined: absent; null: present but unreadable
  }
};

const readLogTail = (logPath: string): string | null => {
  try {
    const size = fs.statSync(logPath).size;
    const length = Math.min(size, 4096);
    if (length === 0) return null;
    const buffer = Buffer.alloc(length);
    const fd = fs.openSync(logPath, 'r');
    try {
      fs.readSync(fd, buffer, 0, length, size - length);
    } finally {
      fs.closeSync(fd);
    }
    const lines = buffer.toString('utf8').split(/\r?\n/).filter(l => l.trim() !== '');
    const last = lines[lines.length - 1];
    return last ? last.slice(-300) : null;
  } catch {
    return null;
  }
};

type Plan = { job: Job; kind: string };

export const reconcile = async (ctx: ReconcileContext): Promise<ReconcileReport> => {
  const { db, procctl } = ctx;
  const now = ctx.now ?? Date.now;
  const report: ReconcileReport = {};
  const log = ctx.log ?? (() => {});
  const bootChanged = (recorded: string | null) => !!recorded && !!ctx.self.boot_id && recorded !== ctx.self.boot_id;

  const claims = await db.runClaim.findMany({ where: { kind: 'run' } });
  const runIds = [...new Set(claims.map(c => c.run_id))];
  const runJobs = runIds.length ? await db.job.findMany({ where: { run_id: { in: runIds } } }) : [];
  const byRun = new Map(runJobs.map(j => [j.run_id!, j]));
  for (const id of runIds) if (!byRun.has(id)) report[id] = 'orphan claims kept (no job row carries this run id)';

  const legacy = (
    await db.job.findMany({ where: { run_id: null, OR: [{ status: { in: ['starting', 'running', 'stopping'] } }, { pid: { not: null } }] } })
  ).filter(isLegacyOccupant);

  const req: ProbeRequest = { ...legacyProbe(legacy.filter(j => !bootChanged(j.boot_id))), runs: {} };
  const plan: Plan[] = [];
  for (const job of byRun.values()) {
    const runId = job.run_id!;
    if (ctx.inFlight.has(runId)) {
      report[runId] = 'launch in flight';
      continue;
    }
    if (bootChanged(job.boot_id) && job.launch_phase && ['claimed', 'spawn_requested', 'attached'].includes(job.launch_phase)) {
      plan.push({ job, kind: 'rebooted' });
      continue;
    }
    const files = job.run_dir ? runFiles(job.run_dir, runId) : null;
    if (job.launch_phase === 'attached' && job.pid_identity) {
      if (files && readExitReceipt(files.exit, runId, job.supervisor_identity)) {
        plan.push({ job, kind: 'exit-receipt' });
        continue;
      }
      const cheap =
        ['starting', 'running'].includes(job.status) && !job.stop && ctx.tick % (ctx.identityEveryTicks ?? 15) !== 0 && job.pid != null && pidExists(job.pid);
      if (cheap) {
        report[runId] = 'alive (pid check)';
        continue;
      }
      req.runs![runId] = { run_id: runId, supervisor: job.supervisor_identity, members: knownMembers(job) };
      plan.push({ job, kind: 'attached' });
    } else if (job.launch_phase === 'claimed') {
      if (job.launcher_identity === ctx.self.identity || !job.launcher_identity || job.launcher_pid == null) {
        plan.push({ job, kind: 'claimed-self' });
      } else {
        req.describe!.push(job.launcher_pid);
        plan.push({ job, kind: 'claimed-other' });
      }
    } else if (job.launch_phase === 'spawn_requested') {
      const receipt = files ? readReceipt(files.receipt, runId) : null;
      if (receipt) {
        plan.push({ job, kind: receipt.state === 'spawned' ? 'receipt-spawned' : 'receipt-failed' });
        continue;
      }
      const claim = files ? readJson(files.boundary) : null;
      if (claim === undefined) {
        // nobody has claimed the run's boundary yet: fence it once the launching side is proved unable to continue
        if (job.launcher_pid != null) req.describe!.push(job.launcher_pid);
        plan.push({ job, kind: 'unclaimed' });
      } else if (claim?.run_id === runId && claim.kind === 'fence') {
        plan.push({ job, kind: 'fenced' });
      } else if (claim?.run_id === runId && claim.kind === 'supervisor' && validIdentity(claim.supervisor)) {
        req.runs![runId] = { run_id: runId, supervisor: claim.supervisor, members: [] };
        plan.push({ job, kind: 'claimed-boundary' });
      } else {
        report[runId] = 'kept: boundary claim unreadable';
      }
    } else {
      report[runId] = `claims kept: unexpected launch phase ${job.launch_phase}`;
    }
  }
  for (const job of legacy) plan.push({ job, kind: 'legacy' });

  const helperKinds = ['attached', 'claimed-other', 'unclaimed', 'claimed-boundary'];
  const needsHelper = !!req.describe?.length || !!Object.keys(req.scans ?? {}).length || plan.some(p => helperKinds.includes(p.kind));
  const res: ProbeResult = needsHelper ? await procctl.probe(req) : { ok: true, describe: {}, runs: {}, scans: {} };
  const helperDown = needsHelper && !res.ok ? `Cannot check process state (${res.error}); ownership kept.` : null;

  const release = async (job: Job, phase: 'claimed' | 'spawn_requested' | 'attached', outcome: ReleaseOutcome, label: string) => {
    const r = await releaseRun(db, job.id, job.run_id!, phase === 'attached' ? { phase, identity: job.pid_identity } : { phase }, outcome);
    if (r.released && r.unexpected) appendUnexpected(job);
    report[job.run_id!] = r.released ? `released: ${label}${r.status ? ` -> ${r.status}` : ''}` : `kept: ${r.reason}`;
  };

  for (const { job, kind } of plan) {
    const runId = job.run_id ?? `legacy:${job.id}`;
    try {
      if (kind === 'rebooted') {
        const phase = job.launch_phase as 'claimed' | 'spawn_requested' | 'attached';
        const outcome: ReleaseOutcome =
          phase === 'attached'
            ? { kind: 'exited', evidence: 'the computer restarted after the run was launched', unexpectedInfo: unexpected(job) }
            : { kind: 'not_spawned', info: 'The computer restarted before the job process was confirmed' };
        await release(job, phase, outcome, 'boot changed');
        continue;
      }
      if (kind === 'claimed-self') {
        // this worker admitted it and has no launch registered for it: the phase never advanced, nothing spawned
        await release(job, 'claimed', { kind: 'not_spawned', info: 'Launch abandoned before its process was started' }, 'never spawned');
        continue;
      }
      if (kind === 'exit-receipt') {
        const files = runFiles(job.run_dir!, job.run_id!);
        const evidence = readExitReceipt(files.exit, job.run_id!, job.supervisor_identity)!.evidence ?? 'supervisor exit receipt';
        await release(job, 'attached', { kind: 'exited', evidence, unexpectedInfo: unexpected(job) }, 'exit receipt');
        continue;
      }
      if (kind === 'receipt-spawned') {
        const r = readReceipt(runFiles(job.run_dir!, job.run_id!).receipt, job.run_id!)!;
        await attachRun(db, job.id, job.run_id!, { identity: r.identity!, supervisor: r.supervisor, boundary: r.boundary, boot_id: r.boot_id });
        report[runId] = `attached from receipt ${r.identity}`;
        continue;
      }
      if (kind === 'receipt-failed') {
        const r = readReceipt(runFiles(job.run_dir!, job.run_id!).receipt, job.run_id!)!;
        await release(job, 'spawn_requested', { kind: 'not_spawned', info: `Error launching job: ${r.error ?? 'not started'}` }, 'receipt says not spawned');
        continue;
      }
      if (kind === 'fenced') {
        await release(job, 'spawn_requested', { kind: 'not_spawned', info: 'The job process never started (launch abandoned and fenced).' }, 'fenced');
        continue;
      }
      if (kind === 'legacy') {
        const state = legacyProcess(job, res);
        if (bootChanged(job.boot_id) || state === 'gone') {
          const evidence = bootChanged(job.boot_id) ? 'the computer restarted' : 'the previous trainer is no longer running';
          await clearLegacy(db, job, evidence);
          report[runId] = 'legacy cleared';
        } else {
          const note = state === 'live' ? null : 'Cannot identify the old training process. If the job has finished, choose Mark as Stopped.';
          if (ctx.self.boot_id && !job.boot_id) {
            await withRetry(() => db.job.updateMany({
              where: { id: job.id, run_id: null, boot_id: null, status: job.status, pid: job.pid, updated_at: job.updated_at },
              data: { boot_id: ctx.self.boot_id, ownership_note: note, ...keepLegacyTime(job) },
            }));
          } else {
            await noteOwnership(db, job, note);
          }
          report[runId] = `legacy kept: ${state}`;
        }
        continue;
      }
      if (helperDown) {
        await noteOwnership(db, job, helperDown);
        report[runId] = 'kept: helper unavailable';
        continue;
      }
      if (kind === 'claimed-other') {
        const d = res.describe[String(job.launcher_pid)];
        const gone = d?.state === 'gone' || (d?.state === 'alive' && d.identity !== job.launcher_identity);
        if (gone) {
          await release(job, 'claimed', { kind: 'not_spawned', info: 'The worker that admitted this run exited before starting it' }, 'admitting worker gone before spawn');
        } else {
          await noteOwnership(db, job, d?.state === 'alive' ? null : `Starting: cannot verify the admitting worker (pid ${job.launcher_pid}).`);
          report[runId] = `kept: admitting worker ${d?.state ?? 'unknown'}`;
        }
        continue;
      }
      if (kind === 'unclaimed') {
        const d = job.launcher_pid != null ? res.describe[String(job.launcher_pid)] : undefined;
        const launcherGone =
          job.relay_exited_at != null ||
          job.launcher_identity === ctx.self.identity || // this worker, nothing registered in flight: its relay has exited
          d?.state === 'gone' ||
          (d?.state === 'alive' && d.identity !== job.launcher_identity);
        if (!launcherGone) {
          await noteOwnership(db, job, `Starting: waiting for the launching worker (pid ${job.launcher_pid}).`);
          report[runId] = 'kept: launcher may still continue';
          continue;
        }
        // A supervisor still on its way (a slow relay) loses this race and starts nothing; if it won, it owns the run.
        const f = await procctl.fence(job.run_id!, job.run_dir!, ctx.self.identity);
        if (f.fenced) {
          await release(job, 'spawn_requested', { kind: 'not_spawned', info: 'The job process never started (its launcher stopped before starting it).' }, 'fenced');
        } else {
          await noteOwnership(db, job, f.error ? `Starting: cannot fence the abandoned launch (${f.error}).` : 'Starting: the run supervisor has just claimed the run.');
          report[runId] = f.error ? `kept: fence failed (${f.error})` : 'kept: supervisor claimed first';
        }
        continue;
      }
      if (kind === 'claimed-boundary') {
        const st = res.runs[job.run_id!];
        if (st?.supervisor === 'exited' && st.boundary === 'open' && st.active === 0) {
          await release(job, 'spawn_requested', { kind: 'not_spawned', info: 'The launch failed after its supervisor started; its job object is empty.' }, 'supervisor gone, job object empty');
        } else if (st?.supervisor === 'alive') {
          await noteOwnership(db, job, 'Starting: the run supervisor is starting the job.');
          report[runId] = 'kept: supervisor alive';
        } else {
          await noteOwnership(db, job, st?.supervisor === 'exited' ? SUPERVISOR_LOST_NOTE : `Starting: cannot verify the run supervisor (${st?.error ?? 'state unknown'}).`);
          report[runId] = `kept: supervisor ${st?.supervisor ?? 'unknown'}, boundary ${st?.boundary ?? 'unknown'}`;
        }
        continue;
      }
      if (kind === 'attached') {
        const st = res.runs[job.run_id!];
        const recorded = savedMembers(job);
        const live = st?.live ?? [];
        if (live.some(m => !recorded.includes(m))) await updateMembers(db, job.id, job.run_id!, [...new Set([...recorded, ...live])]);
        if (st?.supervisor === 'exited' && st.boundary === 'open' && st.active === 0) {
          await release(job, 'attached', { kind: 'exited', evidence: 'job object open with no active process, supervisor gone', unexpectedInfo: unexpected(job) }, 'empty job object');
          continue;
        }
        let note: string | null;
        if (st?.supervisor === 'exited') note = SUPERVISOR_LOST_NOTE;
        else if (!st || st.boundary !== 'open' || st.supervisor !== 'alive') note = `Cannot verify the run's processes (${st?.error ?? `supervisor ${st?.supervisor ?? 'unknown'}, boundary ${st?.boundary ?? 'unknown'}`}); ownership kept.`;
        else note = st.members[job.pid_identity!] === 'alive' ? null : `Waiting for ${st.active} remaining process(es) of this run to exit.`;
        // the former Windows stop backstop: a stop request this old gets a force stop through the run boundary
        const requested = job.stop_requested_at?.getTime();
        if (requested != null && now() - requested > (ctx.graceMs ?? 60_000) && st?.boundary === 'open' && (st.active ?? 0) > 0) {
          const attempts = ctx.backstopAttempts ?? new Map<string, number>();
          if (now() - (attempts.get(job.run_id!) ?? 0) > BACKSTOP_RETRY_MS) {
            attempts.set(job.run_id!, now());
            log(`Job ${job.id} still running ${(ctx.graceMs ?? 60_000) / 1000}s after the stop request; force stopping it`);
            const t = await procctl.terminate(job.run_id!, job.supervisor_identity);
            note = t.supported
              ? t.verified
                ? 'Force stopped after the stop request timed out; confirming exit.'
                : `Force stop after the stop request timed out did not finish (${t.error ?? `${t.remaining} remaining`}).`
              : `Stop pending: ${t.error}`;
          }
        }
        await noteOwnership(db, job, note);
        report[runId] = `kept: ${st?.boundary ?? 'unknown'} boundary, ${st?.active ?? '?'} active, supervisor ${st?.supervisor ?? 'unknown'}`;
      }
    } catch (e: any) {
      report[runId] = `error (kept): ${e?.message ?? e}`;
      console.error(`Reconcile ${runId}:`, e);
    }
  }
  return report;

  function unexpected(job: Job) {
    const tail = job.run_dir ? readLogTail(runFiles(job.run_dir, job.run_id!).log) : null;
    return tail ? `Job process exited unexpectedly. Last log line: ${tail}` : 'Job process exited unexpectedly.';
  }
  function appendUnexpected(job: Job) {
    try {
      if (job.run_dir) fs.appendFileSync(runFiles(job.run_dir, job.run_id!).log, `\n${unexpected(job)}\n`);
    } catch {
      // the log is a convenience
    }
  }
};
