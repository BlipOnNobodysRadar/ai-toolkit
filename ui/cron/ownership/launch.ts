// Launching an admitted run (docs/STATE_MACHINES.md M10, "Launch phases").
//
//   claimed          claims held, nothing spawned. Any failure here is a KNOWN pre-spawn failure: release.
//   spawn_requested  committed BEFORE the relay is spawned. From here a child may exist; claims are released only
//                    on a receipt that says the child was never created (or was terminated and verified gone), or
//                    by the reconciler once the launcher is proved gone and a complete scan finds no process.
//   attached         the spawn receipt was read: root identity, the run's supervisor and its process boundary.
//
// Process chain: worker -> relay (`ui_procctl.py launch`, exits) -> supervisor (`ui_procctl.py supervise`, detached,
// lives as long as the run) -> root (the job) and its descendants, all inside the run's boundary (Windows job
// object). The run id is in every one of those command lines (it is part of the per-run config path), so a restarted
// worker can find any of them even if this worker died before recording anything. Time never releases a run: this
// code waits for the relay's real exit, however long it takes.
import { ChildProcess, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import type { Job } from '@prisma/client';
import { Db, attachRun, markRelayExited, markSpawnRequested, noteOwnership, recordRelay, releaseRun } from './store';
import { bindLaunchConfig } from './launchConfig';
import { withOpLock } from './opLock';
import { writerLeasePath } from './writerLease';

export const RELAY_EXIT = { SPAWNED: 0, NOT_SPAWNED: 2, RECEIPT_FAILED: 3, UNCERTAIN: 4 };

export type LaunchContext = {
  db: Db;
  toolkitRoot: string;
  helperPath: string;
  helperPython: string; // console interpreter for the short-lived relay (hidden window)
  jobPython: string; // interpreter for the job (pythonw on Windows)
  runPy: string;
  sqliteDbPath: string;
  env: Record<string, string>;
  inFlight: Set<string>;
  platform?: string; // defaults to process.platform (tests pass another value to exercise the refusal)
  hooks?: {
    beforeSpawn?: () => Promise<void>;
    afterRelaySpawn?: (relay: ChildProcess) => Promise<void>;
    relayEnv?: Record<string, string>;
  };
};

export type LaunchResult =
  | { result: 'attached'; identity: string }
  | { result: 'not_spawned'; info: string }
  | { result: 'cancelled' }
  | { result: 'uncertain'; info: string };

// Per-run files (names must match toolkit/ui_procctl.py run_files).
export const runFiles = (runDir: string, runId: string) => ({
  config: path.join(runDir, `.job_config.${runId}.json`),
  receipt: path.join(runDir, `.run.${runId}.receipt.json`),
  members: path.join(runDir, `.run.${runId}.members.json`),
  exit: path.join(runDir, `.run.${runId}.exit.json`),
  // exclusive claim on the run's boundary: the supervisor's, or the reconciler's fence (hard-linked, never replaced)
  boundary: path.join(runDir, `.run.${runId}.boundary.json`),
  log: path.join(runDir, 'log.txt'),
});

export type Receipt = {
  run_id: string;
  state: 'spawned' | 'failed' | 'exited';
  pid?: number;
  identity?: string;
  supervisor?: string;
  boundary?: string;
  members?: string[];
  evidence?: string;
  exit_code?: number;
  boot_id?: string | null;
  error?: string;
};

const readJson = (file: string): any => {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
};

export const validIdentity = (x: any): x is string => typeof x === 'string' && /^\d+@[^@\s]+$/.test(x) && !x.endsWith('@None');

// A spawn receipt counts only if it is complete JSON for this very run (written with temp+fsync+replace).
export const readReceipt = (receiptPath: string, runId: string): Receipt | null => {
  const r = readJson(receiptPath);
  if (r?.run_id !== runId) return null;
  if (r.state === 'spawned' && validIdentity(r.identity) && validIdentity(r.supervisor)) return r;
  if (r.state === 'failed') return r;
  return null;
};

// The supervisor's exit receipt: proof that the run's whole boundary is empty. Bound to run id AND supervisor.
export const readExitReceipt = (exitPath: string, runId: string, supervisor: string | null): Receipt | null => {
  const r = readJson(exitPath);
  if (r?.run_id !== runId || r.state !== 'exited' || !supervisor || r.supervisor !== supervisor) return null;
  return { ...r, exit_code: Number.isSafeInteger(r.exit_code) ? r.exit_code : undefined };
};

export const readMembers = (membersPath: string, runId: string): string[] => {
  const r = readJson(membersPath);
  return r?.run_id === runId && Array.isArray(r.members) ? r.members.filter(validIdentity) : [];
};

export const savedMembers = (job: Pick<Job, 'run_members'>): string[] => {
  try {
    const m: unknown = JSON.parse(job.run_members || '[]');
    return Array.isArray(m) ? m.filter(validIdentity) : [];
  } catch {
    return [];
  }
};

export const knownMembers = (job: Job): string[] => {
  const recorded = savedMembers(job);
  if (job.pid_identity) recorded.push(job.pid_identity);
  if (job.run_id && job.run_dir) recorded.push(...readMembers(runFiles(job.run_dir, job.run_id).members, job.run_id));
  return [...new Set(recorded)];
};

const writeAtomic = (file: string, text: string) => {
  const tmp = `${file}.${process.pid}.tmp`;
  const fd = fs.openSync(tmp, 'w');
  try {
    fs.writeSync(fd, text);
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  fs.renameSync(tmp, file);
};

const appendLog = (logPath: string, message: string) => {
  try {
    fs.appendFileSync(logPath, message);
  } catch (e) {
    console.error('Error writing to job log:', e);
  }
};

const rotateLog = (folder: string, logPath: string) => {
  try {
    if (fs.existsSync(logPath)) {
      const logsFolder = path.join(folder, 'logs');
      fs.mkdirSync(logsFolder, { recursive: true });
      let num = 0;
      while (fs.existsSync(path.join(logsFolder, `${num}_log.txt`))) num++;
      fs.renameSync(logPath, path.join(logsFolder, `${num}_log.txt`));
    }
  } catch (e) {
    console.error('Error moving log file:', e);
  }
};

class PreSpawnError extends Error {}

// The helper implements a Windows job object, Linux subreaper, and macOS process group.
export const SUPPORTED_PLATFORMS = ['win32', 'linux', 'darwin'];

export const launchRun = async (ctx: LaunchContext, job: Job): Promise<LaunchResult> => {
  const runId = job.run_id!;
  const folder = job.run_dir!;
  const files = runFiles(folder, runId);
  const { db } = ctx;
  ctx.inFlight.add(runId); // normally already registered by the caller, synchronously at admission
  let spawnRequested = false;
  let logFd: number | null = null;
  try {
    // ---- claimed: nothing exists yet; every failure is a known pre-spawn failure
    // Bind the saved config to the claimed folder BEFORE any run-folder/config write. In particular this rebinds a
    // stale Training Folder setting and process/config names, and refuses ambiguous multi-process layouts.
    const jobConfig = bindLaunchConfig(job.job_config, job.name, folder);
    for (const proc of jobConfig.config.process) proc.sqlite_db_path = ctx.sqliteDbPath;
    // CLI trainers have no UI claim. Respect their lease during launch preparation too, before rotating their log
    // or replacing a config. Release before spawning: the Python constructor acquires its lifetime lease itself.
    // If another trainer wins that handoff, Python refuses before training/config writes.
    const prepared = await withOpLock(writerLeasePath(folder), async () => {
      // A delayed launch may acquire this lock after Delete or a later run. Check before touching either one's files.
      if (!(await db.job.findFirst({ where: { id: job.id, run_id: runId, launch_phase: 'claimed' }, select: { id: true } }))) return false;
      fs.mkdirSync(folder, { recursive: true });
      rotateLog(folder, files.log);
      writeAtomic(files.config, JSON.stringify(jobConfig, null, 2));
      logFd = fs.openSync(files.log, 'a');
      return true;
    });
    if (!prepared) return { result: 'cancelled' };
    if (!fs.existsSync(ctx.runPy)) throw new PreSpawnError('run.py not found');
    const platform = ctx.platform ?? process.platform;
    if (!SUPPORTED_PLATFORMS.includes(platform)) {
      // no process boundary is implemented there, so a launched run could never be proven finished
      throw new PreSpawnError(`launching jobs is not supported on ${platform} in this version: no process ownership boundary is implemented for it`);
    }

    if (!(await markSpawnRequested(db, job.id, runId))) {
      // a stop arrived before anything was spawned, or the run is no longer this one: nothing was started
      const now = await db.job.findUnique({ where: { id: job.id } });
      if (now?.run_id === runId && now.launch_phase === 'claimed') {
        const info = now.stop ? 'Job stopped before its process started' : 'Launch cancelled';
        await releaseRun(db, job.id, runId, { phase: 'claimed' }, { kind: 'not_spawned', info });
        return { result: 'not_spawned', info };
      }
      return { result: 'cancelled' };
    }
    spawnRequested = true;
    if (ctx.hooks?.beforeSpawn) await ctx.hooks.beforeSpawn();

    // ---- spawn_requested: from here on a child may exist
    const env = {
      ...process.env,
      ...ctx.env,
      ...(ctx.hooks?.relayEnv ?? {}),
      AITK_JOB_ID: job.id,
      AITK_RUN_ID: runId,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      IS_AI_TOOLKIT_UI: '1',
      PYTHONUNBUFFERED: '1', // write Python output immediately so it is not lost on a crash
    };
    const args = ['-I', '-S', ctx.helperPath, 'launch', '--run-id', runId, '--run-dir', folder, '--cwd', ctx.toolkitRoot, '--', ctx.jobPython, ctx.runPy, files.config];
    const relay = spawn(ctx.helperPython, args, { env, cwd: ctx.toolkitRoot, windowsHide: true, stdio: ['ignore', logFd, logFd] });
    const ended = new Promise<{ code: number | null; spawnError?: Error }>(resolve => {
      relay.once('error', error => {
        if (relay.pid === undefined) resolve({ code: null, spawnError: error });
      });
      relay.once('exit', code => resolve({ code }));
    });
    if (relay.pid !== undefined) {
      await recordRelay(db, job.id, runId, relay.pid).catch(e => console.error('Could not record relay pid:', e));
      if (ctx.hooks?.afterRelaySpawn) await ctx.hooks.afterRelaySpawn(relay);
    }
    const { code, spawnError } = await ended;
    if (spawnError) {
      // the relay process itself was never created, so it cannot have started anything
      const info = `Error launching job: ${spawnError.message}`;
      if (logFd !== null) appendLog(files.log, `${info}\n`);
      await releaseRun(db, job.id, runId, { phase: 'spawn_requested' }, { kind: 'not_spawned', info });
      return { result: 'not_spawned', info };
    }
    await markRelayExited(db, job.id, runId).catch(e => console.error('Could not record relay exit:', e));

    const receipt = readReceipt(files.receipt, runId);
    // a 'spawned' receipt is authoritative whatever the exit code: the child exists (or existed) with that identity
    if (receipt?.state === 'spawned') {
      await attachRun(db, job.id, runId, { identity: receipt.identity!, supervisor: receipt.supervisor, boundary: receipt.boundary, boot_id: receipt.boot_id });
      try {
        fs.writeFileSync(path.join(folder, 'pid.txt'), String(receipt.pid ?? ''), { flag: 'w' });
      } catch (e) {
        console.error('Error writing pid file:', e);
      }
      return { result: 'attached', identity: receipt.identity! };
    }
    if (code === RELAY_EXIT.NOT_SPAWNED || receipt?.state === 'failed') {
      const info = `Error launching job: ${receipt?.error ?? 'the job process could not be started'}`;
      await releaseRun(db, job.id, runId, { phase: 'spawn_requested' }, { kind: 'not_spawned', info });
      return { result: 'not_spawned', info };
    }
    if (code === RELAY_EXIT.RECEIPT_FAILED) {
      // the supervisor terminated the run's whole job object and saw the kernel's member count reach 0
      const info = 'Error launching job: its process identity could not be recorded, so it was stopped';
      await releaseRun(db, job.id, runId, { phase: 'spawn_requested' }, { kind: 'not_spawned', info });
      return { result: 'not_spawned', info };
    }
    const info = `Launch outcome unknown (launcher exit code ${code}, no usable receipt); the GPUs stay reserved until the process is found or ruled out.`;
    const now = await db.job.findUnique({ where: { id: job.id } });
    if (now) await noteOwnership(db, now, info).catch(() => {});
    return { result: 'uncertain', info };
  } catch (e: any) {
    if (!spawnRequested) {
      const info = `Error launching job: ${e?.message || 'Unknown error'}`;
      console.error('Error launching process:', e);
      if (logFd !== null) appendLog(files.log, `${info}\n`);
      // if this release fails the claims stay in 'claimed'; the reconciler releases them (no spawn ever happened)
      await releaseRun(db, job.id, runId, { phase: 'claimed' }, { kind: 'not_spawned', info }).catch(err => console.error('Release failed:', err));
      return { result: 'not_spawned', info };
    }
    const info = `Launch outcome unknown (${e?.message || e}); the GPUs stay reserved until the process is found or ruled out.`;
    console.error('Error after spawn request:', e);
    return { result: 'uncertain', info };
  } finally {
    ctx.inFlight.delete(runId);
    if (logFd !== null) {
      try {
        fs.closeSync(logFd);
      } catch {
        // nothing useful to do if the log handle will not close
      }
    }
  }
};
