// Node side of toolkit/ui_procctl.py (identity checks, command-line scans, run-boundary state, termination).
// A helper failure or timeout is always reported as "unknown"; it is never evidence that a process exited.
import { execFile } from 'child_process';
import path from 'path';

export type ScanResult = {
  complete: boolean;
  matches: { identity: string; ppid: number; cmdline: string }[];
  unreadable?: number;
  denied?: number;
  error?: string;
};
export type Describe = { state: 'alive' | 'gone' | 'denied' | 'unknown'; identity?: string; started_ms?: number; cmdline?: string | null; why?: string };
// The run's ownership container (Windows job object) as seen now. 'absent' = the object no longer exists.
export type RunState = {
  supervisor: 'alive' | 'exited' | 'unknown';
  boundary: 'open' | 'absent' | 'unknown';
  active?: number;
  live: string[];
  members: Record<string, 'alive' | 'exited' | 'unknown'>;
  error?: string;
};
export type ProbeRequest = {
  describe?: number[];
  runs?: Record<string, { run_id: string; supervisor: string | null; members?: string[] }>;
  scans?: Record<string, { needle: string; since_ms: number; legacy_launchers_only?: boolean }>;
};
export type ProbeResult = {
  ok: boolean;
  error?: string;
  boot_id?: string | null;
  describe: Record<string, Describe>;
  runs: Record<string, RunState>;
  scans: Record<string, ScanResult>;
};
export type TerminateResult = { supported: boolean; verified: boolean; boundary?: string; remaining?: number; requested?: boolean; error?: string };

export type FenceResult = { fenced: boolean; boundary?: any; error?: string };
export type ProcCtl = {
  probe(req: ProbeRequest): Promise<ProbeResult>;
  terminate(runId: string, supervisor: string | null, runDir?: string | null): Promise<TerminateResult>;
  terminateLegacy(jobId: string, jobName: string, identity: string): Promise<TerminateResult>;
  // claim an unattached run's boundary for good (see toolkit/ui_procctl.py "Boundary claim")
  fence(runId: string, runDir: string, by: string): Promise<FenceResult>;
  self(pid: number): Promise<{ identity: string; boot_id: string | null } | null>;
  helperPath: string;
  python: string;
};

const HELPER_TIMEOUT_MS = 60_000;

const call = (python: string, helper: string, cmd: string, payload: any): Promise<{ ok: true; value: any } | { ok: false; error: string }> =>
  new Promise(resolve => {
    const child = execFile(
      python,
      cmd === 'terminate-legacy' ? ['-I', helper, cmd] : ['-I', '-S', helper, cmd],
      { windowsHide: true, timeout: HELPER_TIMEOUT_MS, maxBuffer: 64 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          resolve({ ok: false, error: `${error.message}${stderr ? `: ${String(stderr).slice(-500)}` : ''}` });
          return;
        }
        try {
          resolve({ ok: true, value: JSON.parse(String(stdout)) });
        } catch (e: any) {
          resolve({ ok: false, error: `unreadable helper output: ${e?.message}` });
        }
      },
    );
    child.stdin?.end(JSON.stringify(payload));
  });

export const makeProcCtl = (python: string, toolkitRoot: string): ProcCtl => {
  const helperPath = path.join(toolkitRoot, 'toolkit', 'ui_procctl.py');
  return {
    helperPath,
    python,
    async probe(req) {
      const r = await call(python, helperPath, 'probe', req);
      if (r.ok === false) return { ok: false, error: r.error, describe: {}, runs: {}, scans: {} };
      return { ok: true, boot_id: r.value.boot_id ?? null, describe: r.value.describe ?? {}, runs: r.value.runs ?? {}, scans: r.value.scans ?? {} };
    },
    async terminate(runId, supervisor, runDir) {
      const r = await call(python, helperPath, 'terminate', { run_id: runId, supervisor, run_dir: runDir });
      if (r.ok === false) return { supported: true, verified: false, error: r.error };
      return r.value;
    },
    async terminateLegacy(jobId, jobName, identity) {
      const r = await call(python, helperPath, 'terminate-legacy', { job_id: jobId, job_name: jobName, identity });
      return r.ok === false ? { supported: true, verified: false, error: r.error } : r.value;
    },
    async fence(runId, runDir, by) {
      const r = await call(python, helperPath, 'fence', { run_id: runId, run_dir: runDir, by });
      if (r.ok === false) return { fenced: false, error: r.error };
      return r.value;
    },
    async self(pid) {
      const r = await call(python, helperPath, 'self', { pid });
      if (r.ok === false || r.value?.identity?.state !== 'alive') return null;
      return { identity: r.value.identity.identity as string, boot_id: r.value.boot_id ?? null };
    },
  };
};

// Cheap liveness for a pid (no identity): ESRCH proves no process has that pid. "alive" may be a reused pid, so it
// only ever keeps ownership, never grants anything.
export const pidExists = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e: any) {
    return e?.code === 'EPERM';
  }
};
