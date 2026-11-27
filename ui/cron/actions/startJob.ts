import prisma from '../prisma';
import type { AdmittedJob } from '../ownership/store';
import path from 'path';
import { TOOLKIT_ROOT, getHFToken, getModelsPath } from '../paths';
import { resolveDetachedPythonPath, resolvePythonPath } from '../pythonPath';
import { LaunchResult, launchRun } from '../ownership/launch';

// Runs this worker is launching right now. The reconciler leaves them alone; once launchRun returns, the
// spawn_requested phase is resolved from the receipt or by the reconciler (docs/STATE_MACHINES.md M10).
export const launchesInFlight = new Set<string>();

// Launches an ADMITTED run (status starting, launch_phase claimed, claims held). The relay is
// toolkit/ui_procctl.py: it starts the run's supervisor detached (so the run is outside the worker's process tree
// and survives the UI being shut down or tree-killed); the supervisor owns the run's process boundary (Windows job
// object), starts the job through pythonw.exe so no console window appears, and records the root's pid@creation
// in an atomic per-run receipt. See ui/cron/ownership/launch.ts.
export default async function startJob(job: AdmittedJob): Promise<LaunchResult> {
  const env: Record<string, string> = {};
  try {
    const hfToken = await getHFToken();
    if (hfToken && hfToken.trim() !== '') {
      env.HF_TOKEN = hfToken;
    }
    // MODELS_PATH - one set in the env always takes precedence (it passes through via process.env); only fall back
    // to the setting if it is not set
    if (!process.env.MODELS_PATH || process.env.MODELS_PATH.trim() === '') {
      const modelsPath = await getModelsPath();
      if (modelsPath && modelsPath.trim() !== '') {
        env.MODELS_PATH = modelsPath;
      }
    }
  } catch (e) {
    // nothing was spawned: unregister so the reconciler releases the still-'claimed' run
    launchesInFlight.delete(job.run_id!);
    throw e;
  }
  const result = await launchRun(
    {
      db: prisma,
      toolkitRoot: TOOLKIT_ROOT,
      helperPath: path.join(TOOLKIT_ROOT, 'toolkit', 'ui_procctl.py'),
      helperPython: resolvePythonPath(),
      jobPython: resolveDetachedPythonPath(),
      runPy: path.join(TOOLKIT_ROOT, 'run.py'),
      sqliteDbPath: path.join(TOOLKIT_ROOT, 'aitk_db.db'),
      env,
      inFlight: launchesInFlight,
    },
    job,
  );
  if (result.result !== 'attached') console.log(`Launch of job ${job.id} run ${job.run_id}: ${JSON.stringify(result)}`);
  return result;
}
