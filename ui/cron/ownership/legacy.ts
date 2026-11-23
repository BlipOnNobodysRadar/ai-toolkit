// Earlier launchers stored the trainer PID but never cleared it on normal completion.
// A launch interrupted before the PID was saved can still be identified by its config.
import type { Job } from '@prisma/client';
import type { ProbeRequest, ProbeResult } from './procctl';

type LegacyJob = Pick<Job, 'name' | 'pid' | 'updated_at'>;
const configName = '.job_config.json';

export const legacyProbe = (jobs: LegacyJob[]): ProbeRequest => ({
  describe: [...new Set(jobs.flatMap(job => job.pid == null ? [] : [job.pid]))],
  scans: jobs.some(job => job.pid == null)
    ? { legacy: { needle: configName, since_ms: 0, legacy_launchers_only: true } }
    : {},
});

const normalize = (s: string) => process.platform === 'win32' ? s.replace(/\\/g, '/').toLowerCase() : s;

// Match the job folder even if the installation's configured training root has since moved.
const usesConfig = (cmd: string, name: string) => normalize(cmd).includes(normalize(`/${name}/${configName}`));

export const legacyProcess = (job: LegacyJob, probe: ProbeResult): 'gone' | 'live' | 'unknown' => {
  if (!probe.ok) return 'unknown';
  if (job.pid != null) {
    const p = probe.describe[String(job.pid)];
    if (p?.state === 'gone') return 'gone';
    if (p?.state !== 'alive') return 'unknown';
    if (p.cmdline && usesConfig(p.cmdline, job.name)) return 'live';
    // Prisma recorded this PID after spawn. A process born later than that write is a different process.
    // Leave room for the OS start-time clock's resolution.
    return p.started_ms != null && p.started_ms > job.updated_at.getTime() + 2000 ? 'gone' : 'unknown';
  }
  const scan = probe.scans.legacy;
  if (scan?.matches.some(p => usesConfig(p.cmdline, job.name))) return 'live';
  // An old Windows relay can take up to 30 seconds to publish the newly spawned PID.
  const launchSettled = Date.now() - job.updated_at.getTime() > 30_000;
  return scan?.complete && launchSettled ? 'gone' : 'unknown';
};
