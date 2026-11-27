import { JobConfig } from '@/types';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

export const startJob = async (jobID: string): Promise<void> => {
  await apiClient.get(`/api/jobs/${jobID}/start`);
};

export const stopJob = async (jobID: string): Promise<void> => {
  await apiClient.get(`/api/jobs/${jobID}/stop`);
};

const snapshot = ({ status, pid, run_id, launch_phase, name, run_dir }: Job) => ({ status, pid, run_id, launch_phase, name, run_dir });

export const deleteJob = async (jobID: string, job?: Job): Promise<{ keptOutput?: string }> => {
  const url = `/api/jobs/${jobID}/delete`;
  const response = job ? await apiClient.post(url, snapshot(job)) : await apiClient.get(url);
  return response.data;
};

export const saveJobNow = async (jobID: string): Promise<void> => {
  await apiClient.get(`/api/jobs/${jobID}/save_now`);
};

export const sampleJobNow = async (jobID: string): Promise<void> => {
  await apiClient.get(`/api/jobs/${jobID}/sample_now`);
};

export const markJobAsStopped = async (jobID: string): Promise<void> => {
  await apiClient.get(`/api/jobs/${jobID}/mark_stopped`);
};

export const confirmJobStopped = (job: Job) =>
  ownershipOf(job).held ? apiClient.post(`/api/jobs/${job.id}/mark_stopped`, snapshot(job)) : markJobAsStopped(job.id);

export const getJobConfig = (job: Job): JobConfig | null => {
  try {
    return JSON.parse(job.job_config) as JobConfig | null;
  } catch {
    return null;
  }
};

// Run ownership as the row shows it (docs/STATE_MACHINES.md M10). The server enforces ownership from its claims;
// this is the display view: a run still holds its GPUs/folder from admission until its process exit is verified,
// even when Python already reported completed/error/stopped; a pre-update ("legacy") row holds them while it has a
// pid or an active status.
export const ownershipOf = (job: Job): { held: boolean; legacy: boolean } => {
  const legacy = job.run_id == null && (['starting', 'running', 'stopping'].includes(job.status) || job.pid != null);
  const run = job.run_id != null && job.launch_phase != null && job.launch_phase !== 'exited';
  return { held: run || legacy, legacy };
};

// Status label shown to the user: the reported status, plus "exiting" while a finished run's process is still owned.
export const displayStatus = (job: Job): string => {
  const { held } = ownershipOf(job);
  if (job.status === 'delete_failed') return 'delete failed';
  if (held && ['completed', 'error', 'stopped', 'queued'].includes(job.status)) return `${job.status} (exiting)`;
  return job.status;
};

export const getAvaliableJobActions = (job: Job) => {
  const jobConfig = getJobConfig(job);
  const { held, legacy } = ownershipOf(job);
  const isStopping = job.status === 'stopping' || (job.stop && ['starting', 'running'].includes(job.status));
  // 'deleting' with no run ownership is a deletion that did not finish (e.g. the server stopped mid-way): Delete
  // resumes it. A deletion still running elsewhere holds the server's deletion lock and the retry is refused (409),
  // so the browser does not need to guess whether the earlier deleter is alive.
  const canDelete = held || ['queued', 'completed', 'stopped', 'error', 'delete_failed', 'deleting'].includes(job.status);
  const isRetryDelete = canDelete && ['deleting', 'delete_failed'].includes(job.status);
  let canEdit = !held && ['queued', 'completed', 'stopped', 'error'].includes(job.status);
  const canRemoveFromQueue = job.status === 'queued';
  const canStop = ['starting', 'running'].includes(job.status) && !isStopping;
  let canStart = !held && ['stopped', 'error'].includes(job.status);
  // can resume if more steps were added
  const processes = jobConfig?.config?.process;
  if (!held && job.status === 'completed' && Array.isArray(processes) && processes.some(proc => (proc?.train?.steps || 0) > job.step)) {
    canStart = true;
  }
  const canForceStop = held && (legacy || job.launch_phase === 'attached');
  // Owners can recover a stale reservation even when the worker could not identify the process.
  const canMarkStopped = !['deleting', 'delete_failed'].includes(job.status) && (held || job.status !== 'stopped');
  return { canDelete, isRetryDelete, canEdit, canStop, canStart, canRemoveFromQueue, canForceStop, canMarkStopped, isStopping };
};

// The server's refusal text for an action (409 ownership conflicts etc.), for showing to the user.
export const actionError = (e: any): string => e?.response?.data?.error || e?.message || 'Request failed';

export const forceStopJob = (jobID: string) => apiClient.get(`/api/jobs/${jobID}/kill`).then(res => res.data);

export const getNumberOfSamples = (job: Job) => {
  const sample = getJobConfig(job)?.config?.process?.[0]?.sample;
  return Math.max(sample?.prompts?.length || 0, sample?.samples?.length || 0);
};

export const getTotalSteps = (job: Job) => {
  const processes = getJobConfig(job)?.config?.process;
  const configured = Number(processes?.[0]?.train?.steps);
  const firstTarget = Number.isSafeInteger(configured) && configured > 0 ? configured : 0;
  // Show edits to a single trainer's target immediately, even before the next launch.
  if (job.job_type === 'train' && processes?.length === 1 && firstTarget > 0) return firstTarget;
  return job.total_steps ?? firstTarget;
};
