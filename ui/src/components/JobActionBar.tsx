import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { Eye, Trash2, Pen, Play, Pause, Cog, X, Copy, Save, OctagonX, Image } from 'lucide-react';
import { LuLoader } from 'react-icons/lu';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { Job } from '@prisma/client';
import {
  startJob,
  stopJob,
  deleteJob,
  getAvaliableJobActions,
  markJobAsStopped,
  saveJobNow,
  sampleJobNow,
  forceStopJob,
  actionError,
} from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { redirect } from 'next/navigation';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';

interface JobActionBarProps {
  job: Job;
  onRefresh?: () => void;
  afterDelete?: () => void;
  hideView?: boolean;
  className?: string;
  autoStartQueue?: boolean;
  // Where the gear menu opens; defaults to below. Use 'top end' when the bar is
  // pinned to the bottom of the screen so the menu opens upward into view.
  menuAnchor?: 'bottom' | 'top end';
}

type PendingAction = 'start' | 'remove' | 'stop' | 'delete' | 'save' | 'sample' | 'markStopped' | 'forceStop';

// If the job never reports a state change (e.g. the request silently failed
// server-side), unlock the bar after this long so it can't stay stuck.
const PENDING_TIMEOUT_MS = 30_000;

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  autoStartQueue = false,
  menuAnchor = 'bottom',
}: JobActionBarProps) {
  const { canStart, canStop, canDelete, isRetryDelete, canEdit, canRemoveFromQueue, canForceStop, canMarkStopped } =
    getAvaliableJobActions(job);

  if (!afterDelete) afterDelete = onRefresh;

  // The action currently in flight. While set, every other action is disabled
  // and the pending one shows a spinner. It is cleared once the job comes back
  // in a different state (status / stop flag), on error, or after a timeout.
  const [pending, setPending] = useState<PendingAction | null>(null);
  // Job state key captured when the pending action was fired. null means the
  // action does not change job state and clears as soon as the request resolves.
  const pendingStateKeyRef = useRef<string | null>(null);
  const jobStateKey = `${job.status}|${job.stop ? 1 : 0}|${job.launch_phase ?? ''}`;

  const clearPending = () => {
    pendingStateKeyRef.current = null;
    setPending(null);
  };

  useEffect(() => {
    if (pending && pendingStateKeyRef.current !== null && pendingStateKeyRef.current !== jobStateKey) {
      clearPending();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobStateKey, pending]);

  useEffect(() => {
    if (!pending) return;
    const timer = setTimeout(clearPending, PENDING_TIMEOUT_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending]);

  const runAction = async (action: PendingAction, fn: () => Promise<void>, waitForJobUpdate = true) => {
    if (pending) return;
    pendingStateKeyRef.current = waitForJobUpdate ? jobStateKey : null;
    setPending(action);
    try {
      await fn();
      if (!waitForJobUpdate) clearPending();
    } catch (e) {
      console.error(`Error running job action "${action}":`, e);
      clearPending();
      // e.g. 409 "the previous run's process has not exited yet": the server's reason, not a silent no-op
      alert(actionError(e));
    }
  };

  const isBusy = pending !== null;
  const iconSizeClass = 'w-5 h-5 sm:w-6 sm:h-6';
  const actionButtonClass = 'ml-1 sm:ml-2 opacity-100 disabled:opacity-40 disabled:cursor-not-allowed';
  const menuItemClass = 'px-4 py-1 rounded flex items-center gap-2';
  const menuItemEnabledClass = `${menuItemClass} cursor-pointer hover:bg-gray-800`;
  const menuItemDisabledClass = `${menuItemClass} opacity-40 cursor-not-allowed`;
  const spinner = <LuLoader className={`${iconSizeClass} animate-spin`} />;
  const menuSpinner = <LuLoader className="w-4 h-4 animate-spin" />;
  const menuPending = pending === 'save' || pending === 'sample' || pending === 'markStopped' || pending === 'forceStop';

  return (
    <div className={`flex items-center flex-shrink-0 ${className ?? ''}`}>
      {canStart && (
        <Button
          disabled={isBusy}
          onClick={() => {
            if (!canStart) return;
            runAction('start', async () => {
              await startJob(job.id);
              // start the queue as well
              if (autoStartQueue) {
                await startQueue(job.gpu_ids);
              }
              if (onRefresh) onRefresh();
            });
          }}
          className={actionButtonClass}
        >
          {pending === 'start' ? spinner : <Play className={iconSizeClass} />}
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          disabled={isBusy}
          onClick={() => {
            if (!canRemoveFromQueue) return;
            runAction('remove', async () => {
              await markJobAsStopped(job.id);
              if (onRefresh) onRefresh();
            });
          }}
          className={actionButtonClass}
        >
          {pending === 'remove' ? spinner : <X className={iconSizeClass} />}
        </Button>
      )}
      {canStop && (
        <Button
          disabled={isBusy}
          onClick={() => {
            if (!canStop || isBusy) return;
            openConfirm({
              title: 'Stop Job',
              message: `Are you sure you want to stop the job "${job.name}"? You CAN resume later.`,
              type: 'info',
              confirmText: 'Stop',
              onConfirm: () =>
                runAction('stop', async () => {
                  await stopJob(job.id);
                  if (onRefresh) onRefresh();
                }),
            });
          }}
          className={actionButtonClass}
        >
          {pending === 'stop' ? spinner : <Pause className={iconSizeClass} />}
        </Button>
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className="ml-1 sm:ml-2 text-gray-200 hover:text-gray-100 inline-block">
          <Eye className={iconSizeClass} />
        </Link>
      )}
      {job.job_type === 'caption' && canEdit && (
        <div
          className="ml-1 sm:ml-2 hover:text-gray-100 inline-block cursor-pointer"
          onClick={() =>
            openCaptionDatasetModal(
              job.job_ref || '',
              () => {
                if (onRefresh) onRefresh();
              },
              { jobId: job.id },
            )
          }
        >
          <Pen className={iconSizeClass} />
        </div>
      )}
      {job.job_type === 'train' && canEdit && (
        <Link href={`/jobs/new?id=${job.id}`} className="ml-1 sm:ml-2 hover:text-gray-100 inline-block">
          <Pen className={iconSizeClass} />
        </Link>
      )}
      <Button
        disabled={isBusy || !canDelete}
        title={
          isRetryDelete
            ? 'Retry Delete: finish the deletion that did not complete'
            : canDelete
              ? 'Delete'
              : 'Available once the job has stopped and its process has exited'
        }
        aria-label={isRetryDelete ? 'Retry Delete' : 'Delete'}
        onClick={() => {
          if (isBusy || !canDelete) return;
          const message = isRetryDelete
            ? `The deletion of "${job.name}" did not finish. Retry it now? Its folder will be removed from your disk. If another deletion of this job is still running, the retry is refused.`
            : `Are you sure you want to delete the job "${job.name}"? This will also permanently remove it from your disk.`;
          openConfirm({
            title: isRetryDelete ? 'Retry Delete' : 'Delete Job',
            message: message,
            type: 'warning',
            confirmText: isRetryDelete ? 'Retry Delete' : 'Delete',
            onConfirm: () =>
              runAction('delete', async () => {
                await deleteJob(job.id);
                if (afterDelete) afterDelete();
              }),
          });
        }}
        className={actionButtonClass}
      >
        {pending === 'delete' ? spinner : <Trash2 className={iconSizeClass} />}
      </Button>
      <div className="border-r border-1 border-gray-700 ml-1 sm:ml-2 inline"></div>
      <Menu>
        <MenuButton className={'ml-1 sm:ml-2'}>{menuPending ? spinner : <Cog className={iconSizeClass} />}</MenuButton>
        <MenuItems
          anchor={{ to: menuAnchor, gap: 16 }}
          className="bg-gray-900 border border-gray-700 rounded shadow-lg w-60 px-2 py-2 z-50"
        >
          {job.job_type === 'train' && (
            <MenuItem>
              <Link
                href={`/jobs/new?cloneId=${job.id}`}
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
              >
                <Copy className="w-4 h-4" />
                Clone Job
              </Link>
            </MenuItem>
          )}
          {job.job_type === 'train' && canStop && (
            <MenuItem disabled={isBusy}>
              <div
                className={isBusy ? menuItemDisabledClass : menuItemEnabledClass}
                onClick={() => {
                  if (isBusy) return;
                  // does not change job state, so release as soon as the request resolves
                  runAction(
                    'save',
                    async () => {
                      await saveJobNow(job.id);
                      if (onRefresh) onRefresh();
                    },
                    false,
                  );
                }}
              >
                {pending === 'save' ? menuSpinner : <Save className="w-4 h-4" />}
                Save Next Step
              </div>
            </MenuItem>
          )}
          {job.job_type === 'train' && canStop && (
            <MenuItem disabled={isBusy}>
              <div
                className={isBusy ? menuItemDisabledClass : menuItemEnabledClass}
                onClick={() => {
                  if (isBusy) return;
                  // does not change job state, so release as soon as the request resolves
                  runAction(
                    'sample',
                    async () => {
                      await sampleJobNow(job.id);
                      if (onRefresh) onRefresh();
                    },
                    false,
                  );
                }}
              >
                {pending === 'sample' ? menuSpinner : <Image className="w-4 h-4" />}
                Sample Next Step
              </div>
            </MenuItem>
          )}
          {canForceStop && (
            <MenuItem disabled={isBusy}>
              <div
                className={isBusy ? menuItemDisabledClass : menuItemEnabledClass}
                onClick={() => {
                  if (isBusy) return;
                  openConfirm({
                    title: 'Force Stop Job',
                    message: `Force stop ends every process of "${job.name}" immediately, without a final save. Continue?`,
                    type: 'warning',
                    confirmText: 'Force Stop',
                    onConfirm: () =>
                      runAction('forceStop', async () => {
                        await forceStopJob(job.id);
                        onRefresh && onRefresh();
                      }),
                  });
                }}
              >
                {pending === 'forceStop' ? menuSpinner : <OctagonX className="w-4 h-4" />}
                Force Stop
              </div>
            </MenuItem>
          )}
          <MenuItem disabled={isBusy || !canMarkStopped}>
            <div
              className={isBusy || !canMarkStopped ? menuItemDisabledClass : menuItemEnabledClass}
              onClick={() => {
                if (isBusy || !canMarkStopped) return;
                let message = `Are you sure you want to mark this job as stopped? This only resets the displayed status of a job whose process is already gone; it will NOT stop anything.`;
                openConfirm({
                  title: 'Mark Job as Stopped',
                  message: message,
                  type: 'warning',
                  confirmText: 'Mark as Stopped',
                  onConfirm: () =>
                    runAction('markStopped', async () => {
                      await markJobAsStopped(job.id);
                      onRefresh && onRefresh();
                    }),
                });
              }}
            >
              {pending === 'markStopped' ? menuSpinner : <OctagonX className="w-4 h-4" />}
              Mark as Stopped
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
    </div>
  );
}
