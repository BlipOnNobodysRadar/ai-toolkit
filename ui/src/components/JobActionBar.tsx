import Link from 'next/link';
import { useRef, useState } from 'react';
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
  confirmJobStopped,
  saveJobNow,
  sampleJobNow,
  forceStopJob,
  actionError,
  ownershipOf,
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

  const [pending, setPending] = useState<PendingAction | null>(null);
  const inFlight = useRef(false);
  const runAction = async (action: PendingAction, fn: () => Promise<void>) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setPending(action);
    try {
      await fn();
    } catch (e) {
      console.error(`Job action ${action} failed`);
      alert(actionError(e));
    } finally {
      inFlight.current = false;
      setPending(null);
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
          aria-label="Queue job"
          title="Queue job"
          disabled={isBusy}
          onClick={() => {
            if (!canStart) return;
            runAction('start', async () => {
              await startJob(job.id);
              // start the queue as well
              if (autoStartQueue) {
                await startQueue(job.gpu_ids);
              }
              await onRefresh?.();
            });
          }}
          className={actionButtonClass}
        >
          {pending === 'start' ? spinner : <Play className={iconSizeClass} />}
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          aria-label="Remove from queue"
          title="Remove from queue"
          disabled={isBusy}
          onClick={() => {
            if (!canRemoveFromQueue) return;
            const remove = () => runAction('remove', async () => {
              await confirmJobStopped(job);
              await onRefresh?.();
            });
            if (ownershipOf(job).held) {
              openConfirm({
                title: 'Remove from Queue',
                message: `Confirm that "${job.name}" has finished. This removes it from the queue and keeps its checkpoints.`,
                type: 'warning',
                confirmText: 'Remove from Queue',
                onConfirm: remove,
              });
            } else {
              remove();
            }
          }}
          className={actionButtonClass}
        >
          {pending === 'remove' ? spinner : <X className={iconSizeClass} />}
        </Button>
      )}
      {canStop && (
        <Button
          aria-label="Stop job"
          title="Stop job"
          disabled={isBusy}
          onClick={() => {
            if (!canStop || isBusy) return;
            openConfirm({
              title: 'Stop Job',
              message: job.job_type === 'train'
                ? `Stop "${job.name}"? Training can resume from its last saved checkpoint. Steps since that checkpoint will run again. Without a checkpoint, training starts over.`
                : `Stop "${job.name}"?`,
              type: 'info',
              confirmText: 'Stop',
              onConfirm: () =>
                runAction('stop', async () => {
                  await stopJob(job.id);
                  await onRefresh?.();
                }),
            });
          }}
          className={actionButtonClass}
        >
          {pending === 'stop' ? spinner : <Pause className={iconSizeClass} />}
        </Button>
      )}
      {!hideView && (
        <Link aria-label="View job" title="View job" href={`/jobs/${job.id}`} className="ml-1 sm:ml-2 text-gray-200 hover:text-gray-100 inline-block">
          <Eye className={iconSizeClass} />
        </Link>
      )}
      {job.job_type === 'caption' && canEdit && (
        <div
          className="ml-1 sm:ml-2 hover:text-gray-100 inline-block cursor-pointer"
          onClick={() =>
            openCaptionDatasetModal(
              job.job_ref || '',
              async () => {
                await onRefresh?.();
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
            message: ownershipOf(job).held ? `Confirm that ${job.name} has finished. Deleting removes the job and its output folder, including checkpoints.` : message,
            type: 'warning',
            confirmText: isRetryDelete ? 'Retry Delete' : 'Delete',
            onConfirm: () =>
              runAction('delete', async () => {
                const result = await deleteJob(job.id, job);
                if (result.keptOutput) alert(result.keptOutput);
                await afterDelete?.();
              }),
          });
        }}
        className={actionButtonClass}
      >
        {pending === 'delete' ? spinner : <Trash2 className={iconSizeClass} />}
      </Button>
      <div className="border-r border-1 border-gray-700 ml-1 sm:ml-2 inline"></div>
      <Menu>
        <MenuButton aria-label="Job options" title="Job options" className="ml-1 sm:ml-2">{menuPending ? spinner : <Cog className={iconSizeClass} />}</MenuButton>
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
                  runAction('save', async () => {
                    await saveJobNow(job.id);
                    await onRefresh?.();
                  });
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
                  runAction('sample', async () => {
                    await sampleJobNow(job.id);
                    await onRefresh?.();
                  });
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
                        await onRefresh?.();
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
                const message = `Confirm that "${job.name}" has finished. This releases its queue reservation and keeps its checkpoints. If it is still training, use Stop instead.`;
                openConfirm({
                  title: 'Mark Job as Stopped',
                  message: message,
                  type: 'warning',
                  confirmText: 'Mark as Stopped',
                  onConfirm: () =>
                    runAction('markStopped', async () => {
                      await confirmJobStopped(job);
                      await onRefresh?.();
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
