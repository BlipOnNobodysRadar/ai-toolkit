import { useMemo, useState } from 'react';
import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { isMac } from '@/helpers/basic';
import JobActionBar from './JobActionBar';
import { Job, Queue } from '@prisma/client';
import useQueueList from '@/hooks/useQueueList';
import classNames from 'classnames';
import { startQueue, stopQueue } from '@/utils/queue';
import { CgSpinner } from 'react-icons/cg';
import useGPUInfo from '@/hooks/useGPUInfo';
import { openConfirm } from '@/components/ConfirmModal';
import { actionError, deleteJob, displayStatus, getAvaliableJobActions, getTotalSteps, ownershipOf } from '@/utils/jobs';
import { Trash2 } from 'lucide-react';

interface JobsTableProps {
  autoStartQueue?: boolean;
  onlyActive?: boolean;
  job_type?: string | null;
}

export default function JobsTable({ onlyActive = false, job_type = null }: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList({ onlyActive, reloadInterval: 5000, job_type });
  const { queues, refreshQueues } = useQueueList(5000);
  const { gpuList } = useGPUInfo();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleteProgress, setDeleteProgress] = useState<{ done: number; total: number } | null>(null);

  const refresh = () => {
    refreshJobs();
    refreshQueues();
  };

  const isDeleting = deleteProgress !== null;
  const allSelected = jobs.length > 0 && jobs.every(job => selectedIds.has(job.id));

  const toggleRow = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleAll = () => {
    setSelectedIds(allSelected ? new Set() : new Set(jobs.map(job => job.id)));
  };

  const onMassDelete = () => {
    const jobsToDelete = jobs.filter(job => selectedIds.has(job.id));
    if (jobsToDelete.length === 0) return;
    // Use the same confirmation and server-side process check as the single-job Delete action.
    const deletable = jobsToDelete.filter(job => getAvaliableJobActions(job).canDelete);
    const owned = jobsToDelete.filter(job => !deletable.includes(job));
    const retries = deletable.filter(job => getAvaliableJobActions(job).isRetryDelete).length;
    let message = `Are you sure you want to delete ${deletable.length} job${
      deletable.length === 1 ? '' : 's'
    }? This will also permanently remove them from your disk.`;
    if (deletable.some(job => ownershipOf(job).held)) {
      message += ' Confirm that the selected runs have finished. Jobs with a known live process will be kept.';
    }
    if (retries > 0) {
      message += ` ${retries} of them ${retries === 1 ? 'is a deletion' : 'are deletions'} that did not finish and will be retried.`;
    }
    if (owned.length > 0) {
      message += ` ${owned.length} selected job${owned.length === 1 ? ' is' : 's are'} still running or exiting and will be skipped; stop ${owned.length === 1 ? 'it' : 'them'} first.`;
    }
    openConfirm({
      title: 'Delete Jobs',
      message: message,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        setDeleteProgress({ done: 0, total: deletable.length });
        const refused: string[] = [];
        const keptFiles: string[] = [];
        for (let i = 0; i < deletable.length; i++) {
          const job = deletable[i];
          try {
            const result = await deleteJob(job.id, job);
            if (result.keptOutput) keptFiles.push(`${job.name}: ${result.keptOutput}`);
            setSelectedIds(prev => {
              const next = new Set(prev);
              next.delete(job.id);
              return next;
            });
          } catch (e) {
            console.error('Error deleting job');
            refused.push(`${job.name}: ${actionError(e)}`);
          }
          setDeleteProgress({ done: i + 1, total: deletable.length });
          refreshJobs();
        }
        const notes = [...(refused.length ? [`Not deleted:\n${refused.join('\n')}`] : []), ...keptFiles];
        if (notes.length) alert(notes.join('\n'));
        setDeleteProgress(null);
        refresh();
      },
    });
  };

  const columns: TableColumn[] = [
    {
      title: (
        <input
          type="checkbox"
          checked={allSelected}
          onChange={toggleAll}
          disabled={isDeleting}
          className="cursor-pointer accent-blue-500"
        />
      ),
      key: 'select',
      className: 'w-8',
      render: row => (
        <input
          type="checkbox"
          checked={selectedIds.has(row.id)}
          onChange={() => toggleRow(row.id)}
          disabled={isDeleting}
          className="cursor-pointer accent-blue-500"
        />
      ),
    },
    {
      title: 'Name',
      key: 'name',
      render: row => {
        let title = row.name;
        let href = `/jobs/${row.id}`;
        // if (row.job_type === 'train') title = `Train: ${title}`;
        if (row.job_type === 'inference') {
          href = `/generate?job=${row.id}`;
          title = (
            <>
              <small className="opacity-50">INFERENCE ENGINE: </small> {row.name}
            </>
          );
        }
        if (row.job_type === 'caption') {
          const datasetPath = row.job_ref?.split(/[/\\]/).filter(Boolean).pop();
          if (datasetPath) href = `/datasets/${encodeURIComponent(datasetPath)}`;
          title = (
            <>
              <small className="opacity-50">CAPTION: </small> {datasetPath || row.name}
            </>
          );
        }
        return (
          <Link href={href} className="font-medium whitespace-nowrap">
            {['starting', 'running', 'stopping'].includes(row.status) || ownershipOf(row).held ? (
              <CgSpinner className="inline animate-spin mr-2 text-blue-400" />
            ) : null}
            {title}
          </Link>
        );
      },
    },
    {
      title: 'Steps',
      key: 'steps',
      render: row => {
        const totalSteps = getTotalSteps(row);
        if (!totalSteps) {
          return <></>;
        }

        return (
          <div>
            <div className="text-xs text-gray-400">
              {row.step} / {totalSteps}
            </div>
            <div className="bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full"
                style={{ width: `${Math.min(100, (row.step / totalSteps) * 100)}%` }}
              ></div>
            </div>
          </div>
        );
      },
    },
    {
      title: 'GPU',
      key: 'gpu_ids',
    },
    {
      title: 'Status',
      key: 'status',
      render: row => {
        let statusClass = 'text-gray-400';
        if (row.status === 'completed') statusClass = 'text-green-400';
        if (row.status === 'error' || row.status === 'delete_failed') statusClass = 'text-red-400';
        if (row.status === 'running' || row.status === 'starting') statusClass = 'text-blue-400';
        if (row.status === 'stopping' || row.status === 'deleting') statusClass = 'text-orange-400';

        // the ownership note explains a retained reservation or a wait (hover)
        return (
          <span className={statusClass} title={row.ownership_note ?? undefined}>
            {displayStatus(row)}
            {row.ownership_note ? ' *' : ''}
          </span>
        );
      },
    },
    {
      title: 'Info',
      key: 'info',
      className: 'truncate max-w-xs',
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'text-right',
      render: row => {
        return <JobActionBar job={row} onRefresh={refreshJobs} autoStartQueue={false} />;
      },
    },
  ];

  // The worker addresses a queue by its complete gpu_ids string (including e.g. "1,0", "-1" and "mps").
  // Hardware monitoring only supplies display names. It may be unavailable while the jobs API is healthy.
  const { queueGroups, idleJobs } = useMemo(() => {
    const groups = new Map<string, { name: string; jobs: Job[] }>();
    const ensureQueue = (id: string) => {
      if (!groups.has(id)) {
        const name = id === '-1' ? 'CPU' : id === 'mps' ? 'Apple GPU' : `GPU ${id}`;
        groups.set(id, { name, jobs: [] });
      }
      return groups.get(id)!;
    };
    gpuList.forEach(gpu => {
      ensureQueue(isMac() ? 'mps' : String(gpu.index)).name = gpu.name;
    });
    queues.forEach(queue => ensureQueue(queue.gpu_ids));
    const idleJobs: Job[] = [];
    jobs.forEach(job => {
      if (['queued', 'starting', 'running', 'stopping'].includes(job.status) || ownershipOf(job).held) {
        ensureQueue(job.gpu_ids).jobs.push(job);
      } else {
        idleJobs.push(job);
      }
    });
    groups.forEach(group => group.jobs.sort((a, b) => a.queue_position - b.queue_position));
    idleJobs.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
    const queueGroups = [...groups.entries()].sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }));
    return { queueGroups, idleJobs };
  }, [jobs, queues, gpuList]);

  const isLoading = jobs.length === 0 && (status === 'idle' || status === 'loading');
  const changeQueue = async (id: string, running: boolean) => {
    try {
      await (running ? stopQueue(id) : startQueue(id));
    } catch (error) {
      alert(actionError(error));
    }
    refresh();
  };

  return (
    <div>
      {(selectedIds.size > 0 || isDeleting) && (
        <div className="fixed top-16 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-4 py-2 bg-gray-800 rounded-lg border border-gray-700 shadow-lg">
          {isDeleting ? (
            <>
              <CgSpinner className="inline animate-spin text-red-400" />
              <span className="text-sm text-gray-300">
                Deleting {deleteProgress.done} / {deleteProgress.total}...
              </span>
            </>
          ) : (
            <>
              <span className="text-sm text-gray-300 flex-1">
                {selectedIds.size} job{selectedIds.size === 1 ? '' : 's'} selected
              </span>
              <button
                onClick={() => setSelectedIds(new Set())}
                className="text-xs text-gray-300 bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded"
              >
                Clear
              </button>
              <button
                onClick={onMassDelete}
                className="text-xs text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded flex items-center gap-1"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Delete Selected
              </button>
            </>
          )}
        </div>
      )}
      {status === 'error' && <p role="alert">Could not load jobs. <button onClick={refresh}>Retry</button></p>}
      {queueGroups.map(([gpuKey, group]) => {
        const queue = queues.find(q => `${q.gpu_ids}` === gpuKey) as Queue;
        return (
          <div key={gpuKey} className="mb-6">
            <div
              className={classNames(
                'text-md flex flex-wrap gap-y-1 px-2 sm:px-4 py-1 rounded-t-lg',
                { 'bg-green-600 dark:bg-green-900': queue?.is_running },
                { 'bg-red-600 dark:bg-red-900': !queue?.is_running },
              )}
            >
              <div className="flex items-center space-x-2 flex-1 min-w-0 py-2">
                <h2 className="font-semibold text-white truncate">{group.name}</h2>
                <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300 flex-shrink-0">
                  # {gpuKey}
                </span>
              </div>
              <div className="text-sm text-gray-300 italic flex items-center flex-shrink-0">
                {queue?.is_running ? (
                  <>
                    <span className="text-green-100 dark:text-green-400 mr-2">Queue Running</span>
                    <button
                      onClick={() => changeQueue(gpuKey, true)}
                      className="ml-2 sm:ml-4 text-xs text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded"
                    >
                      STOP
                    </button>
                  </>
                ) : (
                  <>
                    <span className="text-red-100 dark:text-red-400 mr-2">Queue Stopped</span>
                    <button
                      onClick={() => changeQueue(gpuKey, false)}
                      className="ml-2 sm:ml-4 text-xs text-white bg-green-600 hover:bg-green-700 px-2 py-1 rounded"
                    >
                      START
                    </button>
                  </>
                )}
              </div>
            </div>
            <UniversalTable
              columns={columns}
              rows={group.jobs}
              isLoading={isLoading}
              onRefresh={refresh}
              theadClassName={
                queue?.is_running
                  ? 'bg-green-700 dark:bg-green-950 text-white dark:text-gray-400'
                  : 'bg-red-700 dark:bg-red-950 text-white dark:text-gray-400'
              }
            />
          </div>
        );
      })}
      {!onlyActive && (
        <div className="mb-6 opacity-50">
          <div className="text-md flex px-4 py-1 rounded-t-lg bg-slate-600">
            <div className="flex items-center space-x-2 flex-1 py-2">
              <h2 className="font-semibold text-gray-100">Idle</h2>
            </div>
          </div>
          <UniversalTable columns={columns} rows={idleJobs} isLoading={isLoading} onRefresh={refresh} />
        </div>
      )}
    </div>
  );
}
