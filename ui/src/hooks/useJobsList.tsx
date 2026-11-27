'use client';

import { useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';
import { RequestOwner, Ticket } from '@/utils/requestOwner';

type UseJobsListProps = {
  onlyActive?: boolean;
  reloadInterval?: number | null;
  job_type?: string | null;
};

// Responses are owned (utils/requestOwner): a manual refresh (after an action) supersedes every request issued
// before it, and an out-of-order response never replaces a newer list.
export default function useJobsList({
  onlyActive = false,
  reloadInterval = null,
  job_type = null,
}: UseJobsListProps = {}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const ownerRef = useRef<RequestOwner | null>(null);
  if (ownerRef.current === null) ownerRef.current = new RequestOwner();
  const filterKey = `${onlyActive ? 1 : 0}|${job_type ?? ''}`;

  const fetchJobs = (ticket: Ticket) => {
    const owner = ownerRef.current!;
    // a callback from an earlier render (old filter) must not issue a request under the new identity
    if (!owner.isCurrent(filterKey)) return Promise.resolve();
    setStatus('loading');
    const params: Record<string, string> = {};
    if (job_type) {
      params.job_type = job_type;
    }
    if (onlyActive) {
      params.only_active = 'true';
    }
    return apiClient
      .get('/api/jobs', { params })
      .then(res => res.data)
      .then(data => {
        if (!owner.accept(ticket)) return;
        if (data.error) {
          setStatus('error');
        } else {
          setJobs(data.jobs);
          setStatus('success');
        }
      })
      .catch(error => {
        if (!owner.accept(ticket)) return;
        console.error('Error fetching jobs');
        setStatus('error');
      });
  };

  const refreshJobs = () => fetchJobs(ownerRef.current!.mutate());

  useEffect(() => {
    ownerRef.current!.reset(filterKey);
  }, [filterKey]);

  usePollLoop(() => fetchJobs(ownerRef.current!.begin()), reloadInterval, [filterKey]);

  return { jobs, setJobs, status, refreshJobs };
}
