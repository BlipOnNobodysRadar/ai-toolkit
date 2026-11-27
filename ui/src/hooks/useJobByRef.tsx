'use client';

import { useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';
import { RequestOwner, Ticket } from '@/utils/requestOwner';

// Snapshot of the latest job for a job_ref (e.g. a dataset being captioned). Responses are owned
// (utils/requestOwner): a response for a previous ref, one issued before a manual refresh, or one older than a
// response already shown is dropped, so a slow poll can never put an older status back on screen.
export default function useJobByRef(jobRef: string | null, reloadInterval: null | number = null) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const ownerRef = useRef<RequestOwner | null>(null);
  if (ownerRef.current === null) ownerRef.current = new RequestOwner();

  const fetchJob = (ticket: Ticket) => {
    const owner = ownerRef.current!;
    // a callback from an earlier render (old jobRef) must not issue a request under the new identity
    if (!owner.isCurrent(jobRef)) return Promise.resolve();
    if (!jobRef) {
      setJob(null);
      setStatus('idle');
      return Promise.resolve();
    }
    setStatus('loading');
    return apiClient
      .get(`/api/jobs?job_ref=${encodeURIComponent(jobRef)}`)
      .then(res => res.data)
      .then(data => {
        if (!owner.accept(ticket)) return;
        setJob(data);
        setStatus('success');
      })
      .catch(error => {
        if (!owner.accept(ticket)) return;
        console.error('Error fetching job:', error);
        setStatus('error');
      });
  };

  // called after an action changed the job: supersedes every request issued before it
  const refreshJob = () => fetchJob(ownerRef.current!.mutate());

  // New ref: drop the old job and anything still in flight for it. Declared before the poll loop so it runs
  // before the first fetch for the new ref.
  useEffect(() => {
    ownerRef.current!.reset(jobRef);
    setJob(null);
  }, [jobRef]);

  usePollLoop(() => fetchJob(ownerRef.current!.begin()), reloadInterval, [jobRef]);

  return { job, setJob, status, refreshJob };
}
