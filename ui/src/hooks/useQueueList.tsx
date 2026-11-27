'use client';

import { useRef, useState } from 'react';
import usePollLoop from '@/hooks/usePollLoop';
import { Queue } from '@prisma/client';
import { apiClient } from '@/utils/api';
import { RequestOwner } from '@/utils/requestOwner';

// Manual refreshes may overlap (e.g. after queue start/stop clicks); only the latest issued one is applied.
export default function useQueueList(reloadInterval: number | null = null) {
  const [queues, setQueues] = useState<Queue[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const ownerRef = useRef<RequestOwner | null>(null);
  if (ownerRef.current === null) ownerRef.current = new RequestOwner();

  const refreshQueues = () => {
    const owner = ownerRef.current!;
    const ticket = owner.mutate();
    setStatus('loading');
    return apiClient
      .get('/api/queue')
      .then(res => res.data)
      .then(data => {
        if (!owner.accept(ticket)) return;
        if (data.error) {
          console.log('Error fetching queues:', data.error);
          setStatus('error');
        } else {
          setQueues(data.queues);
          setStatus('success');
        }
      })
      .catch(error => {
        if (!owner.accept(ticket)) return;
        console.error('Error fetching queues:', error);
        setStatus('error');
      });
  };
  usePollLoop(refreshQueues, reloadInterval);

  return { queues, setQueues, status, refreshQueues };
}
