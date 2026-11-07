'use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';
import { RequestOwner, Ticket } from '@/utils/requestOwner';

// Responses are owned (utils/requestOwner): another job's, superseded or out-of-order responses are dropped.
export default function useSampleImages(jobID: string, reloadInterval: null | number = null) {
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const ownerRef = useRef<RequestOwner | null>(null);
  if (ownerRef.current === null) ownerRef.current = new RequestOwner();

  const fetchSamples = (ticket: Ticket) => {
    const owner = ownerRef.current!;
    // a callback from an earlier render (old jobID) must not issue a request under the new identity
    if (!owner.isCurrent(jobID)) return Promise.resolve();
    setStatus('loading');
    return apiClient
      .get(`/api/jobs/${jobID}/samples`)
      .then(res => res.data)
      .then(data => {
        if (!owner.accept(ticket)) return;
        if (data.samples) {
          setSampleImages(data.samples);
        }
        setStatus('success');
      })
      .catch(error => {
        if (!owner.accept(ticket)) return;
        console.error('Error fetching sample images:', error);
        setStatus('error');
      });
  };

  const refreshSampleImages = () => fetchSamples(ownerRef.current!.mutate());

  useEffect(() => {
    ownerRef.current!.reset(jobID);
    setSampleImages([]);
  }, [jobID]);

  usePollLoop(() => fetchSamples(ownerRef.current!.begin()), reloadInterval, [jobID]);

  return { sampleImages, setSampleImages, status, refreshSampleImages };
}
