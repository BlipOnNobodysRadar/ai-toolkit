'use client';

import { useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';
import { RequestOwner, Ticket } from '@/utils/requestOwner';

interface FileObject {
  path: string;
  size: number;
}

// Responses are owned (utils/requestOwner): another job's, superseded or out-of-order responses are dropped.
export default function useFilesList(jobID: string, reloadInterval: null | number = null) {
  const [files, setFiles] = useState<FileObject[]>([]);
  const didInitialLoadRef = useRef(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');
  const ownerRef = useRef<RequestOwner | null>(null);
  if (ownerRef.current === null) ownerRef.current = new RequestOwner();

  const fetchFiles = (ticket: Ticket) => {
    const owner = ownerRef.current!;
    // a callback from an earlier render (old jobID) must not issue a request under the new identity
    if (!owner.isCurrent(jobID)) return Promise.resolve();
    let loadStatus: 'loading' | 'refreshing' = 'loading';
    if (didInitialLoadRef.current) {
      loadStatus = 'refreshing';
    }
    setStatus(loadStatus);
    return apiClient
      .get(`/api/jobs/${jobID}/files`)
      .then(res => res.data)
      .then(data => {
        if (!owner.accept(ticket)) return;
        if (data.files) {
          setFiles(data.files);
        }
        setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch(error => {
        if (!owner.accept(ticket)) return;
        console.error('Error fetching files');
        setStatus('error');
      });
  };

  const refreshFiles = () => fetchFiles(ownerRef.current!.mutate());

  useEffect(() => {
    ownerRef.current!.reset(jobID);
    didInitialLoadRef.current = false;
    setFiles([]);
  }, [jobID]);

  usePollLoop(() => fetchFiles(ownerRef.current!.begin()), reloadInterval, [jobID]);

  return { files, setFiles, status, refreshFiles };
}
