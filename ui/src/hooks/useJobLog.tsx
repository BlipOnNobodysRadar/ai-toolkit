'use client';

import { useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';
import { TerminalEmulator } from '@/utils/terminalEmulator';
import usePollLoop from '@/hooks/usePollLoop';
import { RequestOwner, Ticket } from '@/utils/requestOwner';

export default function useJobLog(jobID: string, reloadInterval: null | number = null) {
  const [log, setLog] = useState<string>('');
  // Emulates a terminal over the raw log stream so carriage returns, cursor
  // movement, and erase sequences collapse lines like a real terminal.
  const terminalRef = useRef<TerminalEmulator | null>(null);
  if (terminalRef.current === null) {
    terminalRef.current = new TerminalEmulator();
  }
  const didInitialLoadRef = useRef(false);
  // Cursor into the log: byte offset already consumed, and the identity of the log file it refers to (a new run
  // rotates log.txt; an offset into the old file means nothing in the new one).
  const offsetRef = useRef<number | null>(null);
  const generationRef = useRef<string | null>(null);
  // One cursor read in flight per job: two reads from the same offset would append the same bytes twice. The slot
  // belongs to the job (identity) that issued it; an old job's request never holds or releases the new job's slot.
  const inFlightRef = useRef<{ ticket: Ticket; promise: Promise<void> } | null>(null);
  const ownerRef = useRef<RequestOwner | null>(null);
  if (ownerRef.current === null) ownerRef.current = new RequestOwner();
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = (): Promise<void> => {
    const owner = ownerRef.current!;
    // a callback from an earlier render (old jobID) must not issue a request under the new identity
    if (!owner.isCurrent(jobID)) return Promise.resolve();
    const current = inFlightRef.current;
    if (current && owner.valid(current.ticket)) {
      return current.promise;
    }
    const ticket = owner.begin();
    setStatus(didInitialLoadRef.current ? 'refreshing' : 'loading');
    const offset = offsetRef.current;
    const generation = generationRef.current;
    const params = offset !== null ? { offset, ...(generation ? { generation } : {}) } : undefined;
    const promise = apiClient
      .get(`/api/jobs/${jobID}/log`, params ? { params } : undefined)
      .then(res => res.data)
      .then(data => {
        if (!owner.accept(ticket)) return;
        const terminal = terminalRef.current!;
        if (data.reset) {
          // new log file, truncated log, or initial load: replace everything
          terminal.reset();
        }
        offsetRef.current = data.offset ?? null;
        generationRef.current = data.generation ?? null;
        terminal.write(data.log ?? '');
        setLog(terminal.toString());
        setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch(error => {
        if (!owner.accept(ticket)) return;
        console.error('Error fetching log:', error);
        setStatus('error');
      })
      .finally(() => {
        if (inFlightRef.current?.ticket === ticket) inFlightRef.current = null;
      });
    inFlightRef.current = { ticket, promise };
    return promise;
  };

  // New job — start fresh. Declared before the poll loop so the reset runs
  // before the first fetch when jobID changes.
  useEffect(() => {
    ownerRef.current!.reset(jobID);
    inFlightRef.current = null;
    offsetRef.current = null;
    generationRef.current = null;
    didInitialLoadRef.current = false;
    terminalRef.current?.reset();
    setLog('');
  }, [jobID]);

  usePollLoop(refresh, reloadInterval, [jobID]);

  return { log, status, refresh };
}
