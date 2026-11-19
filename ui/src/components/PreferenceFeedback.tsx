'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';
import { RequestOwner, Ticket } from '@/utils/requestOwner';
import { LuImage, LuLoader, LuVote } from 'react-icons/lu';

type PreferenceChoice = 'a' | 'b' | 'skip';
type PreferenceStatus = 'pending' | 'rated' | 'skipped' | 'invalid';
type TabKey = 'pending' | 'rated' | 'skipped' | 'invalid';

interface PreferenceVote {
  version: 1;
  id: string;
  choice: PreferenceChoice;
  voted_at: string;
}

interface PreferencePair {
  id: string;
  step: number;
  prompt: string;
  seeds: [number, number];
  width: number;
  height: number;
  reference_id: string;
  created_at: string;
  status: PreferenceStatus;
  vote: PreferenceVote | null;
  image_urls: { a: string; b: string };
}

interface PreferenceFeedbackList {
  enabled: boolean;
  pairs: PreferencePair[];
  counts: Record<TabKey, number>;
}

const configuredForFeedback = (job: Job): boolean => {
  try {
    const parsed = JSON.parse(job.job_config) as any;
    const feedback = parsed?.config?.process?.[0]?.train?.dpo_feedback;
    return feedback !== null && typeof feedback === 'object' && !Array.isArray(feedback);
  } catch {
    return false;
  }
};

const statusLabel: Record<PreferenceStatus, string> = {
  pending: 'Pending',
  rated: 'Rated',
  skipped: 'Skipped',
  invalid: 'Vote unavailable',
};

const choiceLabel: Record<PreferenceChoice, string> = {
  a: 'A',
  b: 'B',
  skip: 'Skip',
};

// The image endpoint is authenticated and intentionally not under the public `/api/img` exemption. Fetch the bytes
// through the bearer-aware client and render an object URL, so a plain <img> never needs to carry a secret and no
// local filesystem path reaches the browser.
function AuthenticatedPreferenceImage({ url, alt }: { url: string; alt: string }) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | null = null;
    setSrc(null);
    setFailed(false);
    apiClient
      .get(url, { responseType: 'blob', signal: controller.signal })
      .then(res => {
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(res.data);
        setSrc(objectUrl);
      })
      .catch(error => {
        if (error?.code === 'ERR_CANCELED' || controller.signal.aborted) return;
        console.error('Could not load preference image:', error);
        setFailed(true);
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);

  if (failed) {
    return (
      <div className="flex h-56 items-center justify-center rounded-lg bg-gray-100 text-xs text-gray-500 dark:bg-gray-800 dark:text-gray-400">
        Image unavailable
      </div>
    );
  }
  if (!src) {
    return (
      <div className="flex h-56 items-center justify-center rounded-lg bg-gray-100 dark:bg-gray-800">
        <LuLoader className="h-5 w-5 animate-spin text-gray-500" />
      </div>
    );
  }
  return <img src={src} alt={alt} className="h-56 w-full rounded-lg bg-black object-contain" />;
}

export default function PreferenceFeedback({ job }: { job: Job }) {
  const [feedback, setFeedback] = useState<PreferenceFeedbackList | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [tab, setTab] = useState<TabKey>('pending');
  const [votingID, setVotingID] = useState<string | null>(null);
  const [voteError, setVoteError] = useState<string | null>(null);
  const ownerRef = useRef<RequestOwner | null>(null);
  // Synchronous guard: two click events can arrive before React paints the disabled state.
  const votingRef = useRef<string | null>(null);
  if (ownerRef.current === null) ownerRef.current = new RequestOwner();

  const configured = useMemo(() => configuredForFeedback(job), [job.job_config]);

  const fetchFeedback = useCallback(
    (ticket: Ticket) => {
      const owner = ownerRef.current!;
      if (!owner.isCurrent(job.id)) return Promise.resolve();
      setStatus(prev => (prev === 'success' ? prev : 'loading'));
      return apiClient
        .get(`/api/jobs/${encodeURIComponent(job.id)}/preferences`)
        .then(res => {
          if (!owner.accept(ticket)) return;
          setFeedback(res.data as PreferenceFeedbackList);
          setStatus('success');
        })
        .catch(error => {
          if (!owner.accept(ticket)) return;
          console.error('Error fetching preference feedback:', error);
          setStatus('error');
        });
    },
    [job.id],
  );

  useEffect(() => {
    ownerRef.current!.reset(job.id);
    setFeedback(null);
    setStatus('idle');
    votingRef.current = null;
    setVotingID(null);
    setVoteError(null);
  }, [job.id]);

  usePollLoop(() => fetchFeedback(ownerRef.current!.begin()), 5000, [job.id]);

  const vote = useCallback(
    async (pair: PreferencePair, choice: PreferenceChoice) => {
      if (votingRef.current !== null) return;
      votingRef.current = pair.id;
      setVotingID(pair.id);
      setVoteError(null);
      // Supersede any list read issued before this mutation. The POST is the authority; an old poll must not put the
      // pair back into pending while the vote is in flight.
      ownerRef.current!.mutate();
      try {
        await apiClient.post(
          `/api/jobs/${encodeURIComponent(job.id)}/preferences/${encodeURIComponent(pair.id)}/vote`,
          { choice },
        );
        await fetchFeedback(ownerRef.current!.mutate());
      } catch (error: any) {
        const message =
          error?.response?.data?.error ||
          (error?.response?.status === 409
            ? 'This pair already has a different vote. Refreshing the gallery.'
            : 'The vote could not be saved. Refreshing the gallery.');
        setVoteError(message);
        await fetchFeedback(ownerRef.current!.mutate());
      } finally {
        votingRef.current = null;
        setVotingID(null);
      }
    },
    [fetchFeedback, job.id],
  );

  const shouldShow = configured || !!feedback?.enabled || (feedback?.pairs.length ?? 0) > 0;
  if (!shouldShow) return null;

  const tabs: Array<{ key: TabKey; label: string }> = [
    { key: 'pending', label: 'Pending' },
    { key: 'rated', label: 'Rated' },
    { key: 'skipped', label: 'Skipped' },
    ...(feedback && feedback.counts.invalid > 0 ? [{ key: 'invalid' as TabKey, label: 'Unavailable' }] : []),
  ];
  const visible = (feedback?.pairs ?? []).filter(pair => pair.status === tab);

  return (
    <section className="mb-4 rounded-xl border border-gray-700 bg-gray-900/70 shadow-lg">
      <div className="flex flex-wrap items-center gap-2 border-b border-gray-700 px-4 py-3">
        <LuVote className="h-4 w-4 text-amber-500" />
        <h2 className="text-sm font-semibold text-gray-100">Preference feedback</h2>
        <span className="text-xs text-gray-400">Choose the image that better matches the prompt.</span>
        <div className="flex-1" />
        {tabs.map(item => {
          const count = feedback?.counts[item.key] ?? 0;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => setTab(item.key)}
              className={`rounded px-2 py-1 text-xs ${
                tab === item.key ? 'bg-gray-700 text-white' : 'text-gray-300 hover:bg-gray-800'
              }`}
            >
              {item.label} ({count})
            </button>
          );
        })}
      </div>

      {voteError && (
        <div className="border-b border-amber-700/40 bg-amber-900/20 px-4 py-2 text-xs text-amber-200">{voteError}</div>
      )}

      {status === 'loading' && feedback === null && (
        <div className="flex items-center gap-2 px-4 py-6 text-sm text-gray-300">
          <LuLoader className="h-4 w-4 animate-spin" /> Loading preference pairs...
        </div>
      )}
      {status === 'error' && feedback === null && (
        <div className="px-4 py-6 text-sm text-red-300">Preference pairs could not be loaded.</div>
      )}

      {feedback && visible.length === 0 && (
        <div className="flex flex-col items-center justify-center px-4 py-10 text-center text-gray-300">
          <LuImage className="mb-2 h-7 w-7 text-gray-500" />
          <p className="text-sm">
            {tab === 'pending' ? 'No preference pairs are waiting for a vote.' : `No ${tab} preference pairs.`}
          </p>
          {tab === 'pending' && feedback.enabled && (
            <p className="mt-1 text-xs text-gray-500">New pairs appear here as the job publishes them.</p>
          )}
        </div>
      )}

      {feedback && visible.length > 0 && (
        <div className="grid grid-cols-1 gap-4 p-4 xl:grid-cols-2">
          {visible.map(pair => (
            <article key={pair.id} className="rounded-lg border border-gray-700 bg-gray-800/60 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-400">
                <span className="rounded bg-gray-700 px-2 py-0.5 text-gray-200">Step {pair.step}</span>
                <span>
                  seeds {pair.seeds[0]} / {pair.seeds[1]}
                </span>
                <span>
                  {pair.width}x{pair.height}
                </span>
                <span>{statusLabel[pair.status]}</span>
                {pair.vote && <span>Voted {choiceLabel[pair.vote.choice]}</span>}
              </div>
              <p className="mb-3 line-clamp-3 whitespace-pre-wrap break-words text-sm text-gray-100" title={pair.prompt}>
                {pair.prompt}
              </p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <AuthenticatedPreferenceImage url={pair.image_urls.a} alt={`Preference ${pair.id}, image A`} />
                  {pair.status === 'pending' && (
                    <button
                      type="button"
                      disabled={votingID !== null}
                      onClick={() => vote(pair, 'a')}
                      className="mt-2 w-full rounded bg-sky-700 px-3 py-1.5 text-sm text-white hover:bg-sky-600 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {votingID === pair.id ? 'Saving...' : 'Choose A'}
                    </button>
                  )}
                </div>
                <div>
                  <AuthenticatedPreferenceImage url={pair.image_urls.b} alt={`Preference ${pair.id}, image B`} />
                  {pair.status === 'pending' && (
                    <button
                      type="button"
                      disabled={votingID !== null}
                      onClick={() => vote(pair, 'b')}
                      className="mt-2 w-full rounded bg-sky-700 px-3 py-1.5 text-sm text-white hover:bg-sky-600 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {votingID === pair.id ? 'Saving...' : 'Choose B'}
                    </button>
                  )}
                </div>
              </div>
              {pair.status === 'pending' && (
                <button
                  type="button"
                  disabled={votingID !== null}
                  onClick={() => vote(pair, 'skip')}
                  className="mt-2 w-full rounded border border-gray-600 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Skip this pair
                </button>
              )}
              {pair.status === 'invalid' && (
                <p className="mt-2 text-xs text-amber-300">An existing vote record is invalid; it will not be overwritten.</p>
              )}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
