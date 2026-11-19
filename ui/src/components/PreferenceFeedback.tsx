'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';
import { RequestOwner, Ticket } from '@/utils/requestOwner';
import { LuChevronLeft, LuChevronRight, LuDownload, LuImage, LuLoader, LuMaximize2, LuVote, LuX } from 'react-icons/lu';

type PreferenceChoice = 'a' | 'b' | 'skip';
type PreferenceStatus = 'pending' | 'rated' | 'skipped' | 'invalid';
type TabKey = 'pending' | 'rated' | 'skipped' | 'invalid';
type ImageArm = 'a' | 'b';

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

interface PreferenceTrainingMetric {
  value: number;
  step: number;
}

interface PreferenceFeedbackTrainingLog {
  preference_updates: PreferenceTrainingMetric | null;
  optimizer_updates: PreferenceTrainingMetric | null;
  foreign_votes: PreferenceTrainingMetric | null;
}

interface PreferenceFeedbackList {
  enabled: boolean;
  pairs: PreferencePair[];
  counts: Record<TabKey, number>;
  training?: PreferenceFeedbackTrainingLog;
}

interface PreferenceComparison {
  pair: PreferencePair;
  arm: ImageArm;
}

const EMPTY_SELECTION: Record<TabKey, string | null> = {
  pending: null,
  rated: null,
  skipped: null,
  invalid: null,
};

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
function AuthenticatedPreferenceImage({
  url,
  alt,
  width,
  height,
  size = 'gallery',
}: {
  url: string;
  alt: string;
  width: number;
  height: number;
  size?: 'gallery' | 'dialog';
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const mediaClass = size === 'dialog' ? 'min-h-64 max-h-[calc(100vh-13rem)]' : 'min-h-48 max-h-[68vh]';
  const stateClass = `flex w-full items-center justify-center rounded-lg bg-black ${mediaClass}`;

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
    return <div className={`${stateClass} text-xs text-gray-400`}>Image unavailable</div>;
  }
  if (!src) {
    return (
      <div className={stateClass}>
        <LuLoader className="h-5 w-5 animate-spin text-gray-500" />
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={alt}
      width={width}
      height={height}
      className={`block h-auto w-full rounded-lg bg-black object-contain ${mediaClass}`}
    />
  );
}

export default function PreferenceFeedback({ job }: { job: Job }) {
  const [feedback, setFeedback] = useState<PreferenceFeedbackList | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [tab, setTab] = useState<TabKey>('pending');
  const [selectedByTab, setSelectedByTab] = useState<Record<TabKey, string | null>>(EMPTY_SELECTION);
  const [comparison, setComparison] = useState<PreferenceComparison | null>(null);
  const [votingID, setVotingID] = useState<string | null>(null);
  const [voteError, setVoteError] = useState<string | null>(null);
  const [downloadingExport, setDownloadingExport] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportNotice, setExportNotice] = useState<string | null>(null);
  const ownerRef = useRef<RequestOwner | null>(null);
  const exportAbortRef = useRef<AbortController | null>(null);
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
    setTab('pending');
    setSelectedByTab({ ...EMPTY_SELECTION });
    setComparison(null);
    votingRef.current = null;
    setVotingID(null);
    setVoteError(null);
    exportAbortRef.current?.abort();
    exportAbortRef.current = null;
    setDownloadingExport(false);
    setExportError(null);
    setExportNotice(null);
  }, [job.id]);

  useEffect(() => () => exportAbortRef.current?.abort(), []);

  usePollLoop(() => fetchFeedback(ownerRef.current!.begin()), 5000, [job.id]);

  const pairs = feedback?.pairs ?? [];
  const visible = pairs.filter(pair => pair.status === tab);
  const selectedID = selectedByTab[tab];
  const selectedIndex = Math.max(
    0,
    visible.findIndex(pair => pair.id === selectedID),
  );
  const focusedPair = visible[selectedIndex] ?? null;

  const selectPairAt = useCallback(
    (index: number) => {
      if (visible.length === 0) return;
      const nextIndex = Math.max(0, Math.min(index, visible.length - 1));
      setSelectedByTab(prev => ({ ...prev, [tab]: visible[nextIndex].id }));
    },
    [tab, visible],
  );

  const vote = useCallback(
    async (pair: PreferencePair, choice: PreferenceChoice) => {
      if (votingRef.current !== null) return;
      votingRef.current = pair.id;
      setVotingID(pair.id);
      setVoteError(null);
      // Supersede any list read issued before this mutation. The POST is the authority; an old poll must not put the
      // pair back into pending while the vote is in flight.
      ownerRef.current!.mutate();
      const currentIndex = visible.findIndex(item => item.id === pair.id);
      const remaining = visible.filter(item => item.id !== pair.id);
      const nextPairID = remaining[Math.min(currentIndex, remaining.length - 1)]?.id ?? null;
      try {
        await apiClient.post(
          `/api/jobs/${encodeURIComponent(job.id)}/preferences/${encodeURIComponent(pair.id)}/vote`,
          { choice },
        );
        await fetchFeedback(ownerRef.current!.mutate());
        if (tab === 'pending') {
          setSelectedByTab(prev => ({ ...prev, pending: nextPairID }));
        }
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
    [fetchFeedback, job.id, tab, visible],
  );

  const ratedCount = feedback?.counts.rated ?? 0;
  const exportRatedPairs = useCallback(async () => {
    if (downloadingExport || ratedCount === 0) return;
    const controller = new AbortController();
    exportAbortRef.current?.abort();
    exportAbortRef.current = controller;
    setDownloadingExport(true);
    setExportError(null);
    setExportNotice(null);
    try {
      // Use the app's bearer-aware client, matching report downloads. The server streams the ZIP; the browser holds
      // only the downloaded Blob needed for a normal authenticated download.
      const response = await apiClient.get(`/api/jobs/${encodeURIComponent(job.id)}/preferences/export`, {
        responseType: 'blob',
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      const blob = response.data instanceof Blob ? response.data : new Blob([response.data], { type: 'application/zip' });
      const href = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = href;
      link.download = 'preference-feedback.zip';
      link.rel = 'noreferrer';
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(href), 0);
      setExportNotice('Rated pairs downloaded.');
    } catch (error: any) {
      if (error?.code === 'ERR_CANCELED' || controller.signal.aborted) return;
      let message = 'The rated pairs could not be exported.';
      const data = error?.response?.data;
      if (data instanceof Blob) {
        try {
          const parsed = JSON.parse(await data.text());
          if (typeof parsed?.error === 'string') message = parsed.error;
        } catch {
          // Keep the clear generic message for a non-JSON error body.
        }
      } else if (typeof data?.error === 'string') {
        message = data.error;
      }
      setExportError(message);
    } finally {
      if (exportAbortRef.current === controller) exportAbortRef.current = null;
      setDownloadingExport(false);
    }
  }, [downloadingExport, job.id, ratedCount]);
  const shouldShow = configured || !!feedback?.enabled || (feedback?.pairs.length ?? 0) > 0;
  if (!shouldShow) return null;

  const training = feedback?.training ?? null;
  const hasLoggedProgress = !!training && !!(training.preference_updates || training.optimizer_updates);
  const recordedVoteCount = (feedback?.counts.rated ?? 0) + (feedback?.counts.skipped ?? 0);

  const tabs: Array<{ key: TabKey; label: string }> = [
    { key: 'pending', label: 'Pending' },
    { key: 'rated', label: 'Rated' },
    { key: 'skipped', label: 'Skipped' },
    ...(feedback && feedback.counts.invalid > 0 ? [{ key: 'invalid' as TabKey, label: 'Unavailable' }] : []),
  ];
  const comparePair = comparison?.pair ?? null;

  return (
    <section
      className="mb-4 rounded-xl border border-gray-700 bg-gray-900/70 shadow-lg"
      aria-labelledby="preference-feedback-title"
    >
      <div className="border-b border-gray-700 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <LuVote className="h-4 w-4 text-amber-500" />
          <h2 id="preference-feedback-title" className="text-sm font-semibold text-gray-100">
            Preference feedback
          </h2>
          <span className="text-xs text-gray-400">
            Review one pair at a time. Click either image to compare full size.
          </span>
          <div className="flex-1" />
          <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Preference pair status">
            {tabs.map(item => {
              const count = feedback?.counts[item.key] ?? 0;
              return (
                <button
                  key={item.key}
                  type="button"
                  aria-pressed={tab === item.key}
                  onClick={() => {
                    setTab(item.key);
                    setComparison(null);
                    setVoteError(null);
                  }}
                  className={`rounded px-2 py-1 text-xs ${
                    tab === item.key ? 'bg-gray-700 text-white' : 'text-gray-300 hover:bg-gray-800'
                  }`}
                >
                  {item.label} ({count})
                </button>
              );
            })}
          </div>
          <button
            type="button"
            title={
              ratedCount > 0
                ? 'Download rated A/B pairs as a chosen/rejected ZIP for offline DPO'
                : 'Rate at least one A/B pair to enable export'
            }
            aria-label="Download rated preference pairs for offline DPO"
            disabled={ratedCount === 0 || downloadingExport}
            onClick={() => void exportRatedPairs()}
            className="inline-flex items-center gap-1.5 rounded border border-gray-600 px-2 py-1 text-xs text-gray-200 hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {downloadingExport ? (
              <LuLoader className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <LuDownload className="h-3.5 w-3.5" />
            )}
            {downloadingExport ? 'Exporting...' : 'Export DPO ZIP'}
          </button>
        </div>
      </div>

      {voteError && (
        <div role="alert" className="border-b border-amber-700/40 bg-amber-900/20 px-4 py-2 text-xs text-amber-200">
          {voteError}
        </div>
      )}
      {exportError && (
        <div role="alert" className="border-b border-red-700/40 bg-red-900/20 px-4 py-2 text-xs text-red-200">
          {exportError}
        </div>
      )}
      {exportNotice && (
        <div role="status" className="border-b border-gray-700 px-4 py-2 text-xs text-gray-400">
          {exportNotice}
        </div>
      )}

      {training && (
        <div className="border-b border-gray-700 bg-gray-900/40 px-4 py-3" aria-label="Latest training log">
          <h3 className="text-xs font-semibold text-gray-300">Latest training log</h3>
          <dl className="mt-1 flex flex-wrap gap-x-5 gap-y-1 text-xs">
            {[
              { label: 'Preference updates', metric: training.preference_updates },
              { label: 'Total optimizer updates', metric: training.optimizer_updates },
              { label: 'Votes incompatible with the reference', metric: training.foreign_votes },
            ].map(({ label, metric }) => metric && (
              <div key={label} className="flex items-baseline gap-1.5">
                <dt className="text-gray-400">{label}</dt>
                <dd className="text-gray-100">
                  {metric.value} <span className="text-gray-500">at log step {metric.step}</span>
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {feedback && !hasLoggedProgress && recordedVoteCount > 0 && (
        <div role="status" className="border-b border-gray-700 px-4 py-2 text-xs text-gray-400">
          Vote saved. Waiting for a training log entry.
        </div>
      )}
      <div id="preference-feedback-panel">
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

        {feedback && focusedPair && (
          <div className="p-4">
            <div className="mb-3 flex items-center justify-center gap-3">
              <button
                type="button"
                title="Previous preference pair"
                aria-label="Previous preference pair"
                disabled={selectedIndex <= 0}
                onClick={() => selectPairAt(selectedIndex - 1)}
                className="rounded border border-gray-600 p-1.5 text-gray-300 hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <LuChevronLeft className="h-4 w-4" />
              </button>
              <span className="min-w-32 text-center text-xs text-gray-400" aria-live="polite">
                Pair {selectedIndex + 1} of {visible.length}
              </span>
              <button
                type="button"
                title="Next preference pair"
                aria-label="Next preference pair"
                disabled={selectedIndex >= visible.length - 1}
                onClick={() => selectPairAt(selectedIndex + 1)}
                className="rounded border border-gray-600 p-1.5 text-gray-300 hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <LuChevronRight className="h-4 w-4" />
              </button>
            </div>

            <article className="rounded-lg border border-gray-700 bg-gray-800/60 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-400">
                <span className="rounded bg-gray-700 px-2 py-0.5 text-gray-200">Step {focusedPair.step}</span>
                <span>
                  {focusedPair.width}x{focusedPair.height}
                </span>
                <span>{statusLabel[focusedPair.status]}</span>
                {focusedPair.vote && (
                  <>
                    <span>Voted {choiceLabel[focusedPair.vote.choice]}</span>
                    <span>
                      seeds {focusedPair.seeds[0]} / {focusedPair.seeds[1]}
                    </span>
                  </>
                )}
              </div>
              <p className="mb-3 whitespace-pre-wrap break-words text-sm text-gray-100" title={focusedPair.prompt}>
                {focusedPair.prompt}
              </p>

              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {(['a', 'b'] as const).map(arm => {
                  const armLabel = arm.toUpperCase();
                  return (
                    <figure key={arm} className="min-w-0">
                      <figcaption className="mb-2 flex items-center justify-between gap-2">
                        <span className="rounded bg-gray-900 px-2 py-0.5 text-xs font-semibold text-gray-200">
                          Image {armLabel}
                        </span>
                        <span className="text-[11px] text-gray-500">
                          {focusedPair.width}x{focusedPair.height}
                        </span>
                      </figcaption>
                      <button
                        type="button"
                        title={`Compare image ${armLabel} full size`}
                        aria-label={`Compare image ${armLabel} full size for preference ${focusedPair.id}`}
                        onClick={() => setComparison({ pair: focusedPair, arm })}
                        className="group relative block w-full rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
                      >
                        <AuthenticatedPreferenceImage
                          url={focusedPair.image_urls[arm]}
                          alt={`Preference ${focusedPair.id}, image ${armLabel}`}
                          width={focusedPair.width}
                          height={focusedPair.height}
                        />
                        <span className="pointer-events-none absolute bottom-2 right-2 flex items-center gap-1 rounded bg-black/75 px-2 py-1 text-[11px] text-white opacity-90 group-hover:opacity-100">
                          <LuMaximize2 className="h-3 w-3" /> Full size
                        </span>
                      </button>
                      {focusedPair.status === 'pending' && (
                        <button
                          type="button"
                          title={`Choose ${armLabel}`}
                          aria-label={`Choose ${armLabel}`}
                          disabled={votingID !== null}
                          onClick={() => vote(focusedPair, arm)}
                          className="mt-2 w-full rounded bg-sky-700 px-3 py-2 text-sm font-medium text-white hover:bg-sky-600 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          {votingID === focusedPair.id ? 'Saving...' : `Choose ${armLabel}`}
                        </button>
                      )}
                    </figure>
                  );
                })}
              </div>

              {focusedPair.status === 'pending' && (
                <button
                  type="button"
                  title="Skip this pair"
                  aria-label="Skip this pair"
                  disabled={votingID !== null}
                  onClick={() => vote(focusedPair, 'skip')}
                  className="mt-3 w-full rounded border border-gray-600 px-3 py-2 text-xs text-gray-300 hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Skip this pair
                </button>
              )}
              {focusedPair.status === 'invalid' && (
                <p className="mt-2 text-xs text-amber-300">
                  An existing vote record is invalid; it will not be overwritten.
                </p>
              )}
            </article>
          </div>
        )}
      </div>

      <Dialog open={comparison !== null} onClose={() => setComparison(null)} className="relative z-50">
        <DialogBackdrop className="fixed inset-0 bg-black/80" />
        <div className="fixed inset-0 flex items-center justify-center overflow-y-auto p-2 sm:p-6">
          <DialogPanel className="w-full max-w-[96rem] rounded-lg border border-gray-700 bg-gray-900 p-3 text-left shadow-2xl sm:p-4">
            <div className="mb-3 flex items-start gap-3">
              <div className="min-w-0 flex-1">
                <DialogTitle className="text-sm font-semibold text-gray-100">
                  Compare image A and image B at full size
                </DialogTitle>
                <p className="mt-1 whitespace-pre-wrap break-words text-xs text-gray-400">{comparePair?.prompt}</p>
              </div>
              <button
                type="button"
                title="Close full-size comparison"
                aria-label="Close full-size comparison"
                onClick={() => setComparison(null)}
                className="rounded border border-gray-600 p-1.5 text-gray-300 hover:bg-gray-700"
              >
                <LuX className="h-4 w-4" />
              </button>
            </div>
            {comparePair && (
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {(['a', 'b'] as const).map(arm => {
                  const armLabel = arm.toUpperCase();
                  return (
                    <figure key={arm} className="min-w-0">
                      <figcaption className="mb-2 flex items-center gap-2 text-xs text-gray-300">
                        <span className="rounded bg-gray-700 px-2 py-0.5 font-semibold">Image {armLabel}</span>
                        <span className="text-gray-500">
                          {comparePair.width}x{comparePair.height}
                        </span>
                      </figcaption>
                      <AuthenticatedPreferenceImage
                        url={comparePair.image_urls[arm]}
                        alt={`Preference ${comparePair.id}, image ${armLabel}`}
                        width={comparePair.width}
                        height={comparePair.height}
                        size="dialog"
                      />
                    </figure>
                  );
                })}
              </div>
            )}
            <div className="mt-3 flex justify-end">
              <button
                type="button"
                title="Close full-size comparison"
                aria-label="Close full-size comparison"
                onClick={() => setComparison(null)}
                className="rounded bg-gray-700 px-3 py-1.5 text-xs text-white hover:bg-gray-600"
              >
                Close comparison
              </button>
            </div>
          </DialogPanel>
        </div>
      </Dialog>
    </section>
  );
}
