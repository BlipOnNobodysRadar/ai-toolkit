'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Job } from '@prisma/client';
import { Activity, BarChart3, Download, FileText, Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '@/utils/api';

type MetricMap = { [key: string]: number };
type ReportFileMeta = {
  fileName: string;
  exists: boolean;
  size: number | null;
  updatedAt: string | null;
  tooLarge: boolean;
  previewTruncated: boolean;
  readError: string | null;
};
type ValidationHistoryEntry = { step: number | null; phase: string; sampleCount: number; metrics: MetricMap };
type ValidationReport = ReportFileMeta & {
  entries: number;
  malformedLines: number;
  kind: 'native' | 'dpo' | 'mixed' | 'unknown';
  history: ValidationHistoryEntry[];
  latest: ValidationHistoryEntry | null;
  bySigma: { sigma: number; sampleCount: number; metrics: MetricMap }[];
};
type EdmPoint = { timestep: number; sigma: number; logVariance: number; predictionWeight: number };
type Edm2Report = ReportFileMeta & {
  pointCount: number;
  points: EdmPoint[];
  minWeight: number | null;
  maxWeight: number | null;
  minSigma: number | null;
  maxSigma: number | null;
};
type ReportsResponse = { validation: ValidationReport; edm2: Edm2Report };

const formatBytes = (size: number | null) => {
  if (size === null) return '';
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
};

const formatDate = (value: string | null) => {
  if (!value) return '';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const formatMetricLabel = (key: string) => {
  const labels: { [key: string]: string } = {
    video_mse: 'Video / image MSE',
    audio_mse: 'Audio MSE',
    dpo_loss: 'DPO loss',
    margin: 'Scaled margin',
    accuracy: 'Accuracy',
    strict_preference_correct: 'Preference accuracy',
    policy_chosen_error: 'Policy chosen error',
    policy_rejected_error: 'Policy rejected error',
    reference_chosen_error: 'Reference chosen error',
    reference_rejected_error: 'Reference rejected error',
    chosen_gain: 'Chosen gain',
    rejected_gain: 'Rejected gain',
  };
  return labels[key] || key.replaceAll('_', ' ');
};

const formatMetricValue = (key: string, value: number) => {
  if (key === 'accuracy' || key === 'strict_preference_correct') return `${(value * 100).toFixed(1)}%`;
  if (Math.abs(value) !== 0 && Math.abs(value) < 0.001) return value.toExponential(3);
  return value.toFixed(5);
};

const primaryMetricKeys = (kind: ValidationReport['kind']) =>
  kind === 'dpo'
    ? ['dpo_loss', 'margin', 'strict_preference_correct']
    : ['video_mse', 'audio_mse'];

const reportDownloadUrl = (jobID: string, fileName: string) =>
  `/api/jobs/${encodeURIComponent(jobID)}/reports/${encodeURIComponent(fileName)}`;

function ReportHeader({
  title,
  subtitle,
  report,
  icon,
  downloading,
  onDownload,
}: {
  title: string;
  subtitle: string;
  report: ReportFileMeta;
  icon: React.ReactNode;
  downloading: boolean;
  onDownload: (fileName: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-gray-800 pb-3">
      <div className="min-w-0">
        <div className="flex items-center gap-2 text-gray-100 font-semibold">
          {icon}
          {title}
        </div>
        <div className="mt-1 text-xs text-gray-400">{subtitle}</div>
      </div>
      <div className="flex items-center gap-3 text-xs text-gray-500">
        {report.updatedAt && <span>{formatDate(report.updatedAt)}</span>}
        {report.size !== null && <span>{formatBytes(report.size)}</span>}
        {report.exists && (
          <button
            type="button"
            onClick={() => onDownload(report.fileName)}
            disabled={downloading}
            className="inline-flex items-center gap-1 rounded bg-gray-800 px-2 py-1 text-gray-200 hover:bg-gray-700 disabled:cursor-wait disabled:opacity-60"
          >
            {downloading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
            {report.fileName}
          </button>
        )}
      </div>
    </div>
  );
}

function NoReport({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-4 rounded-lg border border-dashed border-gray-700 bg-gray-950/40 px-4 py-6 text-sm text-gray-400">
      {children}
    </div>
  );
}

function WeightChart({ points }: { points: EdmPoint[] }) {
  const chart = useMemo(() => {
    if (points.length === 0) return null;
    const width = 720;
    const height = 220;
    const pad = 34;
    const sigmas = points.map(point => point.sigma);
    const weights = points.map(point => point.predictionWeight);
    let minX = Math.min(...sigmas);
    let maxX = Math.max(...sigmas);
    let minY = Math.min(...weights);
    let maxY = Math.max(...weights);
    if (minX === maxX) {
      minX -= 0.5;
      maxX += 0.5;
    }
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    const x = (value: number) => pad + ((value - minX) / (maxX - minX)) * (width - pad * 2);
    const y = (value: number) => height - pad - ((value - minY) / (maxY - minY)) * (height - pad * 2);
    const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'}${x(point.sigma).toFixed(2)},${y(point.predictionWeight).toFixed(2)}`).join(' ');
    return { width, height, pad, path, minX, maxX, minY, maxY };
  }, [points]);

  if (!chart) return <NoReport>The EDM2 curve exists but has no numeric points.</NoReport>;
  return (
    <div className="mt-4 rounded-lg bg-gray-950/60 p-3">
      <svg viewBox={`0 0 ${chart.width} ${chart.height}`} className="h-56 w-full" role="img" aria-label="EDM2 prediction weight curve">
        <rect x="0" y="0" width={chart.width} height={chart.height} fill="transparent" />
        <line
          x1={chart.pad}
          y1={chart.height - chart.pad}
          x2={chart.width - chart.pad}
          y2={chart.height - chart.pad}
          stroke="currentColor"
          className="text-gray-700"
        />
        <line x1={chart.pad} y1={chart.pad} x2={chart.pad} y2={chart.height - chart.pad} stroke="currentColor" className="text-gray-700" />
        <path d={chart.path} fill="none" stroke="currentColor" strokeWidth="2" className="text-blue-400" />
        <text x={chart.pad} y={chart.height - 8} fill="currentColor" fontSize="10" className="text-gray-500">
          sigma {chart.minX.toFixed(3)}
        </text>
        <text x={chart.width - chart.pad} y={chart.height - 8} textAnchor="end" fill="currentColor" fontSize="10" className="text-gray-500">
          {chart.maxX.toFixed(3)}
        </text>
        <text x={4} y={chart.pad} fill="currentColor" fontSize="10" className="text-gray-500">
          {chart.maxY.toFixed(3)}
        </text>
        <text x={4} y={chart.height - chart.pad} fill="currentColor" fontSize="10" className="text-gray-500">
          {chart.minY.toFixed(3)}
        </text>
      </svg>
    </div>
  );
}

export default function JobReports({ job }: { job: Job }) {
  const [reports, setReports] = useState<ReportsResponse | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [error, setError] = useState('');
  const [downloadError, setDownloadError] = useState('');
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null);

  const load = useCallback(async () => {
    setStatus(previous => (previous === 'ready' ? 'loading' : previous));
    try {
      const response = await apiClient.get(`/api/jobs/${job.id}/reports`);
      setReports(response.data);
      setError('');
      setStatus('ready');
    } catch (requestError: any) {
      setError(requestError?.response?.data?.error || requestError?.message || 'Could not load reports');
      setStatus('error');
    }
  }, [job.id]);

  const downloadReport = useCallback(
    async (fileName: string) => {
      setDownloadingFile(fileName);
      setDownloadError('');
      try {
        const response = await apiClient.get(reportDownloadUrl(job.id, fileName), { responseType: 'blob' });
        const blob = response.data instanceof Blob ? response.data : new Blob([response.data]);
        const href = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = href;
        link.download = fileName;
        link.rel = 'noreferrer';
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(href), 0);
      } catch (requestError: any) {
        setDownloadError(requestError?.response?.data?.error || requestError?.message || `Could not download ${fileName}`);
      } finally {
        setDownloadingFile(null);
      }
    },
    [job.id],
  );

  useEffect(() => {
    setReports(null);
    void load();
  }, [load]);

  if (status === 'loading' && reports === null) {
    return (
      <div className="flex items-center justify-center gap-2 rounded-xl border border-gray-800 bg-gray-900 py-16 text-gray-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading run reports...
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="rounded-xl border border-rose-900 bg-rose-950/30 px-4 py-6 text-sm text-rose-200">
        <p>{error}</p>
        <button
          type="button"
          onClick={() => void load()}
          className="mt-3 inline-flex items-center gap-1 rounded bg-rose-800 px-2 py-1 text-xs text-white hover:bg-rose-700"
        >
          <RefreshCw className="h-3.5 w-3.5" /> Retry
        </button>
      </div>
    );
  }

  if (reports === null) return null;

  const { validation, edm2 } = reports;
  const validationMissing = !validation.exists;
  const edm2Missing = !edm2.exists;
  const latest = validation.latest;
  const DPO = validation.kind === 'dpo' || validation.kind === 'mixed';

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-gray-100">Run reports</h2>
          <p className="mt-1 text-xs text-gray-400">
            Read-only summaries of the fork&apos;s held-out validation and EDM2 weighting artifacts. Raw files stay in
            the job folder and remain downloadable.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          className="inline-flex items-center gap-1 rounded bg-gray-800 px-2 py-1 text-xs text-gray-200 hover:bg-gray-700"
        >
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </button>
      </div>

      {downloadError && (
        <div className="rounded-lg border border-rose-900 bg-rose-950/30 px-3 py-2 text-xs text-rose-200">{downloadError}</div>
      )}

      {validationMissing && edm2Missing && (
        <NoReport>
          No fork report files were found for this job. Held-out validation writes <code>validation.jsonl</code> when
          <code> train.validation_config</code> is configured. EDM2 writes <code>edm2-weighting.csv</code> when EDM2 is
          enabled and a checkpoint is saved.
        </NoReport>
      )}

      <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
        <ReportHeader
          title="Held-out validation"
          subtitle="Native video/image/audio MSE and DPO preference metrics appended by the trainer."
          report={validation}
          icon={<Activity className="h-4 w-4 text-emerald-400" />}
          downloading={downloadingFile === validation.fileName}
          onDownload={fileName => void downloadReport(fileName)}
        />
        {validationMissing ? (
          <NoReport>
            No <code>validation.jsonl</code> yet. Configure native held-out datasets under
            <code> train.validation_config</code> to record reproducible validation metrics.
          </NoReport>
        ) : validation.readError ? (
          <NoReport>The validation report could not be read ({validation.readError}).</NoReport>
        ) : latest === null ? (
          <NoReport>
            {validation.tooLarge
              ? 'The validation report is too large to preview and its recent tail has no valid metric records. Download the raw file to inspect it.'
              : 'The validation report has no valid metric records yet.'}
          </NoReport>
        ) : (
          <>
            {validation.previewTruncated && (
              <div className="mt-3 rounded-lg border border-amber-900 bg-amber-950/30 px-3 py-2 text-xs text-amber-200">
                This report is larger than the 8 MB preview limit. Showing a recent tail of up to 2 MB; download the raw
                file for the complete history.
              </div>
            )}
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-gray-400">
              <span className="rounded-full bg-gray-800 px-2 py-0.5">
                {validation.entries} validation record{validation.entries === 1 ? '' : 's'}
                {validation.previewTruncated ? ' in recent tail' : ''}
              </span>
              <span className="rounded-full bg-gray-800 px-2 py-0.5">
                latest step {latest.step ?? '?'} / {latest.phase || 'unknown phase'}
              </span>
              <span className="rounded-full bg-gray-800 px-2 py-0.5">{latest.sampleCount} raw metric rows</span>
              {validation.malformedLines > 0 && (
                <span className="rounded-full bg-amber-950 px-2 py-0.5 text-amber-300">
                  {validation.malformedLines} malformed line{validation.malformedLines === 1 ? '' : 's'} skipped
                </span>
              )}
            </div>

            <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {primaryMetricKeys(validation.kind)
                .filter(key => latest.metrics[key] !== undefined)
                .map(key => (
                  <div key={key} className="rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2">
                    <div className="text-[11px] uppercase tracking-wide text-gray-500">{formatMetricLabel(key)}</div>
                    <div className="mt-1 font-mono text-lg text-gray-100">{formatMetricValue(key, latest.metrics[key])}</div>
                  </div>
                ))}
            </div>

            <div className="mt-4 overflow-x-auto rounded-lg border border-gray-800">
              <table className="min-w-full text-left text-xs">
                <thead className="bg-gray-800/70 text-gray-300">
                  <tr>
                    <th className="px-3 py-2 font-medium">Step</th>
                    <th className="px-3 py-2 font-medium">Phase</th>
                    <th className="px-3 py-2 font-medium">Rows</th>
                    {primaryMetricKeys(validation.kind)
                      .filter(key => validation.history.some(entry => entry.metrics[key] !== undefined))
                      .map(key => (
                        <th key={key} className="px-3 py-2 font-medium">
                          {formatMetricLabel(key)}
                        </th>
                      ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800 text-gray-300">
                  {validation.history
                    .slice(-20)
                    .reverse()
                    .map((entry, index) => (
                      <tr key={`${entry.step ?? 'unknown'}-${entry.phase}-${index}`}>
                        <td className="px-3 py-2 font-mono">{entry.step ?? '?'}</td>
                        <td className="px-3 py-2">{entry.phase || '-'}</td>
                        <td className="px-3 py-2 font-mono">{entry.sampleCount}</td>
                        {primaryMetricKeys(validation.kind)
                          .filter(key => validation.history.some(item => item.metrics[key] !== undefined))
                          .map(key => (
                            <td key={key} className="px-3 py-2 font-mono">
                              {entry.metrics[key] === undefined ? '-' : formatMetricValue(key, entry.metrics[key])}
                            </td>
                          ))}
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
            {validation.bySigma.length > 0 && (
              <details className="mt-3 text-xs text-gray-400">
                <summary className="cursor-pointer text-gray-300">Metrics by validation sigma</summary>
                <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {validation.bySigma.map(row => (
                    <div key={row.sigma} className="rounded border border-gray-800 bg-gray-950/40 px-3 py-2">
                      <div className="text-gray-300">
                        sigma {row.sigma} <span className="text-gray-500">({row.sampleCount} rows)</span>
                      </div>
                      {primaryMetricKeys(validation.kind)
                        .filter(key => row.metrics[key] !== undefined)
                        .map(key => (
                          <div key={key} className="mt-1 flex justify-between gap-2 font-mono">
                            <span className="text-gray-500">{formatMetricLabel(key)}</span>
                            <span>{formatMetricValue(key, row.metrics[key])}</span>
                          </div>
                        ))}
                    </div>
                  ))}
                </div>
              </details>
            )}
            {DPO && (
              <p className="mt-3 text-xs text-gray-500">
                Accuracy is the fraction of held-out pairs whose chosen image gained more denoising error reduction than
                its rejected mate. Inspect individual pairs in the downloadable JSONL.
              </p>
            )}
          </>
        )}
      </section>

      <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
        <ReportHeader
          title="EDM2 weighting curve"
          subtitle="Per-timestep prediction weights written alongside the latest checkpoint."
          report={edm2}
          icon={<BarChart3 className="h-4 w-4 text-sky-400" />}
          downloading={downloadingFile === edm2.fileName}
          onDownload={fileName => void downloadReport(fileName)}
        />
        {edm2Missing ? (
          <NoReport>
            No <code>edm2-weighting.csv</code> yet. The curve is written when EDM2 is enabled and the trainer saves a
            checkpoint.
          </NoReport>
        ) : edm2.tooLarge ? (
          <NoReport>The EDM2 curve is larger than the UI&apos;s 8 MB preview limit. Download the raw file to inspect it.</NoReport>
        ) : edm2.readError ? (
          <NoReport>The EDM2 curve could not be read ({edm2.readError}).</NoReport>
        ) : edm2.pointCount === 0 ? (
          <NoReport>The EDM2 curve has no valid numeric rows.</NoReport>
        ) : (
          <>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-gray-400">
              <span className="rounded-full bg-gray-800 px-2 py-0.5">{edm2.pointCount} curve points</span>
              {edm2.minWeight !== null && edm2.maxWeight !== null && (
                <span className="rounded-full bg-gray-800 px-2 py-0.5">
                  weight {edm2.minWeight.toFixed(4)} .. {edm2.maxWeight.toFixed(4)}
                </span>
              )}
              {edm2.minSigma !== null && edm2.maxSigma !== null && (
                <span className="rounded-full bg-gray-800 px-2 py-0.5">
                  sigma {edm2.minSigma.toFixed(3)} .. {edm2.maxSigma.toFixed(3)}
                </span>
              )}
              <span className="inline-flex items-center gap-1 text-gray-500">
                <FileText className="h-3.5 w-3.5" /> showing {edm2.points.length} of {edm2.pointCount} points
              </span>
            </div>
            <WeightChart points={edm2.points} />
            <p className="mt-2 text-xs text-gray-500">
              The chart is a direct read-only rendering of the saved weights. It is not a quality claim; use held-out
              validation metrics to judge the run.
            </p>
          </>
        )}
      </section>
    </div>
  );
}
