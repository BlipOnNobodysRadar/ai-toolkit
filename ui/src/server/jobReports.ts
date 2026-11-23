import fs from 'fs';
import path from 'path';
import { PreferenceFeedbackError, resolveBoundJobRoot } from './preferenceFeedback';

// Read-only summaries of the reports fork training writes next to its checkpoints.
// Both formats are documented in training_extras/README.md. Only these fixed
// names are read; callers never supply a path.
export const VALIDATION_FILE = 'validation.jsonl';
export const EDM2_FILE = 'edm2-weighting.csv';
export const REPORT_FILE_NAMES = [VALIDATION_FILE, EDM2_FILE] as const;
export type ReportFileName = (typeof REPORT_FILE_NAMES)[number];
const MAX_REPORT_BYTES = 8 * 1024 * 1024;
const MAX_VALIDATION_TAIL_BYTES = 2 * 1024 * 1024;
const MAX_VALIDATION_LINES = 5000;
const MAX_VALIDATION_HISTORY = 500;
const MAX_EDM_POINTS = 360;

const REPORT_CONTENT_TYPES: Record<ReportFileName, string> = {
  [VALIDATION_FILE]: 'application/x-ndjson; charset=utf-8',
  [EDM2_FILE]: 'text/csv; charset=utf-8',
};

export const isReportFileName = (value: unknown): value is ReportFileName =>
  typeof value === 'string' && (REPORT_FILE_NAMES as readonly string[]).includes(value);

export type ReportFileMeta = {
  fileName: ReportFileName;
  exists: boolean;
  size: number | null;
  updatedAt: string | null;
  tooLarge: boolean;
  previewTruncated: boolean;
  readError: string | null;
};

export type ReportMetrics = { [key: string]: number };

export type ValidationHistoryEntry = {
  step: number | null;
  phase: string;
  sampleCount: number;
  metrics: ReportMetrics;
};

export type ValidationSigmaSummary = {
  sigma: number;
  sampleCount: number;
  metrics: ReportMetrics;
};

export type ValidationReport = ReportFileMeta & {
  entries: number;
  malformedLines: number;
  kind: 'native' | 'dpo' | 'mixed' | 'unknown';
  history: ValidationHistoryEntry[];
  latest: ValidationHistoryEntry | null;
  bySigma: ValidationSigmaSummary[];
};

export type EdmPoint = {
  timestep: number;
  sigma: number;
  logVariance: number;
  predictionWeight: number;
};

export type Edm2Report = ReportFileMeta & {
  pointCount: number;
  points: EdmPoint[];
  minWeight: number | null;
  maxWeight: number | null;
  minSigma: number | null;
  maxSigma: number | null;
};

export type JobReports = {
  validation: ValidationReport;
  edm2: Edm2Report;
};

export type ReportJob = {
  id: string;
  name: string;
};

const METRIC_KEYS = [
  'video_mse',
  'audio_mse',
  'dpo_loss',
  'margin',
  'accuracy',
  'strict_preference_correct',
  'policy_chosen_error',
  'policy_rejected_error',
  'reference_chosen_error',
  'reference_rejected_error',
  'chosen_gain',
  'rejected_gain',
] as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const finiteNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const metricNumber = (value: unknown): number | null => {
  if (typeof value === 'boolean') return value ? 1 : 0;
  return finiteNumber(value);
};

const meanMetrics = (rows: Record<string, unknown>[]): ReportMetrics => {
  const sums: { [key: string]: { sum: number; count: number } } = {};
  for (const row of rows) {
    for (const key of METRIC_KEYS) {
      const value = metricNumber(row[key]);
      if (value === null) continue;
      const bucket = (sums[key] ||= { sum: 0, count: 0 });
      bucket.sum += value;
      bucket.count += 1;
    }
  }
  const metrics: ReportMetrics = {};
  for (const key of METRIC_KEYS) {
    const bucket = sums[key];
    if (bucket && bucket.count > 0) metrics[key] = bucket.sum / bucket.count;
  }
  return metrics;
};

const hasAnyMetric = (row: Record<string, unknown>, keys: string[]) => keys.some(key => key in row);

/** Parse validation.jsonl without trusting its content. Unknown/malformed lines are counted, never thrown. */
export const summarizeValidationJsonl = (text: string): Omit<ValidationReport, keyof ReportFileMeta> => {
  const history: ValidationHistoryEntry[] = [];
  const bySigmaMap = new Map<number, Record<string, unknown>[]>();
  let malformedLines = 0;
  let nativeSeen = false;
  let dpoSeen = false;

  const lines = text.replace(/^\uFEFF/, '').trimEnd().split(/\r?\n/);
  for (const rawLine of lines.slice(-MAX_VALIDATION_LINES)) {
    const line = rawLine.trim();
    if (!line) continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      malformedLines += 1;
      continue;
    }
    if (!isRecord(parsed) || !Array.isArray(parsed.raw_metrics)) {
      malformedLines += 1;
      continue;
    }

    const rows = parsed.raw_metrics.filter(isRecord);
    if (rows.length !== parsed.raw_metrics.length) malformedLines += 1;
    if (rows.length === 0) continue;

    const isDpo = rows.some(row => hasAnyMetric(row, ['dpo_loss', 'margin', 'strict_preference_correct']));
    if (isDpo) dpoSeen = true;
    else nativeSeen = true;

    const step = finiteNumber(parsed.step);
    history.push({
      step: step === null ? null : Math.trunc(step),
      phase: typeof parsed.phase === 'string' ? parsed.phase : '',
      sampleCount: rows.length,
      metrics: meanMetrics(rows),
    });

    for (const row of rows) {
      const sigma = finiteNumber(row.sigma);
      if (sigma === null) continue;
      const sigmaRows = bySigmaMap.get(sigma) || [];
      sigmaRows.push(row);
      bySigmaMap.set(sigma, sigmaRows);
    }
  }

  const bySigma: ValidationSigmaSummary[] = [...bySigmaMap.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([sigma, rows]) => ({ sigma, sampleCount: rows.length, metrics: meanMetrics(rows) }));
  const retained = history.slice(-MAX_VALIDATION_HISTORY);
  const kind = nativeSeen && dpoSeen ? 'mixed' : dpoSeen ? 'dpo' : nativeSeen ? 'native' : 'unknown';

  return {
    entries: history.length,
    malformedLines,
    kind,
    history: retained,
    latest: retained.length ? retained[retained.length - 1] : null,
    bySigma,
  };
};
const parseCsvLine = (line: string): string[] => {
  const cells: string[] = [];
  let cell = '';
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === ',' && !quoted) {
      cells.push(cell);
      cell = '';
    } else {
      cell += char;
    }
  }
  cells.push(cell);
  return cells.map(value => value.trim());
};

const downsample = <T,>(values: T[], maxPoints: number): T[] => {
  if (values.length <= maxPoints) return values;
  const out: T[] = [];
  const step = (values.length - 1) / (maxPoints - 1);
  for (let index = 0; index < maxPoints; index += 1) {
    out.push(values[Math.round(index * step)]);
  }
  return out;
};

/** Parse the generated EDM2 weight curve. The generated CSV is simple, but quoted cells are accepted too. */
export const summarizeEdm2Csv = (text: string): Omit<Edm2Report, keyof ReportFileMeta> => {
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter(line => line.trim() !== '');
  const empty = {
    pointCount: 0,
    points: [] as EdmPoint[],
    minWeight: null,
    maxWeight: null,
    minSigma: null,
    maxSigma: null,
  };
  if (lines.length < 2) return empty;

  const header = parseCsvLine(lines[0]).map(value => value.toLowerCase());
  const timestepIndex = header.indexOf('timestep');
  const sigmaIndex = header.indexOf('sigma');
  const logVarianceIndex = header.indexOf('log_variance');
  const weightIndex = header.indexOf('prediction_weight');
  if (sigmaIndex < 0 || weightIndex < 0) return empty;

  const allPoints: EdmPoint[] = [];
  for (const line of lines.slice(1)) {
    const cells = parseCsvLine(line);
    const sigma = Number(cells[sigmaIndex]);
    const predictionWeight = Number(cells[weightIndex]);
    if (!Number.isFinite(sigma) || !Number.isFinite(predictionWeight)) continue;
    const timestep = timestepIndex >= 0 ? Number(cells[timestepIndex]) : allPoints.length;
    const logVariance = logVarianceIndex >= 0 ? Number(cells[logVarianceIndex]) : Number.NaN;
    if (!Number.isFinite(timestep)) continue;
    allPoints.push({
      timestep,
      sigma,
      logVariance: Number.isFinite(logVariance) ? logVariance : 0,
      predictionWeight,
    });
  }
  if (allPoints.length === 0) return empty;

  const weights = allPoints.map(point => point.predictionWeight);
  const sigmas = allPoints.map(point => point.sigma);
  return {
    pointCount: allPoints.length,
    points: downsample(allPoints, MAX_EDM_POINTS),
    minWeight: Math.min(...weights),
    maxWeight: Math.max(...weights),
    minSigma: Math.min(...sigmas),
    maxSigma: Math.max(...sigmas),
  };
};

type BoundReportRoot = { lexical: string; real: string };
type VerifiedReportFile = { lexical: string; real: string; size: number; updatedAt: string };

const samePath = (left: string, right: string): boolean => {
  const a = path.resolve(left);
  const b = path.resolve(right);
  return process.platform === 'win32' ? a.toLowerCase() === b.toLowerCase() : a === b;
};

const lstatOptional = async (file: string): Promise<fs.Stats | null> => {
  try {
    return await fs.promises.lstat(file);
  } catch (error: any) {
    if (error?.code === 'ENOENT') return null;
    throw new PreferenceFeedbackError('The report file could not be inspected safely.', 500, 'REPORT_UNREADABLE');
  }
};

// The job root was already accepted by resolveBoundJobRoot. This check is deliberately repeated for the report
// itself: a fixed name must be a real regular file whose physical location is the verified job folder, not a
// symlink/junction to another file.
const verifyReportFile = async (bound: BoundReportRoot, fileName: ReportFileName): Promise<VerifiedReportFile | null> => {
  const lexical = path.join(bound.lexical, fileName);
  const stat = await lstatOptional(lexical);
  if (!stat) return null;
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new PreferenceFeedbackError('The requested report file is not a regular file.', 403, 'UNSAFE_REPORT_FILE');
  }
  let real: string;
  try {
    real = await fs.promises.realpath(lexical);
  } catch {
    throw new PreferenceFeedbackError('The requested report file could not be resolved safely.', 403, 'UNSAFE_REPORT_FILE');
  }
  if (!samePath(path.dirname(real), bound.real)) {
    throw new PreferenceFeedbackError('The requested report file resolves outside the job folder.', 403, 'UNSAFE_REPORT_FILE');
  }
  return { lexical, real, size: stat.size, updatedAt: stat.mtime.toISOString() };
};

const baseMeta = (fileName: ReportFileName): ReportFileMeta => ({
  fileName,
  exists: false,
  size: null,
  updatedAt: null,
  tooLarge: false,
  previewTruncated: false,
  readError: null,
});

const errorCode = (error: any): string => error?.code || error?.message || 'read failed';

const readReport = async (
  bound: BoundReportRoot,
  fileName: ReportFileName,
  readLargeValidationTail: boolean,
): Promise<{ meta: ReportFileMeta; text: string | null }> => {
  const meta = baseMeta(fileName);
  const verified = await verifyReportFile(bound, fileName);
  if (!verified) return { meta, text: null };

  meta.exists = true;
  meta.size = verified.size;
  meta.updatedAt = verified.updatedAt;
  meta.tooLarge = verified.size > MAX_REPORT_BYTES;
  if (meta.tooLarge && !readLargeValidationTail) return { meta, text: null };

  let handle: fs.promises.FileHandle | null = null;
  try {
    handle = await fs.promises.open(verified.lexical, 'r');
    const opened = await handle.stat();
    if (!opened.isFile()) {
      throw new PreferenceFeedbackError('The requested report file is not a regular file.', 403, 'UNSAFE_REPORT_FILE');
    }
    meta.size = opened.size;
    meta.updatedAt = opened.mtime.toISOString();
    meta.tooLarge = opened.size > MAX_REPORT_BYTES;

    if (meta.tooLarge && !readLargeValidationTail) return { meta, text: null };

    if (meta.tooLarge) {
      const start = Math.max(0, opened.size - MAX_VALIDATION_TAIL_BYTES);
      const length = opened.size - start;
      const buffer = Buffer.alloc(length);
      const { bytesRead } = await handle.read(buffer, 0, length, start);
      let text = buffer.subarray(0, bytesRead).toString('utf8');
      // The read starts in the middle of a line; discard that partial record rather than counting it as malformed.
      const firstNewline = text.indexOf('\n');
      if (start > 0 && firstNewline >= 0) text = text.slice(firstNewline + 1);
      else if (start > 0) text = '';
      meta.previewTruncated = true;
      return { meta, text };
    }

    const buffer = Buffer.alloc(opened.size);
    let offset = 0;
    while (offset < opened.size) {
      const { bytesRead } = await handle.read(buffer, offset, opened.size - offset, offset);
      if (bytesRead === 0) break;
      offset += bytesRead;
    }
    return { meta, text: buffer.subarray(0, offset).toString('utf8') };
  } catch (error: any) {
    if (error instanceof PreferenceFeedbackError) throw error;
    meta.readError = errorCode(error);
    return { meta, text: null };
  } finally {
    if (handle) await handle.close().catch(() => undefined);
  }
};

/** Read the two fixed report names from the verified job folder. */
export const readJobReports = async (job: ReportJob, trainingRoot: string): Promise<JobReports> => {
  const bound = await resolveBoundJobRoot(job, trainingRoot);
  const validationFile = await readReport(bound, VALIDATION_FILE, true);
  const edm2File = await readReport(bound, EDM2_FILE, false);

  const validation: ValidationReport = {
    ...validationFile.meta,
    entries: 0,
    malformedLines: 0,
    kind: 'unknown',
    history: [],
    latest: null,
    bySigma: [],
    ...(validationFile.text === null ? {} : summarizeValidationJsonl(validationFile.text)),
  };
  const edm2: Edm2Report = {
    ...edm2File.meta,
    pointCount: 0,
    points: [],
    minWeight: null,
    maxWeight: null,
    minSigma: null,
    maxSigma: null,
    ...(edm2File.text === null ? {} : summarizeEdm2Csv(edm2File.text)),
  };
  return { validation, edm2 };
};

export type JobReportDownload = {
  stream: fs.ReadStream;
  size: number;
  fileName: ReportFileName;
  contentType: string;
};

/** Open one fixed report name from the verified job folder and return a bounded-memory file stream. */
export const readJobReportDownload = async (
  job: ReportJob,
  trainingRoot: string,
  reportName: string,
): Promise<JobReportDownload> => {
  if (!isReportFileName(reportName)) {
    throw new PreferenceFeedbackError('Invalid report file name.', 400, 'INVALID_REPORT_FILE');
  }
  const bound = await resolveBoundJobRoot(job, trainingRoot);
  const verified = await verifyReportFile(bound, reportName);
  if (!verified) {
    throw new PreferenceFeedbackError('Report file not found.', 404, 'REPORT_NOT_FOUND');
  }

  let handle: fs.promises.FileHandle;
  try {
    handle = await fs.promises.open(verified.lexical, 'r');
  } catch {
    throw new PreferenceFeedbackError('The requested report file could not be opened.', 500, 'REPORT_UNREADABLE');
  }
  try {
    const opened = await handle.stat();
    if (!opened.isFile()) {
      throw new PreferenceFeedbackError('The requested report file is not a regular file.', 403, 'UNSAFE_REPORT_FILE');
    }
    return {
      // Pin the stream to the size observed on the already-open handle so append-only reports cannot exceed
      // the Content-Length sent to the browser.
      stream: opened.size > 0 ? handle.createReadStream({ start: 0, end: opened.size - 1 }) : handle.createReadStream(),
      size: opened.size,
      fileName: reportName,
      contentType: REPORT_CONTENT_TYPES[reportName],
    };
  } catch (error) {
    await handle.close().catch(() => undefined);
    throw error;
  }
};
