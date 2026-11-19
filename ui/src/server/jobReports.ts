import fs from 'fs';
import path from 'path';

// Read-only summaries of the reports fork training writes next to its checkpoints.
// Both formats are documented in training_extras/README.md. Only these fixed
// names are read; callers never supply a path.
export const VALIDATION_FILE = 'validation.jsonl';
export const EDM2_FILE = 'edm2-weighting.csv';
const MAX_REPORT_BYTES = 8 * 1024 * 1024;
const MAX_VALIDATION_LINES = 5000;
const MAX_VALIDATION_HISTORY = 500;
const MAX_EDM_POINTS = 360;

export type ReportFileMeta = {
  fileName: string;
  path: string;
  exists: boolean;
  size: number | null;
  updatedAt: string | null;
  tooLarge: boolean;
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
  let lineCount = 0;
  let nativeSeen = false;
  let dpoSeen = false;

  for (const rawLine of text.replace(/^\uFEFF/, '').split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (lineCount >= MAX_VALIDATION_LINES) break;
    lineCount += 1;

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

const readKnownFile = async (filePath: string): Promise<{ meta: ReportFileMeta; text: string | null }> => {
  const meta: ReportFileMeta = {
    fileName: path.basename(filePath),
    path: filePath,
    exists: false,
    size: null,
    updatedAt: null,
    tooLarge: false,
    readError: null,
  };

  let stats;
  try {
    stats = await fs.promises.stat(filePath);
  } catch {
    return { meta, text: null };
  }
  if (!stats.isFile()) return { meta, text: null };

  meta.exists = true;
  meta.size = stats.size;
  meta.updatedAt = stats.mtime.toISOString();
  if (stats.size > MAX_REPORT_BYTES) {
    meta.tooLarge = true;
    return { meta, text: null };
  }

  try {
    return { meta, text: await fs.promises.readFile(filePath, 'utf-8') };
  } catch (error: any) {
    meta.readError = error?.code || error?.message || 'read failed';
    return { meta, text: null };
  }
};

/** Resolve a job folder under the configured training root, rejecting traversal and absolute job names. */
export const resolveJobFolder = (trainingRoot: string, jobName: string): string => {
  const root = path.resolve(trainingRoot);
  const folder = path.resolve(root, jobName);
  const relative = path.relative(root, folder);
  if (relative === '' || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error('Job folder is outside the training root');
  }
  return folder;
};

/** Resolve an existing job folder physically too, so a junction/symlink cannot expose a report outside the root. */
export const resolveExistingJobFolder = async (trainingRoot: string, jobName: string): Promise<string> => {
  const lexicalFolder = resolveJobFolder(trainingRoot, jobName);
  const realRoot = await fs.promises.realpath(path.resolve(trainingRoot));
  let realFolder: string;
  try {
    realFolder = await fs.promises.realpath(lexicalFolder);
  } catch (error: any) {
    // A not-yet-created job folder has no reports. Other errors are real read failures.
    if (error?.code === 'ENOENT') return lexicalFolder;
    throw error;
  }
  const relative = path.relative(realRoot, realFolder);
  if (relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) {
    throw new Error('Job folder is outside the training root');
  }
  return realFolder;
};

export const readJobReports = async (jobFolder: string): Promise<JobReports> => {
  const validationFile = await readKnownFile(path.join(jobFolder, VALIDATION_FILE));
  const edm2File = await readKnownFile(path.join(jobFolder, EDM2_FILE));

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
