import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { once } from 'events';
import archiver from 'archiver';
import sqlite3 from 'sqlite3';
import { jobFolderError } from '../../cron/jobFolder';

// Job-scoped reader/writer for MOSS preference-feedback pairs.
//
// The browser supplies only a job id, a pair id and a fixed arm/choice. Every filesystem path is derived from the
// configured training folder and job name; pair metadata and vote files are treated as untrusted input. The image route is never a
// generic file server: the only filenames it can serve are a.png and b.png inside a validated pair directory.

export type PreferenceChoice = 'a' | 'b' | 'skip';
export type PreferenceStatus = 'pending' | 'rated' | 'skipped' | 'invalid';

export interface PreferenceVote {
  version: 1;
  id: string;
  choice: PreferenceChoice;
  voted_at: string;
}

export interface PreferencePair {
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

export interface PreferenceTrainingMetric {
  /** Trainer value recorded at this exact loss-log step; each metric has its own semantics and step. */
  value: number;
  /** The loss-log step that owns the row, never the job's current step. */
  step: number;
}

export interface PreferenceFeedbackTrainingLog {
  /** Completed preference updates reported by the latest feedback/updates row. */
  preference_updates: PreferenceTrainingMetric | null;
  /** Successful optimizer updates reported by the latest feedback/optimizer_updates row. */
  optimizer_updates: PreferenceTrainingMetric | null;
  /** Aggregate incompatible votes at the last recorded feedback/foreign_votes scan. */
  foreign_votes: PreferenceTrainingMetric | null;
}

export interface PreferenceFeedbackList {
  enabled: boolean;
  pairs: PreferencePair[];
  counts: { pending: number; rated: number; skipped: number; invalid: number };
  /** Present only when at least one real feedback metric row exists in the job's loss log. */
  training?: PreferenceFeedbackTrainingLog;
}

type BoundJob = {
  id: string;
  name: string;
  job_config?: string | null;
};

export class PreferenceFeedbackError extends Error {
  status: number;
  code: string;

  constructor(message: string, status = 400, code = 'PREFERENCE_FEEDBACK_ERROR') {
    super(message);
    this.name = 'PreferenceFeedbackError';
    this.status = status;
    this.code = code;
  }
}

const UUID_V4_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const ISO_UTC_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|\+00:00)$/;
const MAX_PAIR_JSON_BYTES = 1024 * 1024;
const MAX_VOTE_JSON_BYTES = 64 * 1024;
const MAX_PROMPT_CHARS = 200_000;

export const isUuidV4 = (value: unknown): value is string => typeof value === 'string' && UUID_V4_RE.test(value);

// Python publishes uuid.uuid4().hex for pairs. Prisma uses hyphenated UUIDs for jobs.
export const isPreferenceID = (value: unknown): value is string =>
  typeof value === 'string' && /^[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}$/.test(value);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

const isIsoUtc = (value: unknown): value is string =>
  typeof value === 'string' && ISO_UTC_RE.test(value) && Number.isFinite(Date.parse(value));

const isSafeInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value);

const isContained = (root: string, candidate: string): boolean => {
  const rel = path.relative(root, candidate);
  return rel === '' || (!path.isAbsolute(rel) && rel !== '..' && !rel.startsWith(`..${path.sep}`));
};

const lstatOptional = async (file: string): Promise<fs.Stats | null> => {
  try {
    return await fs.promises.lstat(file);
  } catch (e: any) {
    if (e?.code === 'ENOENT') return null;
    throw new PreferenceFeedbackError('The job folder could not be inspected safely.', 500, 'UNREADABLE_JOB_FOLDER');
  }
};

const assertRegularFile = async (file: string, root: string, code: string): Promise<fs.Stats> => {
  const stat = await lstatOptional(file);
  if (!stat) throw new PreferenceFeedbackError('The requested preference file does not exist.', 404, code);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new PreferenceFeedbackError('The requested preference file is not a regular file.', 403, code);
  }
  const real = await fs.promises.realpath(file).catch(() => {
    throw new PreferenceFeedbackError('The requested preference file could not be resolved safely.', 403, code);
  });
  if (!isContained(root, real)) {
    throw new PreferenceFeedbackError('The requested preference file resolves outside the job folder.', 403, code);
  }
  return stat;
};

const readJsonFile = async (file: string, root: string, maxBytes: number, code: string): Promise<unknown> => {
  const stat = await assertRegularFile(file, root, code);
  if (stat.size > maxBytes) {
    throw new PreferenceFeedbackError('The preference metadata is too large.', 400, code);
  }
  let text: string;
  try {
    text = await fs.promises.readFile(file, 'utf8');
  } catch {
    throw new PreferenceFeedbackError('The preference metadata could not be read.', 500, code);
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new PreferenceFeedbackError('The preference metadata is not valid JSON.', 400, code);
  }
};

const parsePairMetadata = (raw: unknown, expectedID: string): Omit<PreferencePair, 'status' | 'vote' | 'image_urls'> => {
  if (!isRecord(raw)) throw new PreferenceFeedbackError('Invalid pair metadata.', 400, 'INVALID_PAIR');
  if (raw.version !== 1) throw new PreferenceFeedbackError('Unsupported pair metadata version.', 400, 'INVALID_PAIR');
  if (raw.id !== expectedID || !isPreferenceID(raw.id)) {
    throw new PreferenceFeedbackError('Pair metadata id does not match its directory.', 400, 'INVALID_PAIR');
  }
  if (!isSafeInteger(raw.step) || raw.step < 0) {
    throw new PreferenceFeedbackError('Pair metadata has an invalid step.', 400, 'INVALID_PAIR');
  }
  if (typeof raw.prompt !== 'string' || raw.prompt.length > MAX_PROMPT_CHARS) {
    throw new PreferenceFeedbackError('Pair metadata has an invalid prompt.', 400, 'INVALID_PAIR');
  }
  if (
    !Array.isArray(raw.seeds) ||
    raw.seeds.length !== 2 ||
    !isSafeInteger(raw.seeds[0]) ||
    !isSafeInteger(raw.seeds[1])
  ) {
    throw new PreferenceFeedbackError('Pair metadata has invalid seeds.', 400, 'INVALID_PAIR');
  }
  if (!isSafeInteger(raw.width) || raw.width <= 0 || !isSafeInteger(raw.height) || raw.height <= 0) {
    throw new PreferenceFeedbackError('Pair metadata has invalid dimensions.', 400, 'INVALID_PAIR');
  }
  if (typeof raw.reference_id !== 'string' || !/^[a-f0-9]{64}$/.test(raw.reference_id)) {
    throw new PreferenceFeedbackError('Pair metadata has an invalid reference id.', 400, 'INVALID_PAIR');
  }
  if (!isIsoUtc(raw.created_at)) {
    throw new PreferenceFeedbackError('Pair metadata has an invalid creation time.', 400, 'INVALID_PAIR');
  }
  return {
    id: expectedID,
    step: raw.step,
    prompt: raw.prompt,
    seeds: [raw.seeds[0], raw.seeds[1]],
    width: raw.width,
    height: raw.height,
    reference_id: raw.reference_id,
    created_at: raw.created_at,
  };
};

const parseVote = (raw: unknown, expectedID: string): PreferenceVote | null => {
  if (!isRecord(raw)) return null;
  if (raw.version !== 1 || raw.id !== expectedID) return null;
  if (raw.choice !== 'a' && raw.choice !== 'b' && raw.choice !== 'skip') return null;
  if (!isIsoUtc(raw.voted_at)) return null;
  return { version: 1, id: expectedID, choice: raw.choice, voted_at: raw.voted_at };
};

const feedbackSettings = (rawJobConfig: string | null | undefined): { enabled: boolean; every: number | null; start_after: number | null } => {
  let parsed: unknown;
  try {
    parsed = typeof rawJobConfig === 'string' ? JSON.parse(rawJobConfig) : rawJobConfig;
  } catch {
    return { enabled: false, every: null, start_after: null };
  }
  if (!isRecord(parsed) || !isRecord(parsed.config)) return { enabled: false, every: null, start_after: null };
  const processes = parsed.config.process;
  if (!Array.isArray(processes) || !isRecord(processes[0]) || !isRecord(processes[0].train)) {
    return { enabled: false, every: null, start_after: null };
  }
  const feedback = processes[0].train.dpo_feedback;
  // Shared contract: absent/None means off; any object means the gallery is enabled. The numeric fields are only
  // display hints, so malformed optional values do not make an otherwise-enabled job disappear.
  if (!isRecord(feedback)) return { enabled: false, every: null, start_after: null };
  return {
    enabled: true,
    every: isSafeInteger(feedback.every) && feedback.every > 0 ? feedback.every : null,
    start_after: isSafeInteger(feedback.start_after) && feedback.start_after >= 0 ? feedback.start_after : null,
  };
};

export const isDpoFeedbackEnabled = (rawJobConfig: string | null | undefined): boolean => feedbackSettings(rawJobConfig).enabled;

type BoundRoot = { lexical: string; real: string; preferencesLexical: string; preferencesReal: string | null };

// The UI stores each job under its name in the configured training folder.
export const resolveBoundJobRoot = async (job: BoundJob, trainingRoot: string): Promise<BoundRoot> => {
  if (!job || !isUuidV4(job.id)) {
    throw new PreferenceFeedbackError('Invalid job id.', 400, 'INVALID_JOB_ID');
  }
  if (jobFolderError(trainingRoot, job.name)) {
    throw new PreferenceFeedbackError('Invalid job folder name.', 400, 'INVALID_JOB_FOLDER');
  }

  const configuredRoot = path.resolve(trainingRoot);
  const configuredStat = await fs.promises.stat(configuredRoot).catch(() => null);
  if (!configuredStat || !configuredStat.isDirectory()) {
    throw new PreferenceFeedbackError('The configured training folder is unavailable.', 500, 'TRAINING_FOLDER_UNAVAILABLE');
  }
  const realConfiguredRoot = await fs.promises.realpath(configuredRoot).catch(() => {
    throw new PreferenceFeedbackError('The configured training folder cannot be resolved safely.', 500, 'TRAINING_FOLDER_UNAVAILABLE');
  });

  const lexical = path.join(configuredRoot, job.name);
  const folderStat = await lstatOptional(lexical);
  if (!folderStat) {
    return { lexical, real: lexical, preferencesLexical: path.join(lexical, 'preferences'), preferencesReal: null };
  }
  if (folderStat.isSymbolicLink() || !folderStat.isDirectory()) {
    throw new PreferenceFeedbackError('The job folder is not a regular directory.', 403, 'UNSAFE_JOB_FOLDER');
  }
  const realFolder = await fs.promises.realpath(lexical).catch(() => {
    throw new PreferenceFeedbackError('The job folder cannot be resolved safely.', 403, 'UNSAFE_JOB_FOLDER');
  });
  if (!isContained(realConfiguredRoot, realFolder)) {
    throw new PreferenceFeedbackError('The job folder resolves outside the training folder.', 403, 'UNSAFE_JOB_FOLDER');
  }

  const preferencesLexical = path.join(lexical, 'preferences');
  const preferencesStat = await lstatOptional(preferencesLexical);
  if (!preferencesStat) return { lexical, real: realFolder, preferencesLexical, preferencesReal: null };
  if (preferencesStat.isSymbolicLink() || !preferencesStat.isDirectory()) {
    throw new PreferenceFeedbackError('The preferences folder is not a regular directory.', 403, 'UNSAFE_PREFERENCES_FOLDER');
  }
  const preferencesReal = await fs.promises.realpath(preferencesLexical).catch(() => {
    throw new PreferenceFeedbackError('The preferences folder cannot be resolved safely.', 403, 'UNSAFE_PREFERENCES_FOLDER');
  });
  if (!isContained(realFolder, preferencesReal)) {
    throw new PreferenceFeedbackError('The preferences folder resolves outside the job folder.', 403, 'UNSAFE_PREFERENCES_FOLDER');
  }
  return { lexical, real: realFolder, preferencesLexical, preferencesReal };
};

const resolvePairDirectory = async (bound: BoundRoot, preferenceID: string): Promise<{ lexical: string; real: string }> => {
  if (!isPreferenceID(preferenceID)) {
    throw new PreferenceFeedbackError('Invalid preference id.', 400, 'INVALID_PREFERENCE_ID');
  }
  if (!bound.preferencesReal) {
    throw new PreferenceFeedbackError('Preference pair not found.', 404, 'PAIR_NOT_FOUND');
  }
  const pairLexical = path.join(bound.preferencesLexical, preferenceID);
  const pairStat = await lstatOptional(pairLexical);
  if (!pairStat) throw new PreferenceFeedbackError('Preference pair not found.', 404, 'PAIR_NOT_FOUND');
  if (pairStat.isSymbolicLink() || !pairStat.isDirectory()) {
    throw new PreferenceFeedbackError('Preference pair is not a regular directory.', 403, 'UNSAFE_PAIR');
  }
  const pairReal = await fs.promises.realpath(pairLexical).catch(() => {
    throw new PreferenceFeedbackError('Preference pair cannot be resolved safely.', 403, 'UNSAFE_PAIR');
  });
  if (!isContained(bound.preferencesReal, pairReal)) {
    throw new PreferenceFeedbackError('Preference pair resolves outside the preferences folder.', 403, 'UNSAFE_PAIR');
  }
  return { lexical: pairLexical, real: pairReal };
};

type PairPath = { lexical: string; real: string };

const VOTE_FILENAME = 'vote.json';
const DECISION_DIRNAME = '.decision';

const serializeVote = (vote: PreferenceVote): Buffer => Buffer.from(JSON.stringify(vote, null, 2) + '\n', 'utf8');

// File data is fsync'd before publication. Directory fsync is best-effort because Windows does not expose directories
// as openable files, while POSIX uses it to make the rename itself durable.
const fsyncDirectory = async (dir: string): Promise<void> => {
  let handle: fs.promises.FileHandle | null = null;
  try {
    handle = await fs.promises.open(dir, 'r');
    await handle.sync();
  } catch {
    // Directory fsync is not portable; the complete vote file was already synced.
  } finally {
    if (handle) await handle.close().catch(() => {});
  }
};

const readTopLevelVote = async (
  pair: PairPath,
  preferenceID: string,
): Promise<{ present: boolean; vote: PreferenceVote | null }> => {
  const votePath = path.join(pair.lexical, VOTE_FILENAME);
  const stat = await lstatOptional(votePath);
  if (!stat) return { present: false, vote: null };
  try {
    return {
      present: true,
      vote: parseVote(await readJsonFile(votePath, pair.real, MAX_VOTE_JSON_BYTES, 'INVALID_VOTE'), preferenceID),
    };
  } catch {
    return { present: true, vote: null };
  }
};

// The fixed .decision directory is the immutable source of truth. It is only ever created by renaming a complete staged
// directory, and a non-empty destination directory cannot be replaced by a concurrent rename on POSIX or Windows.
const readDecisionVote = async (
  pair: PairPath,
  preferenceID: string,
): Promise<{ present: boolean; vote: PreferenceVote | null }> => {
  const decisionPath = path.join(pair.lexical, DECISION_DIRNAME);
  const stat = await lstatOptional(decisionPath);
  if (!stat) return { present: false, vote: null };
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new PreferenceFeedbackError('The committed vote record is not a regular directory.', 403, 'INVALID_VOTE');
  }
  const decisionReal = await fs.promises.realpath(decisionPath).catch(() => {
    throw new PreferenceFeedbackError('The committed vote record could not be resolved safely.', 403, 'INVALID_VOTE');
  });
  if (!isContained(pair.real, decisionReal)) {
    throw new PreferenceFeedbackError('The committed vote record resolves outside the preference pair.', 403, 'INVALID_VOTE');
  }

  const votePath = path.join(decisionPath, VOTE_FILENAME);
  let raw: unknown;
  try {
    raw = await readJsonFile(votePath, pair.real, MAX_VOTE_JSON_BYTES, 'INVALID_VOTE');
  } catch (error) {
    if (error instanceof PreferenceFeedbackError && error.status === 404) return { present: true, vote: null };
    throw error;
  }
  return { present: true, vote: parseVote(raw, preferenceID) };
};

const projectionConflict = (): PreferenceFeedbackError =>
  new PreferenceFeedbackError('This preference pair already has a different immutable vote.', 409, 'VOTE_CONFLICT');

const projectionInvalid = (): PreferenceFeedbackError =>
  new PreferenceFeedbackError('An existing vote record is invalid; refusing to overwrite it.', 409, 'VOTE_CONFLICT');

// Project the committed decision into the legacy-compatible top-level vote.json receipt. A complete temp file is
// fsync'd and atomically renamed. Any existing receipt is validated first and is never replaced by a conflicting choice.
const projectVote = async (pair: PairPath, preferenceID: string, vote: PreferenceVote): Promise<void> => {
  const votePath = path.join(pair.lexical, VOTE_FILENAME);

  const existingStat = await lstatOptional(votePath);
  if (existingStat) {
    let existing: PreferenceVote | null = null;
    try {
      existing = parseVote(await readJsonFile(votePath, pair.real, MAX_VOTE_JSON_BYTES, 'INVALID_VOTE'), preferenceID);
    } catch {
      existing = null;
    }
    if (!existing) throw projectionInvalid();
    if (existing.choice !== vote.choice) throw projectionConflict();
    return;
  }

  const temp = path.join(pair.lexical, `.vote.project.${process.pid}.${randomUUID()}.tmp`);
  let handle: fs.promises.FileHandle | null = null;
  try {
    handle = await fs.promises.open(temp, 'wx', 0o600);
    await handle.writeFile(serializeVote(vote));
    await handle.sync();
    await handle.close();
    handle = null;

    // Respect a legacy receipt that appeared while this projection was being staged.
    const appeared = await lstatOptional(votePath);
    if (appeared) {
      let existing: PreferenceVote | null = null;
      try {
        existing = parseVote(await readJsonFile(votePath, pair.real, MAX_VOTE_JSON_BYTES, 'INVALID_VOTE'), preferenceID);
      } catch {
        existing = null;
      }
      if (!existing) throw projectionInvalid();
      if (existing.choice !== vote.choice) throw projectionConflict();
      return;
    }

    await fs.promises.rename(temp, votePath);
    await fsyncDirectory(pair.lexical);
  } catch (error) {
    if (error instanceof PreferenceFeedbackError) throw error;
    throw new PreferenceFeedbackError('The vote receipt could not be projected safely.', 500, 'VOTE_WRITE_FAILED');
  } finally {
    if (handle) await handle.close().catch(() => {});
    await fs.promises.unlink(temp).catch(() => {});
  }
};

const readVoteForPair = async (
  pair: PairPath,
  preferenceID: string,
  options: { healProjection?: boolean } = {},
): Promise<{ vote: PreferenceVote | null; invalid: boolean }> => {
  const votePath = path.join(pair.lexical, VOTE_FILENAME);
  const voteStat = await lstatOptional(votePath);
  if (voteStat) {
    try {
      const vote = parseVote(
        await readJsonFile(votePath, pair.real, MAX_VOTE_JSON_BYTES, 'INVALID_VOTE'),
        preferenceID,
      );
      return { vote, invalid: vote === null };
    } catch {
      return { vote: null, invalid: true };
    }
  }

  let decision: { present: boolean; vote: PreferenceVote | null };
  try {
    decision = await readDecisionVote(pair, preferenceID);
  } catch {
    return { vote: null, invalid: true };
  }
  if (!decision.present) return { vote: null, invalid: false };
  if (!decision.vote) return { vote: null, invalid: true };

  // Exports are strictly read-only. They still honor a committed decision, but must not heal its top-level receipt.
  if (options.healProjection === false) return { vote: decision.vote, invalid: false };

  // A crash can leave the committed decision without its top-level receipt. Serve the committed choice even when the
  // receipt cannot currently be healed (for example on a temporarily read-only checkout).
  try {
    await projectVote(pair, preferenceID, decision.vote);
  } catch {
    // The decision remains authoritative; the trainer can ignore the pair until the receipt is restored.
  }
  return { vote: decision.vote, invalid: false };
};

const readPair = async (
  bound: BoundRoot,
  preferenceID: string,
  pair: PairPath,
  options: { healProjection?: boolean } = {},
) => {
  const metadata = parsePairMetadata(
    await readJsonFile(path.join(pair.lexical, 'pair.json'), pair.real, MAX_PAIR_JSON_BYTES, 'INVALID_PAIR'),
    preferenceID,
  );
  // The two displayed images are exact fixed names. Validate both before publishing the pair to the UI.
  await assertRegularFile(path.join(pair.lexical, 'a.png'), pair.real, 'INVALID_PAIR_IMAGE');
  await assertRegularFile(path.join(pair.lexical, 'b.png'), pair.real, 'INVALID_PAIR_IMAGE');

  const { vote, invalid: voteInvalid } = await readVoteForPair(pair, preferenceID, options);
  const status: PreferenceStatus = voteInvalid ? 'invalid' : vote === null ? 'pending' : vote.choice === 'skip' ? 'skipped' : 'rated';
  return { metadata, vote, status };
};

const imageUrls = (jobID: string, preferenceID: string) => ({
  a: `/api/jobs/${encodeURIComponent(jobID)}/preferences/${encodeURIComponent(preferenceID)}/image/a`,
  b: `/api/jobs/${encodeURIComponent(jobID)}/preferences/${encodeURIComponent(preferenceID)}/image/b`,
});

const TRAINING_METRIC_KEYS = ['feedback/updates', 'feedback/optimizer_updates', 'feedback/foreign_votes'] as const;
const SQLITE_SIDECAR_SUFFIXES = ['-wal', '-shm', '-journal'] as const;
// Each lookup seeks to the end of one (key, step) index range even on long runs.
const TRAINING_METRIC_LOOKUP_SQL = TRAINING_METRIC_KEYS.map(() => `
  SELECT * FROM (
    SELECT key, step, value_real FROM metrics
    WHERE key = ? AND value_real IS NOT NULL
    ORDER BY step DESC LIMIT 1
  )
`).join(' UNION ALL ');

type TrainingMetricRow = { key: string; step: number; value_real: number | null };

const latestTrainingMetric = (
  rows: TrainingMetricRow[],
  key: (typeof TRAINING_METRIC_KEYS)[number],
): PreferenceTrainingMetric | null => {
  const row = rows.find(candidate => candidate.key === key);
  if (!row || !isSafeInteger(row.step) || row.step < 0) return null;
  if (typeof row.value_real !== 'number' || !Number.isFinite(row.value_real)) return null;
  return { value: row.value_real, step: row.step };
};

// Query only the three latest feedback rows by the metrics (key, step) index. The read-only connection and bounded
// LIMIT keep this independent of the potentially large loss graph/history. Any open/query/lock failure is soft: the
// preference gallery remains usable and the panel simply has no recorded-training summary.
const readLatestTrainingMetrics = (dbPath: string): Promise<PreferenceFeedbackTrainingLog | undefined> =>
  new Promise(resolve => {
    const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READONLY, openError => {
      if (openError) {
        resolve(undefined);
        return;
      }
      db.all(TRAINING_METRIC_LOOKUP_SQL, [...TRAINING_METRIC_KEYS], (queryError, rows: TrainingMetricRow[]) => {
        if (queryError) {
          db.close(() => resolve(undefined));
          return;
        }
        const metricRows = Array.isArray(rows) ? rows : [];
        const training: PreferenceFeedbackTrainingLog = {
          preference_updates: latestTrainingMetric(metricRows, 'feedback/updates'),
          optimizer_updates: latestTrainingMetric(metricRows, 'feedback/optimizer_updates'),
          foreign_votes: latestTrainingMetric(metricRows, 'feedback/foreign_votes'),
        };

        const present = training.preference_updates || training.optimizer_updates || training.foreign_votes;
        db.close(() => resolve(present ? training : undefined));
      });
    });
    // A locked/corrupt/hostile log must not stall pair listing indefinitely.
    db.configure('busyTimeout', 750);
  });

const readSafeTrainingLog = async (bound: BoundRoot): Promise<PreferenceFeedbackTrainingLog | undefined> => {
  const lexical = path.join(bound.lexical, 'loss_log.db');
  let stat: fs.Stats;
  try {
    stat = await fs.promises.lstat(lexical);
  } catch {
    return undefined;
  }
  // The trainer owns a regular pre-existing SQLite file. Never follow a linked DB or a linked WAL/SHM/journal.
  if (stat.isSymbolicLink() || !stat.isFile()) return undefined;
  const real = await fs.promises.realpath(lexical).catch(() => null);
  if (!real || !isContained(bound.real, real)) return undefined;

  for (const suffix of SQLITE_SIDECAR_SUFFIXES) {
    const sidecar = `${lexical}${suffix}`;
    let sidecarStat: fs.Stats | null;
    try {
      sidecarStat = await fs.promises.lstat(sidecar);
    } catch (error: any) {
      if (error?.code === 'ENOENT') continue;
      return undefined;
    }
    if (!sidecarStat) continue;
    if (sidecarStat.isSymbolicLink() || !sidecarStat.isFile()) return undefined;
    const sidecarReal = await fs.promises.realpath(sidecar).catch(() => null);
    if (!sidecarReal || !isContained(bound.real, sidecarReal)) return undefined;
  }

  try {
    return await readLatestTrainingMetrics(real);
  } catch {
    return undefined;
  }
};

export const listPreferenceFeedback = async (
  job: BoundJob,
  trainingRoot: string,
): Promise<PreferenceFeedbackList> => {
  const enabled = isDpoFeedbackEnabled(job.job_config);
  const bound = await resolveBoundJobRoot(job, trainingRoot);
  const counts = { pending: 0, rated: 0, skipped: 0, invalid: 0 };
  if (!bound.preferencesReal) {
    const empty = { enabled, pairs: [], counts };
    if (!enabled) return empty;
    const training = await readSafeTrainingLog(bound);
    return training ? { ...empty, training } : empty;
  }

  let entries: fs.Dirent[];
  try {
    entries = await fs.promises.readdir(bound.preferencesLexical, { withFileTypes: true });
  } catch {
    throw new PreferenceFeedbackError('The preferences folder could not be listed.', 500, 'PREFERENCES_UNREADABLE');
  }

  const pairs: PreferencePair[] = [];
  for (const entry of entries) {
    // Only strict lowercase UUIDv4 directory names are eligible. Symlinks, stray files and malformed entries are ignored.
    if (!entry.isDirectory() || !isPreferenceID(entry.name)) continue;
    try {
      const pair = await resolvePairDirectory(bound, entry.name);
      const { metadata, vote, status } = await readPair(bound, entry.name, pair);
      counts[status] += 1;
      pairs.push({ ...metadata, status, vote, image_urls: imageUrls(job.id, entry.name) });
    } catch (error) {
      // One malformed pair must not make every healthy pair disappear. Security errors stay localized to that pair.
      if (!(error instanceof PreferenceFeedbackError)) {
        console.error('Unexpected preference pair read failure:', error);
      }
    }
  }

  pairs.sort((a, b) => b.step - a.step || b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id));
  const result = { enabled, pairs, counts };
  if (!enabled && pairs.length === 0) return result;
  const training = await readSafeTrainingLog(bound);
  return training ? { ...result, training } : result;
};

export const readPreferenceImage = async (
  job: BoundJob,
  trainingRoot: string,
  preferenceID: string,
  arm: string,
): Promise<{ bytes: Buffer; contentType: 'image/png' }> => {
  if (arm !== 'a' && arm !== 'b') {
    throw new PreferenceFeedbackError('Invalid image arm.', 400, 'INVALID_IMAGE_ARM');
  }
  const bound = await resolveBoundJobRoot(job, trainingRoot);
  const pair = await resolvePairDirectory(bound, preferenceID);
  await readPair(bound, preferenceID, pair);
  const file = path.join(pair.lexical, `${arm}.png`);
  await assertRegularFile(file, pair.real, 'INVALID_PAIR_IMAGE');
  const bytes = await fs.promises.readFile(file);
  return { bytes, contentType: 'image/png' };
};

type ExportImage = { path: string; identity: fs.Stats };
type PreferenceExportPair = {
  id: string;
  step: number;
  createdAt: string;
  chosen: ExportImage;
  rejected: ExportImage;
  prompt: string;
};

// Export eligibility deliberately reuses the same verified job folder, strict pair-id rule, metadata parser, image
// checks and vote resolver as the gallery. The read-only resolver is important: exporting must never heal or create a
// vote receipt as a side effect.
const listPreferenceExportPairs = async (bound: BoundRoot): Promise<PreferenceExportPair[]> => {
  if (!bound.preferencesReal) return [];

  let entries: fs.Dirent[];
  try {
    entries = await fs.promises.readdir(bound.preferencesLexical, { withFileTypes: true });
  } catch {
    throw new PreferenceFeedbackError('The preferences folder could not be listed.', 500, 'PREFERENCES_UNREADABLE');
  }

  const pairs: PreferenceExportPair[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || !isPreferenceID(entry.name)) continue;
    try {
      const pair = await resolvePairDirectory(bound, entry.name);
      const { metadata, vote, status } = await readPair(bound, entry.name, pair, { healProjection: false });
      if (status !== 'rated' || !vote || (vote.choice !== 'a' && vote.choice !== 'b')) continue;

      // The winning arm becomes chosen; the other original PNG becomes rejected. Fixed arm names prevent any
      // browser- or manifest-supplied path from reaching the filesystem or the archive entry names.
      const chosenArm = vote.choice;
      const rejectedArm = chosenArm === 'a' ? 'b' : 'a';
      const chosenPath = path.join(pair.real, `${chosenArm}.png`);
      const rejectedPath = path.join(pair.real, `${rejectedArm}.png`);

      pairs.push({
        id: entry.name,
        step: metadata.step,
        createdAt: metadata.created_at,
        chosen: { path: chosenPath, identity: await assertRegularFile(chosenPath, pair.real, 'INVALID_PAIR_IMAGE') },
        rejected: { path: rejectedPath, identity: await assertRegularFile(rejectedPath, pair.real, 'INVALID_PAIR_IMAGE') },
        prompt: metadata.prompt,
      });
    } catch (error) {
      // Pending/skipped/invalid/malformed pairs are omitted individually, matching gallery behavior.
      if (!(error instanceof PreferenceFeedbackError)) {
        console.error('Unexpected preference export pair read failure:', error);
      }
    }
  }

  pairs.sort((a, b) => b.step - a.step || b.createdAt.localeCompare(a.createdAt) || b.id.localeCompare(a.id));
  return pairs;
};

export interface PreferenceFeedbackExportStream {
  archive: ReturnType<typeof archiver>;
  pairCount: number;
  fileName: string;
}

// Prepare a streamed ZIP without writing a server-side artifact. Image bytes are appended as file streams so archiver
// applies backpressure as the response is consumed; only the small validated metadata and prompt are held in memory.
export const createPreferenceFeedbackExport = async (
  job: BoundJob,
  trainingRoot: string,
): Promise<PreferenceFeedbackExportStream> => {
  const bound = await resolveBoundJobRoot(job, trainingRoot);
  const pairs = await listPreferenceExportPairs(bound);
  if (pairs.length === 0) {
    throw new PreferenceFeedbackError(
      'No rated preference pairs are available to export.',
      404,
      'NO_RATED_PREFERENCE_PAIRS',
    );
  }

  const archive = archiver('zip', { store: true });
  const lifetime = new AbortController();
  archive.once('close', () => {
    lifetime.abort();
    archive.abort();
  });

  const appendEntry = async (name: string, source: fs.ReadStream | Buffer) => {
    const processed = once(archive, 'entry', { signal: lifetime.signal });
    if (!Buffer.isBuffer(source)) {
      source.once('error', error => archive.destroy(error));
    }
    try {
      archive.append(source, { name });
      await processed;
    } finally {
      if (!Buffer.isBuffer(source)) source.destroy();
    }
  };

  const appendImage = async (name: string, image: ExportImage) => {
    const file = await fs.promises.open(image.path, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
    try {
      const stat = await file.stat();
      if (!stat.isFile() || stat.dev !== image.identity.dev || stat.ino !== image.identity.ino) {
        throw new Error('The rated image changed during export.');
      }
      await appendEntry(name, fs.createReadStream(image.path, { fd: file }));
    } finally {
      await file.close();
    }
  };

  // Archiver starts consuming appended streams immediately. Waiting for each entry keeps file handles and
  // buffered image data bounded, including when an authenticated client downloads slowly or disconnects.
  const writeEntries = async () => {
    for (const pair of pairs) {
      await appendImage(`chosen/${pair.id}.png`, pair.chosen);
      await appendEntry(`chosen/${pair.id}.txt`, Buffer.from(pair.prompt, 'utf8'));
      await appendImage(`rejected/${pair.id}.png`, pair.rejected);
    }
    await archive.finalize();
  };
  void writeEntries().catch(error => {
    if (!archive.destroyed) archive.destroy(error);
  });

  return { archive, pairCount: pairs.length, fileName: 'preference-feedback.zip' };
};
export const castPreferenceVote = async (
  job: BoundJob,
  trainingRoot: string,
  preferenceID: string,
  choice: unknown,
): Promise<{ vote: PreferenceVote; created: boolean }> => {
  if (choice !== 'a' && choice !== 'b' && choice !== 'skip') {
    throw new PreferenceFeedbackError('choice must be "a", "b" or "skip".', 400, 'INVALID_CHOICE');
  }
  const bound = await resolveBoundJobRoot(job, trainingRoot);
  const pair = await resolvePairDirectory(bound, preferenceID);
  await readPair(bound, preferenceID, pair);

  // Legacy vote.json wins over any new decision. This keeps pre-existing clients and files authoritative.
  const topLevel = await readTopLevelVote(pair, preferenceID);
  if (topLevel.present) {
    if (!topLevel.vote) throw projectionInvalid();
    if (topLevel.vote.choice === choice) return { vote: topLevel.vote, created: false };
    throw projectionConflict();
  }

  // Recover a winner whose top-level projection was interrupted. Validate it before writing anything new.
  let interrupted: { present: boolean; vote: PreferenceVote | null };
  try {
    interrupted = await readDecisionVote(pair, preferenceID);
  } catch {
    throw projectionInvalid();
  }
  if (interrupted.present) {
    if (!interrupted.vote) throw projectionInvalid();
    await projectVote(pair, preferenceID, interrupted.vote);
    if (interrupted.vote.choice === choice) return { vote: interrupted.vote, created: false };
    throw projectionConflict();
  }

  const vote: PreferenceVote = {
    version: 1,
    id: preferenceID,
    choice,
    voted_at: new Date().toISOString(),
  };
  const decisionPath = path.join(pair.lexical, DECISION_DIRNAME);
  const stage = path.join(pair.lexical, `.vote.stage.${process.pid}.${randomUUID()}`);
  let handle: fs.promises.FileHandle | null = null;

  try {
    // Stage a complete directory. Only the atomic directory rename below can make it visible as a decision.
    await fs.promises.mkdir(stage, { mode: 0o700 });
    handle = await fs.promises.open(path.join(stage, VOTE_FILENAME), 'wx', 0o600);
    await handle.writeFile(serializeVote(vote));
    await handle.sync();
    await handle.close();
    handle = null;
    await fsyncDirectory(stage);

    try {
      // On POSIX and Windows a non-empty .decision directory cannot be replaced by this rename. The loser therefore
      // validates and adopts the winner rather than overwriting it.
      await fs.promises.rename(stage, decisionPath);
      await fsyncDirectory(pair.lexical);
    } catch (renameError) {
      let winner: { present: boolean; vote: PreferenceVote | null } | null = null;
      try {
        winner = await readDecisionVote(pair, preferenceID);
      } catch {
        winner = null;
      }
      if (!winner?.present) {
        throw new PreferenceFeedbackError('The vote could not be published safely.', 500, 'VOTE_WRITE_FAILED');
      }
      if (!winner.vote) throw projectionInvalid();
      try {
        await projectVote(pair, preferenceID, winner.vote);
      } catch (projectionError) {
        if (projectionError instanceof PreferenceFeedbackError && projectionError.status === 409) throw projectionError;
        // The committed decision is still valid; its top-level receipt can be healed by a later read or retry.
      }
      if (winner.vote.choice === choice) return { vote: winner.vote, created: false };
      throw projectionConflict();
    }

    try {
      await projectVote(pair, preferenceID, vote);
    } catch (projectionError) {
      if (projectionError instanceof PreferenceFeedbackError && projectionError.status === 409) throw projectionError;
      throw new PreferenceFeedbackError(
        'The vote was committed but its top-level receipt could not be projected.',
        500,
        'VOTE_WRITE_FAILED',
      );
    }
    return { vote, created: true };
  } catch (error) {
    if (error instanceof PreferenceFeedbackError) throw error;
    throw new PreferenceFeedbackError('The vote could not be published safely.', 500, 'VOTE_WRITE_FAILED');
  } finally {
    if (handle) await handle.close().catch(() => {});
    await fs.promises.rm(stage, { recursive: true, force: true }).catch(() => {});
  }
};
