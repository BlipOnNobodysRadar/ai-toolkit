import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { jobFolderError } from '../../cron/ownership/resources';

// Job-scoped reader/writer for MOSS preference-feedback pairs.
//
// The browser supplies only a job id, a pair id and a fixed arm/choice. Every filesystem path is derived from the
// persisted job binding; pair metadata and vote files are treated as untrusted input. The image route is never a
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

export interface PreferenceFeedbackList {
  enabled: boolean;
  pairs: PreferencePair[];
  counts: { pending: number; rated: number; skipped: number; invalid: number };
}

type BoundJob = {
  id: string;
  name: string;
  run_dir?: string | null;
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

const sameName = (a: string, b: string): boolean =>
  process.platform === 'win32' ? a.toLowerCase() === b.toLowerCase() : a === b;

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

// Bind to the exact folder the invariant launch code admitted. `run_dir` is the durable claim; old rows without one
// fall back to the same direct-child rule used by admission. A nested symlink/junction is refused rather than followed.
export const resolveBoundJobRoot = async (job: BoundJob, trainingRoot: string): Promise<BoundRoot> => {
  if (!job || !isUuidV4(job.id)) {
    throw new PreferenceFeedbackError('Invalid job id.', 400, 'INVALID_JOB_ID');
  }
  const nameError = jobFolderError(trainingRoot, job.name);
  if (nameError) throw new PreferenceFeedbackError(nameError, 400, 'INVALID_JOB_FOLDER');

  const configuredRoot = path.resolve(trainingRoot);
  const configuredStat = await fs.promises.stat(configuredRoot).catch(() => null);
  if (!configuredStat || !configuredStat.isDirectory()) {
    throw new PreferenceFeedbackError('The configured training folder is unavailable.', 500, 'TRAINING_FOLDER_UNAVAILABLE');
  }
  const realConfiguredRoot = await fs.promises.realpath(configuredRoot).catch(() => {
    throw new PreferenceFeedbackError('The configured training folder cannot be resolved safely.', 500, 'TRAINING_FOLDER_UNAVAILABLE');
  });

  const expected = path.resolve(configuredRoot, job.name);
  let lexical = expected;
  if (job.run_dir !== undefined && job.run_dir !== null && job.run_dir !== '') {
    if (typeof job.run_dir !== 'string') {
      throw new PreferenceFeedbackError('The job has an invalid run folder binding.', 409, 'INVALID_RUN_BINDING');
    }
    const bound = path.resolve(job.run_dir);
    if (!sameName(path.dirname(bound), configuredRoot) || !sameName(path.basename(bound), job.name)) {
      throw new PreferenceFeedbackError('The job run folder is not the claimed job folder.', 409, 'INVALID_RUN_BINDING');
    }
    lexical = bound;
  }

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

const readPair = async (bound: BoundRoot, preferenceID: string, pair: { lexical: string; real: string }) => {
  const metadata = parsePairMetadata(
    await readJsonFile(path.join(pair.lexical, 'pair.json'), pair.real, MAX_PAIR_JSON_BYTES, 'INVALID_PAIR'),
    preferenceID,
  );
  // The two displayed images are exact fixed names. Validate both before publishing the pair to the UI.
  await assertRegularFile(path.join(pair.lexical, 'a.png'), pair.real, 'INVALID_PAIR_IMAGE');
  await assertRegularFile(path.join(pair.lexical, 'b.png'), pair.real, 'INVALID_PAIR_IMAGE');

  let vote: PreferenceVote | null = null;
  let voteInvalid = false;
  const votePath = path.join(pair.lexical, 'vote.json');
  const voteStat = await lstatOptional(votePath);
  if (voteStat) {
    try {
      vote = parseVote(await readJsonFile(votePath, pair.real, MAX_VOTE_JSON_BYTES, 'INVALID_VOTE'), preferenceID);
      if (!vote) voteInvalid = true;
    } catch {
      voteInvalid = true;
    }
  }
  const status: PreferenceStatus = voteInvalid ? 'invalid' : vote === null ? 'pending' : vote.choice === 'skip' ? 'skipped' : 'rated';
  return { metadata, vote, status };
};

const imageUrls = (jobID: string, preferenceID: string) => ({
  a: `/api/jobs/${encodeURIComponent(jobID)}/preferences/${encodeURIComponent(preferenceID)}/image/a`,
  b: `/api/jobs/${encodeURIComponent(jobID)}/preferences/${encodeURIComponent(preferenceID)}/image/b`,
});

export const listPreferenceFeedback = async (
  job: BoundJob,
  trainingRoot: string,
): Promise<PreferenceFeedbackList> => {
  const enabled = isDpoFeedbackEnabled(job.job_config);
  const bound = await resolveBoundJobRoot(job, trainingRoot);
  const counts = { pending: 0, rated: 0, skipped: 0, invalid: 0 };
  if (!bound.preferencesReal) return { enabled, pairs: [], counts };

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
  return { enabled, pairs, counts };
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

const readExistingVote = async (pair: { lexical: string; real: string }, preferenceID: string): Promise<PreferenceVote | null> => {
  const votePath = path.join(pair.lexical, 'vote.json');
  const stat = await lstatOptional(votePath);
  if (!stat) return null;
  return parseVote(await readJsonFile(votePath, pair.real, MAX_VOTE_JSON_BYTES, 'INVALID_VOTE'), preferenceID);
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

  const votePath = path.join(pair.lexical, 'vote.json');
  const vote: PreferenceVote = {
    version: 1,
    id: preferenceID,
    choice,
    voted_at: new Date().toISOString(),
  };
  const bytes = Buffer.from(JSON.stringify(vote, null, 2) + '\n', 'utf8');
  const temp = path.join(pair.lexical, `.vote.${process.pid}.${randomUUID()}.tmp`);
  let handle: fs.promises.FileHandle | null = null;
  let linked = false;
  try {
    handle = await fs.promises.open(temp, 'wx', 0o600);
    await handle.writeFile(bytes);
    await handle.sync();
    await handle.close();
    handle = null;
    try {
      // Hard-link publication is atomic and cannot replace an existing vote.json. A same-choice retry is resolved
      // below from the immutable winner; a different choice is a real conflict and stays 409 forever.
      await fs.promises.link(temp, votePath);
      linked = true;
    } catch (error: any) {
      if (error?.code !== 'EEXIST') throw error;
    }
  } catch (error) {
    if (error instanceof PreferenceFeedbackError) throw error;
    throw new PreferenceFeedbackError('The vote could not be published safely.', 500, 'VOTE_WRITE_FAILED');
  } finally {
    if (handle) await handle.close().catch(() => {});
    await fs.promises.unlink(temp).catch(() => {});
  }

  if (linked) return { vote, created: true };

  let existing: PreferenceVote | null = null;
  try {
    existing = await readExistingVote(pair, preferenceID);
  } catch {
    existing = null;
  }
  if (!existing) {
    throw new PreferenceFeedbackError('An existing vote record is invalid; refusing to overwrite it.', 409, 'VOTE_CONFLICT');
  }
  if (existing.choice === choice) return { vote: existing, created: false };
  throw new PreferenceFeedbackError('This preference pair already has a different immutable vote.', 409, 'VOTE_CONFLICT');
};
