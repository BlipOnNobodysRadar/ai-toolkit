// Chunked upload sessions for /api/loras/upload (docs/STATE_MACHINES.md M19).
//
// One directory per upload under <loras>/.uploads/<id>/: state.json (the durable record: immutable name/size/
// overwrite, every received chunk with its length and SHA-256, publication intent and result), data.part (the
// bytes) and op.lock (the crash-released operation lock, ui/cron/ownership/opLock.ts). Every chunk, finish and
// cancel runs entirely under that lock, so no operation can overlap another one's file I/O, in any process.
//
//   receiving --chunk--> receiving (only after the bytes are written and fsynced is the chunk recorded)
//   receiving --finish, complete coverage--> publishing (intent + part file identity recorded FIRST)
//   publishing --link (no overwrite) / rename (overwrite) + verified identity--> published (result kept for retries)
//   publishing --link unsupported here--> publish_unsupported (part kept, explicit error; no copy fallback)
//   publishing --identity no longer provable--> indeterminate (evidence kept, explicit error)
//   receiving/... --cancel--> cancelled (terminal record kept; a failed part removal is reported, not hidden)
// A finish interrupted at any point (even after the link/rename, before state.json says published) is completed by
// the next finish/cancel from the recorded intent: the destination is ours iff it is the part file's identity.
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { OpLockBusy, withOpLock } from '../../cron/ownership/opLock';

export type UploadState = {
  version: 1;
  id: string;
  name: string;
  size: number;
  overwrite: boolean;
  state: 'receiving' | 'publishing' | 'published' | 'cancelled' | 'publish_unsupported' | 'indeterminate';
  chunks: Record<string, { len: number; sha256: string }>;
  created_at: number;
  publish?: { dest: string; mode: 'link' | 'rename'; ino: string; size: number };
  result?: { path: string; name: string; size: number };
  error?: string;
  cleanup_error?: string;
};

export type Reply = { status: number; body: any };

// Test seams (fault injection); production passes nothing.
export type UploadDeps = {
  lockWaitMs?: number;
  link?: (src: string, dst: string) => Promise<void>;
  unlink?: (p: string) => Promise<void>;
  saveState?: (file: string, st: UploadState) => Promise<void>;
  hooks?: {
    duringWrite?: () => Promise<void> | void;
    afterLink?: () => Promise<void> | void;
    afterRename?: () => Promise<void> | void;
  };
};

const STALE_MS = 24 * 3600 * 1000;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const UNSUPPORTED_LINK = ['EPERM', 'ENOTSUP', 'EXDEV', 'EINVAL', 'ENOSYS', 'EOPNOTSUPP'];

export const sessionPaths = (partDir: string, id: string) => {
  const dir = path.join(partDir, id);
  return { dir, state: path.join(dir, 'state.json'), part: path.join(dir, 'data.part'), lock: path.join(dir, 'op.lock') };
};

const reply = (status: number, body: any): Reply => ({ status, body });
const err = (status: number, error: string, extra: any = {}) => reply(status, { error, ...extra });

const writeAtomic = async (file: string, text: string) => {
  const tmp = `${file}.${process.pid}.${Date.now()}.tmp`;
  const fh = await fs.promises.open(tmp, 'w');
  try {
    await fh.writeFile(text);
    await fh.sync();
  } finally {
    await fh.close();
  }
  await fs.promises.rename(tmp, file);
};

const readState = async (file: string): Promise<UploadState | null> => {
  try {
    return JSON.parse(await fs.promises.readFile(file, 'utf8'));
  } catch (e: any) {
    if (e?.code === 'ENOENT') return null;
    throw e;
  }
};
const saveState = (file: string, st: UploadState) => writeAtomic(file, JSON.stringify(st));

const statOrNull = async (p: string) => {
  try {
    return await fs.promises.stat(p, { bigint: true });
  } catch (e: any) {
    if (e?.code === 'ENOENT') return null;
    throw e;
  }
};

const missingRanges = (st: UploadState): [number, number][] => {
  const spans = Object.entries(st.chunks)
    .map(([o, c]) => [Number(o), Number(o) + c.len] as [number, number])
    .sort((a, b) => a[0] - b[0]);
  const missing: [number, number][] = [];
  let at = 0;
  for (const [s, e] of spans) {
    if (s > at) missing.push([at, s]);
    at = Math.max(at, e);
  }
  if (at < st.size) missing.push([at, st.size]);
  return missing;
};

const locked = async (lockFile: string, deps: UploadDeps, fn: () => Promise<Reply>): Promise<Reply> => {
  try {
    return await withOpLock(lockFile, fn, deps.lockWaitMs ?? 5000);
  } catch (e: any) {
    if (e instanceof OpLockBusy) return err(503, 'This upload is busy with another request; retry.', { busy: true });
    throw e;
  }
};

export const validId = (id: string | null): id is string => !!id && UUID_RE.test(id);

// ---------------------------------------------------------------- start
export const startUpload = async (partDir: string, destRoot: string, name: string, size: number, overwrite: boolean): Promise<Reply> => {
  if (!overwrite && fs.existsSync(path.join(destRoot, name))) {
    return err(409, 'A file with that name already exists', { exists: true });
  }
  await fs.promises.mkdir(partDir, { recursive: true });
  await removeStaleSessions(partDir);
  const id = crypto.randomUUID();
  const p = sessionPaths(partDir, id);
  await fs.promises.mkdir(p.dir);
  await fs.promises.writeFile(p.part, Buffer.alloc(0));
  await saveState(p.state, { version: 1, id, name, size, overwrite, state: 'receiving', chunks: {}, created_at: Date.now() });
  return reply(200, { uploadId: id, fileName: name });
};

// ---------------------------------------------------------------- chunk
export const writeChunk = async (partDir: string, id: string, offset: number, bytes: Buffer, deps: UploadDeps = {}): Promise<Reply> => {
  const p = sessionPaths(partDir, id);
  if (!fs.existsSync(p.state)) return err(404, 'Upload not found or expired');
  return locked(p.lock, deps, async () => {
    const st = await readState(p.state);
    if (!st) return err(404, 'Upload not found or expired');
    if (st.state === 'cancelled') return err(410, 'Upload was cancelled');
    if (st.state !== 'receiving') return err(409, `Upload is ${st.state}; no more chunks are accepted`);
    if (!Number.isSafeInteger(offset) || offset < 0) return err(400, 'Invalid offset');
    if (bytes.length === 0) return err(400, 'Empty chunk');
    if (offset + bytes.length > st.size) return err(400, `Chunk [${offset}, ${offset + bytes.length}) is beyond the declared size ${st.size}`);
    const sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
    const known = st.chunks[String(offset)];
    if (known) {
      // a retry of a recorded chunk is accepted only if it is the same bytes
      if (known.len === bytes.length && known.sha256 === sha256) return reply(200, { received: bytes.length, duplicate: true });
      return err(409, `Chunk at offset ${offset} was already received with different content`, { conflict: true });
    }
    for (const [o, c] of Object.entries(st.chunks)) {
      const s = Number(o);
      if (offset < s + c.len && s < offset + bytes.length) {
        return err(409, `Chunk [${offset}, ${offset + bytes.length}) overlaps received chunk [${s}, ${s + c.len})`, { conflict: true });
      }
    }
    const fh = await fs.promises.open(p.part, 'r+');
    try {
      let done = 0;
      while (done < bytes.length) {
        const { bytesWritten } = await fh.write(bytes, done, bytes.length - done, offset + done);
        if (bytesWritten <= 0) throw new Error('write made no progress');
        done += bytesWritten;
        if (deps.hooks?.duringWrite) await deps.hooks.duringWrite();
      }
      await fh.sync();
    } finally {
      await fh.close();
    }
    // recorded only now: an interrupted write is simply not acknowledged, and its retry rewrites the range
    st.chunks[String(offset)] = { len: bytes.length, sha256 };
    await saveState(p.state, st);
    return reply(200, { received: bytes.length });
  });
};

// ---------------------------------------------------------------- finish / publication
const published = async (p: ReturnType<typeof sessionPaths>, st: UploadState, deps: UploadDeps): Promise<Reply> => {
  st.state = 'published';
  st.result = { path: st.publish!.dest, name: st.name, size: st.size };
  await saveState(p.state, st);
  if (st.publish!.mode === 'link') {
    try {
      await (deps.unlink ?? fs.promises.unlink)(p.part); // only the second name of the published file
    } catch (e: any) {
      if (e?.code !== 'ENOENT') {
        st.cleanup_error = `could not remove the upload's temporary name: ${e?.message ?? e}`;
        await saveState(p.state, st);
      }
    }
  }
  return reply(200, st.result);
};

const indeterminate = async (p: ReturnType<typeof sessionPaths>, st: UploadState, why: string): Promise<Reply> => {
  st.state = 'indeterminate';
  st.error = why;
  await saveState(p.state, st);
  return err(500, `Upload outcome cannot be established: ${why}. Nothing was overwritten; its files are kept for inspection.`, { indeterminate: true });
};

// Completes (or safely abandons) a recorded publication. Runs under the session lock.
const publish = async (p: ReturnType<typeof sessionPaths>, st: UploadState, deps: UploadDeps): Promise<Reply> => {
  const intent = st.publish!;
  const dest = await statOrNull(intent.dest);
  if (dest && String(dest.ino) === intent.ino) return published(p, st, deps); // already done before an interruption
  const part = await statOrNull(p.part);
  const partIsOurs = !!part && String(part.ino) === intent.ino && Number(part.size) === intent.size;
  if (intent.mode === 'link') {
    if (dest) {
      // someone else's file has that name: never replaced
      if (partIsOurs) {
        st.state = 'receiving';
        delete st.publish;
        await saveState(p.state, st);
        return err(409, 'A file with that name already exists', { exists: true });
      }
      return indeterminate(p, st, 'the destination is another file and the upload data has moved');
    }
    if (!partIsOurs) return indeterminate(p, st, 'the upload data is no longer the file that was received');
    try {
      await (deps.link ?? fs.promises.link)(p.part, intent.dest);
    } catch (e: any) {
      if (e?.code === 'EEXIST') {
        st.state = 'receiving';
        delete st.publish;
        await saveState(p.state, st);
        return err(409, 'A file with that name already exists', { exists: true });
      }
      if (UNSUPPORTED_LINK.includes(e?.code)) {
        st.state = 'publish_unsupported';
        st.error = `this file system cannot publish without overwriting (hard link failed: ${e.code})`;
        await saveState(p.state, st);
        return err(500, `Upload received but not published: ${st.error}. The data is kept in ${p.part}.`, { unsupported: true });
      }
      throw e; // still 'publishing': the next finish retries from the intent
    }
    if (deps.hooks?.afterLink) await deps.hooks.afterLink();
    return published(p, st, deps);
  }
  // overwrite: rename replaces the destination (what the user asked for)
  if (!partIsOurs) return indeterminate(p, st, 'neither the destination nor the upload data is the file that was received');
  await fs.promises.rename(p.part, intent.dest);
  if (deps.hooks?.afterRename) await deps.hooks.afterRename();
  return published(p, st, deps);
};

export const finishUpload = async (partDir: string, destRoot: string, id: string, deps: UploadDeps = {}): Promise<Reply> => {
  const p = sessionPaths(partDir, id);
  if (!fs.existsSync(p.state)) return err(404, 'Upload not found or expired');
  return locked(p.lock, deps, async () => {
    const st = await readState(p.state);
    if (!st) return err(404, 'Upload not found or expired');
    if (st.state === 'published') return reply(200, st.result); // a retry after a lost reply
    if (st.state === 'cancelled') return err(410, 'Upload was cancelled');
    if (st.state === 'publish_unsupported' || st.state === 'indeterminate') return err(500, st.error ?? st.state, { [st.state]: true });
    if (st.state === 'publishing') return publish(p, st, deps); // interrupted earlier: complete from the intent
    const missing = missingRanges(st);
    if (missing.length) return err(409, `Upload incomplete: ${missing.length} missing range(s)`, { missing });
    const part = await statOrNull(p.part);
    if (!part || Number(part.size) !== st.size) return err(409, `Upload data has ${part ? Number(part.size) : 0} of ${st.size} bytes`);
    const dest = path.join(destRoot, st.name);
    if (!st.overwrite && (await statOrNull(dest))) return err(409, 'A file with that name already exists', { exists: true });
    st.state = 'publishing';
    st.publish = { dest, mode: st.overwrite ? 'rename' : 'link', ino: String(part.ino), size: st.size };
    await saveState(p.state, st); // durable intent before anything becomes visible
    return publish(p, st, deps);
  });
};

// ---------------------------------------------------------------- cancel / expiry
export const cancelUpload = async (partDir: string, id: string, deps: UploadDeps = {}): Promise<Reply> => {
  const p = sessionPaths(partDir, id);
  if (!fs.existsSync(p.state)) return reply(200, { ok: true });
  return locked(p.lock, deps, async () => {
    const st = await readState(p.state);
    if (!st) return reply(200, { ok: true });
    if (st.state === 'publishing') {
      const r = await publish(p, st, deps); // settle an interrupted publication first (it records its outcome)
      if ((await readState(p.state))?.state === 'published') return reply(409, { error: 'The upload was already published', result: r.body });
    }
    if (st.state === 'published') return reply(409, { error: 'The upload was already published', result: st.result });
    const save = deps.saveState ?? saveState;
    if (st.state !== 'cancelled') {
      // the terminal decision is durable BEFORE any data is removed: a failure here changes nothing (still receiving,
      // data intact, error returned); a failure after it can no longer leave acknowledged chunks without their bytes
      st.state = 'cancelled';
      await save(p.state, st);
    }
    // (retried) cleanup under the same lock
    let cleanupError: string | undefined;
    try {
      await (deps.unlink ?? fs.promises.unlink)(p.part);
    } catch (e: any) {
      if (e?.code !== 'ENOENT') cleanupError = `could not remove the upload data: ${e?.message ?? e}`;
    }
    if (cleanupError !== st.cleanup_error) {
      st.cleanup_error = cleanupError;
      await save(p.state, st).catch(() => {}); // only the note; the session is cancelled either way
    }
    return reply(200, { ok: true, cleanup_error: cleanupError });
  });
};

// Old sessions (over a day) are removed under their own lock; a session busy right now is left. Kept until an
// explicit Cancel instead: 'publishing' (needs recovery), 'publish_unsupported' and 'indeterminate' (their retained
// data is the evidence/recovery the error message promises).
export const removeStaleSessions = async (partDir: string) => {
  let entries: string[];
  try {
    entries = await fs.promises.readdir(partDir);
  } catch {
    return;
  }
  const now = Date.now();
  for (const e of entries) {
    const full = path.join(partDir, e);
    if (e.endsWith('.part')) {
      // pre-update layout (<id>.part, no state): unusable by this protocol; same 24 h rule as before
      try {
        const s = await fs.promises.stat(full);
        if (now - s.mtimeMs > STALE_MS) await fs.promises.unlink(full);
      } catch {
        // ignore
      }
      continue;
    }
    if (!validId(e)) continue;
    const p = sessionPaths(partDir, e);
    let removed = false;
    try {
      removed = await withOpLock(
        p.lock,
        async () => {
          const st = await readState(p.state);
          if (!st || ['publishing', 'publish_unsupported', 'indeterminate'].includes(st.state) || now - st.created_at <= STALE_MS) return false;
          await fs.promises.rm(p.part, { force: true });
          await fs.promises.rm(p.state, { force: true });
          return true;
        },
        0,
      );
    } catch {
      // busy or unreadable: leave it
    }
    // the lock file is closed now; a request opening it after this finds no state and gets 404
    if (removed) await fs.promises.rm(p.dir, { recursive: true, force: true }).catch(() => {});
  }
};
