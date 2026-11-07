// Cross-process operation lock (docs/STATE_MACHINES.md M19 uploads, M35 job deletion): an EXCLUSIVE transaction on a
// small sidecar SQLite file, held for the whole operation (job deletion, one upload's chunk/finish/cancel). SQLite
// takes an OS file lock,
// so the lock is released when the holder's process dies; there is no owner file to judge stale and no unlink, so
// a replacement owner can never be removed by a waiter (the ABA problem of create-exclusive lock files). Nothing
// durable lives in this transaction: operations keep their state in their own files, written before/after I/O.
import fs from 'fs';
import path from 'path';
import sqlite3 from 'sqlite3';

export class OpLockBusy extends Error {}

const open = (file: string) =>
  new Promise<sqlite3.Database>((resolve, reject) => {
    const db = new sqlite3.Database(file, err => (err ? reject(err) : resolve(db)));
  });
const exec = (db: sqlite3.Database, sql: string) => new Promise<void>((resolve, reject) => db.exec(sql, err => (err ? reject(err) : resolve())));
const close = (db: sqlite3.Database) => new Promise<void>(resolve => db.close(() => resolve()));

const isBusy = (e: any) => e?.code === 'SQLITE_BUSY' || /SQLITE_BUSY|database is locked/i.test(String(e?.message ?? ''));

// Runs fn while holding the lock. Waits at most waitMs for another holder, then throws OpLockBusy (callers turn that
// into a "busy, retry" response; a request never hangs on an unknown owner).
export const withOpLock = async <T>(lockFile: string, fn: () => Promise<T>, waitMs = 0): Promise<T> => {
  fs.mkdirSync(path.dirname(lockFile), { recursive: true });
  const db = await open(lockFile);
  try {
    db.configure('busyTimeout', waitMs);
    try {
      await exec(db, 'BEGIN EXCLUSIVE');
    } catch (e: any) {
      if (isBusy(e)) throw new OpLockBusy(`another operation holds ${path.basename(lockFile)}`);
      throw e;
    }
    try {
      return await fn();
    } finally {
      await exec(db, 'ROLLBACK').catch(() => {});
    }
  } finally {
    await close(db);
  }
};
