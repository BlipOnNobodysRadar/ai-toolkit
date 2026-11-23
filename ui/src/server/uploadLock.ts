// Serialize changes to an upload across requests. SQLite releases the file lock if the holder exits.
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
