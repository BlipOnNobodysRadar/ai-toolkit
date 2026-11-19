import { createHash } from 'crypto';
import path from 'path';
import { canonicalFolder } from './resources';

// Shared with toolkit/save_root_lease.py. The SQLite operation lock lives outside the save root, so deletion cannot
// unlink the lock and let another writer acquire a replacement. Never remove these files to "clear" a busy owner.
export const writerLeasePath = (folder: string): string => {
  const canonical = canonicalFolder(path.dirname(folder), path.basename(folder));
  const key = createHash('sha256').update(canonical, 'utf8').digest('hex');
  return path.join(path.dirname(canonical), '.aitk_locks', `writer.${key}.lock`);
};
