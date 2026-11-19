// Resources a run owns while its process may be alive (docs/STATE_MACHINES.md M10, "Resources and claims").
// Shared by the worker (admission/reconciliation) and the API routes (start/edit/delete guards).
import fs from 'fs';
import path from 'path';

export type ResourceSet = {
  // numeric GPU indices ('0', '1') and 'mps'
  gpus: string[];
  // owns every GPU: an opaque token (UUID/MIG name) whose aliasing with indices is unknown, or a legacy set that
  // could not be parsed (an empty CUDA_VISIBLE_DEVICES still exposes every GPU on this platform)
  exclusive: boolean;
  // canonical training folder; every job type owns one
  folder: string;
};

export type GpuParse = { ok: true; gpus: string[]; exclusive: boolean; cpuOnly: boolean } | { ok: false; error: string };

export const EMPTY_GPU_ERROR =
  'No GPU is selected for this job. Choose a GPU (or "-1" for a CPU-only job) and save it again.';

// gpu_ids is passed verbatim as CUDA_VISIBLE_DEVICES (startJob). The UI pickers write comma-separated indices and
// macOS forces 'mps'. Ordering and duplicates do not change which devices are owned: '1,0' owns the same as '0,1'.
// '-1' alone is an explicit CPU-only job (owns only its folder). Anything else is an opaque token: owned
// exclusively, because nothing here can tell which index a UUID or MIG name refers to.
// legacy=true is for rows launched before run ownership: an empty or unparsable set is treated as exclusive
// instead of refused, since such a process may be using any GPU.
export const parseGpuIds = (raw: string | null | undefined, legacy = false): GpuParse => {
  const tokens = (raw ?? '')
    .split(',')
    .map(t => t.trim())
    .filter(t => t !== '');
  if (tokens.length === 0) {
    return legacy ? { ok: true, gpus: [], exclusive: true, cpuOnly: false } : { ok: false, error: EMPTY_GPU_ERROR };
  }
  if (tokens.length === 1 && tokens[0] === '-1') return { ok: true, gpus: [], exclusive: false, cpuOnly: true };
  if (tokens.includes('-1')) {
    if (legacy) return { ok: true, gpus: [], exclusive: true, cpuOnly: false };
    return { ok: false, error: `"-1" (CPU only) cannot be combined with GPUs: "${raw}"` };
  }
  const gpus = new Set<string>();
  let exclusive = false;
  for (const t of tokens) {
    if (/^\d+$/.test(t)) gpus.add(String(Number(t)));
    else if (t.toLowerCase() === 'mps') gpus.add('mps');
    else exclusive = true;
  }
  const sorted = [...gpus].sort((a, b) => (a === 'mps' ? 1 : b === 'mps' ? -1 : Number(a) - Number(b)));
  return { ok: true, gpus: exclusive ? [] : sorted, exclusive, cpuOnly: false };
};

// Raised when a folder's physical identity cannot be established (permissions, I/O): a lexical path in its place
// could give one directory two claim keys, so callers refuse/wait instead of guessing.
export class FolderIdentityError extends Error {}

// The real directory an absolute path names: its nearest existing ancestor resolved through junctions, symlinks
// and short names (realpath.native), plus the components that do not exist yet. Two job names (or two spellings of
// the training root) that reach one physical folder through a junction therefore get one key. Only a genuinely
// missing component is skipped over; any other failure is a FolderIdentityError.
const physicalPath = (abs: string): string => {
  const tail: string[] = [];
  let cur = abs;
  for (;;) {
    try {
      const real = fs.realpathSync.native(cur);
      return tail.length ? path.join(real, ...tail.reverse()) : real;
    } catch (e: any) {
      const parent = path.dirname(cur);
      if (e?.code !== 'ENOENT' || parent === cur) {
        throw new FolderIdentityError(`Cannot determine the real location of ${cur} (${e?.code ?? e?.message ?? e}); the job folder cannot be claimed safely.`);
      }
      tail.push(path.basename(cur));
      cur = parent;
    }
  }
};

// The folder a job runs in, normalized so that two names Windows (or default macOS) would map to one directory
// compare equal: physical path (above), case-folded, and on Windows trailing dots/spaces of each component dropped
// as the OS does.
export const canonicalFolder = (trainingRoot: string, name: string, platform: string = process.platform): string => {
  const lexical = path.resolve(trainingRoot, name);
  const abs = platform === process.platform ? physicalPath(lexical) : lexical;
  if (platform === 'win32') {
    const parts = abs.split(/[\\/]+/).map((p, i) => (i === 0 ? p : p.replace(/[. ]+$/, '')));
    return parts.join('\\').toLowerCase();
  }
  if (platform === 'darwin') return abs.normalize('NFC').toLowerCase();
  return abs;
};

// A job folder must be a direct child of the training root (a name such as '..' or 'a/b' must never make start or
// delete act outside it).
export const jobFolderError = (trainingRoot: string, name: string | null | undefined): string | null => {
  if (typeof name !== 'string' || name.trim() === '') return 'Job name is required';
  const root = path.resolve(trainingRoot);
  const folder = path.resolve(root, name);
  if (path.dirname(folder) !== root || folder === root) return `Invalid job name "${name}"`;
  // the operation-lock directory (store.ts LOCK_DIR) lives here; refuse every spelling Windows maps to it
  if (name.trim().replace(/[. ]+$/, '').toLowerCase() === '.aitk_locks') return `"${name}" is reserved`;
  if (process.platform === 'win32' && name.replace(/[. ]+$/, '') === '') return `Invalid job name "${name}"`;
  return null;
};

export const resourceSet = (gpu: { gpus: string[]; exclusive: boolean }, folder: string): ResourceSet => ({
  gpus: gpu.gpus,
  exclusive: gpu.exclusive,
  folder,
});

// Claim rows for a set. 'gpu:*' marks an exclusive GPU owner.
export const claimKeys = (set: ResourceSet): string[] => [
  `folder:${set.folder}`,
  ...(set.exclusive ? ['gpu:*'] : set.gpus.map(g => `gpu:${g}`)),
];

// Rebuilds a set from claim rows (folder-only sets come from deletion reservations or CPU-only runs).
export const setFromClaims = (resources: string[]): ResourceSet => {
  const set: ResourceSet = { gpus: [], exclusive: false, folder: '' };
  for (const r of resources) {
    if (r.startsWith('folder:')) set.folder = r.slice('folder:'.length);
    else if (r === 'gpu:*') set.exclusive = true;
    else if (r.startsWith('gpu:')) set.gpus.push(r.slice('gpu:'.length));
  }
  return set;
};

const usesGpu = (s: ResourceSet) => s.exclusive || s.gpus.length > 0;

// Folder ownership conflicts when two canonical folders are the same directory OR one lies inside the other
// (a junction can put one job's folder under another's; deleting the outer one would remove the inner). Compared at
// path-component boundaries, so 'a' and 'ab' stay disjoint.
const inside = (inner: string, outer: string) =>
  inner.length > outer.length && inner.startsWith(outer) && (/[\\/]$/.test(outer) || inner[outer.length] === '\\' || inner[outer.length] === '/');
export const foldersOverlap = (a: string, b: string) => !!a && !!b && (a === b || inside(a, b) || inside(b, a));

// What two sets have in common, as human-readable labels ([] = disjoint).
export const overlap = (a: ResourceSet, b: ResourceSet): string[] => {
  const out: string[] = [];
  if (foldersOverlap(a.folder, b.folder)) out.push(a.folder === b.folder ? 'job folder' : 'job folder (nested in or containing it)');
  if ((a.exclusive && usesGpu(b)) || (b.exclusive && usesGpu(a))) out.push('all GPUs');
  else {
    const shared = a.gpus.filter(g => b.gpus.includes(g));
    if (shared.length) out.push(`GPU ${shared.join(',')}`);
  }
  return out;
};
