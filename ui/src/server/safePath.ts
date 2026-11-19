import fs from 'fs';
import path from 'path';

// Filesystem safety for UI mutation routes.
//
// The lexical check rejects `..`, sibling-prefix paths and a root itself before any access. The physical check then
// compares real paths: an explicitly configured root may itself be a junction/symlink, but a nested link that leaves
// that root is not accepted for a mutation. This is a check-then-act guard: it does not protect against an adversarial
// actor swapping a link/junction concurrently between validation and unlink/write. Callers must not claim that.
export class UnsafePathError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UnsafePathError';
  }
}

export type ResolvedMutationPath = {
  configuredRoot: string;
  realRoot: string;
  lexicalPath: string;
  realPath: string;
};

const isContained = (root: string, candidate: string, allowEqual = false): boolean => {
  const rel = path.relative(root, candidate);
  if (rel === '') return allowEqual;
  return !path.isAbsolute(rel) && rel !== '..' && !rel.startsWith(`..${path.sep}`);
};

const realpath = async (file: string): Promise<string> => {
  try {
    return await fs.promises.realpath(file);
  } catch (e: any) {
    throw new UnsafePathError(`Cannot resolve the real location of ${file} (${e?.code ?? e?.message ?? e})`);
  }
};

// Resolve a destination that may not exist yet. If the destination itself exists (including as a link), its real path
// must resolve. Otherwise resolve the nearest existing ancestor physically and append only the missing components.
const physicalDestination = async (abs: string): Promise<string> => {
  let stat;
  try {
    stat = await fs.promises.lstat(abs);
  } catch (e: any) {
    if (e?.code !== 'ENOENT') {
      throw new UnsafePathError(`Cannot inspect ${abs} (${e?.code ?? e?.message ?? e})`);
    }
    const parent = path.dirname(abs);
    if (parent === abs) throw new UnsafePathError(`Cannot resolve ${abs}`);
    const physicalParent = await physicalDestination(parent);
    return path.join(physicalParent, path.basename(abs));
  }

  if (stat.isSymbolicLink()) {
    try {
      // A dangling link is not a safe destination: a subsequent write could create its target outside the root.
      return await fs.promises.realpath(abs);
    } catch (e: any) {
      throw new UnsafePathError(`Cannot resolve link destination ${abs} (${e?.code ?? e?.message ?? e})`);
    }
  }
  return realpath(abs);
};

/** Validates one mutation target under one configured root, preserving a deliberately linked root. */
export const resolveMutationPath = async (root: string, candidate: string): Promise<ResolvedMutationPath> => {
  if (typeof root !== 'string' || root.trim() === '') throw new UnsafePathError('The configured root is empty');
  if (typeof candidate !== 'string' || candidate.trim() === '') throw new UnsafePathError('The path is empty');

  const configuredRoot = path.resolve(root);
  const lexicalPath = path.resolve(candidate);
  if (!isContained(configuredRoot, lexicalPath)) {
    throw new UnsafePathError(`Path ${lexicalPath} is outside ${configuredRoot}`);
  }

  const realRoot = await realpath(configuredRoot);
  const realPath = await physicalDestination(lexicalPath);
  if (!isContained(realRoot, realPath)) {
    throw new UnsafePathError(`Path ${lexicalPath} resolves outside ${configuredRoot}`);
  }
  return { configuredRoot, realRoot, lexicalPath, realPath };
};

/** Tries each configured root; lexical non-matches are skipped, and the last unsafe-path reason is returned. */
export const resolveAllowedMutationPath = async (
  roots: Array<string | null | undefined>,
  candidate: string,
): Promise<ResolvedMutationPath> => {
  const lexicalCandidate = path.resolve(candidate);
  let firstUnsafe: UnsafePathError | null = null;
  for (const root of roots) {
    if (typeof root !== 'string' || root.trim() === '') continue;
    const configuredRoot = path.resolve(root);
    if (!isContained(configuredRoot, lexicalCandidate)) continue;
    try {
      return await resolveMutationPath(configuredRoot, candidate);
    } catch (e: any) {
      if (e instanceof UnsafePathError) firstUnsafe ??= e;
      else throw e;
    }
  }
  throw firstUnsafe ?? new UnsafePathError(`Path ${lexicalCandidate} is outside the allowed roots`);
};

// Actual caption UI options are txt/json/caption (Dataset page CreatableSelectInput). Restricting writes to this set
// prevents a caller from replacing arbitrary non-caption files such as `foo.bat` or the source image itself.
export const CAPTION_EXTENSIONS = ['txt', 'json', 'caption'] as const;
export type CaptionExtension = (typeof CAPTION_EXTENSIONS)[number];

export const normalizeCaptionExtension = (ext: unknown): CaptionExtension | null => {
  if (ext === undefined || ext === null || ext === '') return 'txt';
  if (typeof ext !== 'string') return null;
  const normalized = ext.replace(/^\.+/, '').trim().toLowerCase();
  if (normalized === '') return 'txt';
  return (CAPTION_EXTENSIONS as readonly string[]).includes(normalized) ? (normalized as CaptionExtension) : null;
};

export const captionPathFor = (imagePath: string, ext: CaptionExtension): string => {
  const parsed = path.parse(imagePath);
  return path.join(parsed.dir, `${parsed.name}.${ext}`);
};

/** Atomic no-overwrite publication from a staged file to its final name. */
export const publishFileNoClobber = async (staged: string, destination: string): Promise<void> => {
  await fs.promises.link(staged, destination);
};
