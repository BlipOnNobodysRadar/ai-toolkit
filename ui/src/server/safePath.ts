import fs from 'fs';
import path from 'path';
export * from '../../cron/safePath';

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
