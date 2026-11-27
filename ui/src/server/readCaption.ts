import fs from 'fs';
import path from 'path';
import { resolveMutationPath, UnsafePathError } from './safePath';

export function readCaptionExtension(value: unknown): string | null {
  if (value == null || value === '') return 'txt';
  if (typeof value !== 'string') return null;
  const ext = value.trim().replace(/^\.+/, '');
  return /[\\/\x00:]/.test(ext) ? null : ext || 'txt';
}

export async function readCaption(root: string, image: string, ext: string): Promise<string> {
  if (typeof image !== 'string' || !image) throw new UnsafePathError('Invalid image path');
  await resolveMutationPath(root, image);
  const { dir, name } = path.parse(image);
  const caption = await resolveMutationPath(root, path.join(dir, name + '.' + ext));
  try {
    return await fs.promises.readFile(caption.realPath, 'utf8');
  } catch (error: any) {
    if (error.code === 'ENOENT') return '';
    throw error;
  }
}
