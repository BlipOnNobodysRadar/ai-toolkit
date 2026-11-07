import { NextResponse } from 'next/server';
import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';
import {
  captionPathFor,
  normalizeCaptionExtension,
  resolveAllowedMutationPath,
} from '@/server/safePath';

const MEDIA_RE = /\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|flac|ogg)$/i;

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, caption, ext } = body;
    if (typeof imgPath !== 'string' || imgPath.trim() === '') {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }
    if (typeof caption !== 'string') {
      return NextResponse.json({ error: 'Caption must be a string' }, { status: 400 });
    }

    const captionExt = normalizeCaptionExtension(ext);
    if (!captionExt) {
      return NextResponse.json(
        { error: 'Captions are limited to .txt, .json, or .caption files' },
        { status: 400 },
      );
    }

    const datasetsPath = await getDatasetsRoot();

    // Validate the image and the final caption destination (not just its stem) before writing. This catches both a
    // caller-selected extension and an existing caption link that resolves outside the dataset root.
    let imageTarget;
    let captionTarget;
    try {
      imageTarget = await resolveAllowedMutationPath([datasetsPath], imgPath);
      captionTarget = await resolveAllowedMutationPath(
        [datasetsPath],
        captionPathFor(imageTarget.lexicalPath, captionExt),
      );
    } catch {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }
    if (imageTarget.lexicalPath === captionTarget.lexicalPath || !MEDIA_RE.test(imageTarget.lexicalPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    let stat;
    try {
      stat = await fs.promises.stat(imageTarget.lexicalPath);
    } catch {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }
    if (!stat.isFile()) {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    // save caption to the validated final destination
    await fs.promises.writeFile(captionTarget.lexicalPath, caption, 'utf8');

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
