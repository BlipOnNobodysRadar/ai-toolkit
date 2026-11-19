import { NextResponse } from 'next/server';
import fs from 'fs';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { captionPathFor, resolveAllowedMutationPath } from '@/server/safePath';

const fileExists = (p: string) => fs.promises.access(p).then(() => true).catch(() => false);
const MEDIA_RE = /\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|flac|ogg)$/i;

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath } = body;
    if (typeof imgPath !== 'string' || imgPath.trim() === '') {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();
    const allowedRoots = [datasetsPath, trainingPath];

    // Validate both final destinations before mutating either. The physical check permits an intentionally linked
    // configured root but rejects a nested link that escapes it. (A concurrent link swap after this check is a
    // residual boundary; see server/safePath.ts.)
    let imageTarget;
    let captionTarget;
    try {
      imageTarget = await resolveAllowedMutationPath(allowedRoots, imgPath);
      captionTarget = await resolveAllowedMutationPath(allowedRoots, captionPathFor(imageTarget.lexicalPath, 'txt'));
    } catch {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }
    if (imageTarget.lexicalPath === captionTarget.lexicalPath) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // make sure it is a media file
    if (!MEDIA_RE.test(imageTarget.lexicalPath)) {
      return NextResponse.json({ error: 'Not an image' }, { status: 400 });
    }

    // if img doesn't exist, ignore (the route is intentionally idempotent for a missing valid media path)
    if (!(await fileExists(imageTarget.lexicalPath))) {
      return NextResponse.json({ success: true });
    }

    await fs.promises.unlink(imageTarget.lexicalPath);

    // check for caption
    if (await fileExists(captionTarget.lexicalPath)) {
      await fs.promises.unlink(captionTarget.lexicalPath);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
