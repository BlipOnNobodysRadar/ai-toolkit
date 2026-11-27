/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import { readCaption, readCaptionExtension } from '@/server/readCaption';
import { UnsafePathError } from '@/server/safePath';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return new NextResponse(null, { status: 499 });
  }

  if (request.signal.aborted) {
    return new NextResponse(null, { status: 499 });
  }

  const { imgPaths, ext } = body ?? {};
  if (!Array.isArray(imgPaths)) {
    return NextResponse.json({ error: 'imgPaths must be an array' }, { status: 400 });
  }

  const captionExt = readCaptionExtension(ext);
  if (!captionExt) return NextResponse.json({ error: 'Invalid caption extension' }, { status: 400 });
  const allowedDir = await getDatasetsRoot();
  const captions: Record<string, string> = Object.create(null);
  await Promise.all(imgPaths.map(async (imgPath: unknown) => {
    if (typeof imgPath !== 'string') return;
    try {
      captions[imgPath] = await readCaption(allowedDir, imgPath, captionExt);
    } catch (error) {
      if (!(error instanceof UnsafePathError)) captions[imgPath] = '';
    }
  }));

  return NextResponse.json({ captions }, { headers: { 'Cache-Control': 'private, no-store' } });
}
