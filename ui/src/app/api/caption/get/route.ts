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
    // Client aborted the request before body was fully sent
    return new NextResponse(null, { status: 499 });
  }

  if (request.signal.aborted) {
    return new NextResponse(null, { status: 499 });
  }

  const extension = readCaptionExtension(body?.ext);
  if (!extension) return new NextResponse('Invalid caption extension', { status: 400 });
  try {
    return new NextResponse(await readCaption(await getDatasetsRoot(), body?.imgPath, extension), {
      headers: { 'Cache-Control': 'private, no-store' },
    });
  } catch (error) {
    if (error instanceof UnsafePathError) return new NextResponse('Access denied', { status: 403 });
    return new NextResponse('Error getting caption', { status: 500 });
  }
}
