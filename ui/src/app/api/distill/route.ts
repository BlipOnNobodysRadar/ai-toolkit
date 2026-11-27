import fs from 'node:fs/promises';
import path from 'node:path';
import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { isUuidV4, resolveBoundJobRoot, PreferenceFeedbackError } from '@/server/preferenceFeedback';

export const dynamic = 'force-dynamic';

// Read only the completion manifest or one generated preview from a bound distillation job.
export async function GET(request: Request) {
  const q = new URL(request.url).searchParams;
  const id = q.get('id') || '';
  const item = q.get('item');
  if (!isUuidV4(id) || (item !== null && !/^p\d{5,}-s\d{5,}$/.test(item))) {
    return NextResponse.json({ error: 'Invalid preview request' }, { status: 400 });
  }
  try {
    const job = await prisma.job.findUnique({ where: { id } });
    if (!job || job.job_type !== 'moss_distill') return NextResponse.json({ error: 'Unknown job' }, { status: 404 });
    const bound = await resolveBoundJobRoot(job, await getTrainingFolder());
    const cache = path.join(bound.real, 'teacher-cache');
    const f = path.join(cache, item === null ? 'ready.json' : item + '.png');
    if (await fs.realpath(f) !== f || !(await fs.lstat(f)).isFile()) {
      return NextResponse.json({ error: 'Not a regular cache file' }, { status: 403 });
    }
    return new Response(new Uint8Array(await fs.readFile(f)), { headers: {
      'Content-Type': item === null ? 'application/json' : 'image/png',
      'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
    } });
  } catch (e) {
    const status = e instanceof PreferenceFeedbackError ? e.status : (e as NodeJS.ErrnoException).code === 'ENOENT' ? 404 : 500;
    return NextResponse.json({ error: 'Finished teacher cache unavailable' }, { status });
  }
}
