import { NextRequest, NextResponse } from 'next/server';
import { Readable } from 'stream';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import {
  PreferenceFeedbackError,
  createPreferenceFeedbackExport,
  isUuidV4,
} from '@/server/preferenceFeedback';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const revalidate = 0;

const noStore = { 'Cache-Control': 'no-store, max-age=0' };

// GET /api/jobs/:id/preferences/export
// The stored job binding and rated pairs determine the contents of the offline DPO dataset.
export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  if (!isUuidV4(jobID)) {
    return NextResponse.json({ error: 'Invalid job id.', code: 'INVALID_JOB_ID' }, { status: 400, headers: noStore });
  }

  try {
    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) return NextResponse.json({ error: 'Job not found.' }, { status: 404, headers: noStore });

    const download = await createPreferenceFeedbackExport(job, await getTrainingFolder());
    const body = Readable.toWeb(download.archive) as unknown as ReadableStream;
    const signal = request?.signal;
    const abort = () => {
      download.archive.destroy(new Error('Preference export cancelled by client.'));
    };

    if (signal) {
      if (signal.aborted) {
        abort();
      } else {
        signal.addEventListener('abort', abort, { once: true });
        download.archive.once('close', () => signal.removeEventListener('abort', abort));
      }
    }
    download.archive.once('error', (error: Error) => {
      if (!signal?.aborted) console.error('Preference feedback export stream failed:', error);
    });

    return new NextResponse(body, {
      status: 200,
      headers: {
        ...noStore,
        'Content-Type': 'application/zip',
        'Content-Disposition': `attachment; filename="${download.fileName}"`,
        'X-Content-Type-Options': 'nosniff',
      },
    });
  } catch (error) {
    if (error instanceof PreferenceFeedbackError) {
      return NextResponse.json({ error: error.message, code: error.code }, { status: error.status, headers: noStore });
    }
    console.error('Failed to export preference feedback:', error);
    return NextResponse.json({ error: 'Failed to export preference feedback.' }, { status: 500, headers: noStore });
  }
}
