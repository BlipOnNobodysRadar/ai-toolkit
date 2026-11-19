import { NextRequest, NextResponse } from 'next/server';
import { Readable } from 'stream';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { readJobReportDownload } from '@/server/jobReports';
import { isUuidV4, PreferenceFeedbackError } from '@/server/preferenceFeedback';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const revalidate = 0;

const noStore = { 'Cache-Control': 'no-store, max-age=0' };

// GET /api/jobs/:id/reports/:report
// The report segment is one of two exact fixed names. No caller-supplied filesystem path is accepted.
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ jobID: string; report: string }> },
) {
  const { jobID, report } = await params;
  if (!isUuidV4(jobID)) {
    return NextResponse.json({ error: 'Invalid job id.', code: 'INVALID_JOB_ID' }, { status: 400, headers: noStore });
  }

  try {
    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) return NextResponse.json({ error: 'Job not found.' }, { status: 404, headers: noStore });
    const download = await readJobReportDownload(job, await getTrainingFolder(), report);
    return new NextResponse(Readable.toWeb(download.stream) as unknown as ReadableStream, {
      status: 200,
      headers: {
        ...noStore,
        'Content-Type': download.contentType,
        'Content-Length': String(download.size),
        'Content-Disposition': `attachment; filename="${download.fileName}"`,
        'X-Content-Type-Options': 'nosniff',
      },
    });
  } catch (error) {
    if (error instanceof PreferenceFeedbackError) {
      return NextResponse.json({ error: error.message, code: error.code }, { status: error.status, headers: noStore });
    }
    console.error('Error downloading job report:', error);
    return NextResponse.json({ error: 'Error downloading job report.' }, { status: 500, headers: noStore });
  }
}
