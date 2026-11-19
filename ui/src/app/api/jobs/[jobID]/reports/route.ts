import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { readJobReports } from '@/server/jobReports';
import { isUuidV4, PreferenceFeedbackError } from '@/server/preferenceFeedback';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const revalidate = 0;

const noStore = { 'Cache-Control': 'no-store, max-age=0' };

// GET /api/jobs/:id/reports
// Read-only summaries of training_extras/validation.jsonl and edm2-weighting.csv.
// The only paths read are fixed names inside this job's own verified run folder.
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ jobID: string }> },
) {
  const { jobID } = await params;
  if (!isUuidV4(jobID)) {
    return NextResponse.json({ error: 'Invalid job id.', code: 'INVALID_JOB_ID' }, { status: 400, headers: noStore });
  }

  try {
    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) return NextResponse.json({ error: 'Job not found.' }, { status: 404, headers: noStore });
    const reports = await readJobReports(job, await getTrainingFolder());
    return NextResponse.json(reports, { headers: noStore });
  } catch (error) {
    if (error instanceof PreferenceFeedbackError) {
      return NextResponse.json({ error: error.message, code: error.code }, { status: error.status, headers: noStore });
    }
    console.error('Error reading job reports:', error);
    return NextResponse.json({ error: 'Error reading job reports.' }, { status: 500, headers: noStore });
  }
}
