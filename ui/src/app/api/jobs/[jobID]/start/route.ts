import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { requestStart } from '../../../../../../cron/ownership/store';

// Queue the job. Refused (409) while it is starting/running/stopping/being deleted, or while the previous run's
// process has not been verified as exited even if Python already reported an ending. A repeated Start of a
// queued job is a no-op.
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const out = await requestStart(prisma, jobID, await getTrainingFolder());
  return NextResponse.json(out.body, { status: out.status });
}
