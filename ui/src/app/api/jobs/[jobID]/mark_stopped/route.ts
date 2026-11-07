import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { markStopped } from '../../../../../../cron/ownership/store';

// Display reset for a row whose process is already gone. Refused (409) while the job owns a process (including a
// run that reported an ending but has not been verified as exited, and pre-update runs with a recorded pid).
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const out = await markStopped(prisma, jobID);
  if (out.status === 200) console.log(`Job ${jobID} marked as stopped`);
  return NextResponse.json(out.body, { status: out.status });
}
