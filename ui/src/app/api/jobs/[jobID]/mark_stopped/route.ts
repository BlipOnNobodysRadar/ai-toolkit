import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { markStopped } from '../../../../../../cron/ownership/store';
import { TOOLKIT_ROOT } from '@/paths';
import { resolvePythonPath } from '../../../../../../cron/pythonPath';
import { makeProcCtl } from '../../../../../../cron/ownership/procctl';
import { checkStopped, isRunSnapshot } from '../../../../../../cron/ownership/recoverStopped';

// The existing GET action applies to unreserved jobs.
export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const out = await markStopped(prisma, jobID);
  return NextResponse.json(out.body, { status: out.status });
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const seen = await request.json().catch(() => null);
  if (!isRunSnapshot(seen)) {
    return NextResponse.json({ error: 'A current job snapshot is required.' }, { status: 400 });
  }
  const { jobID } = await params;
  const check = await checkStopped(prisma, makeProcCtl(resolvePythonPath(), TOOLKIT_ROOT), jobID, seen);
  const out = check.status === 200 ? await markStopped(prisma, jobID, seen) : check;
  return NextResponse.json(out.body, { status: out.status });
}
