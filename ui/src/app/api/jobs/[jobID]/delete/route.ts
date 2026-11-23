import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { deleteJob } from '../../../../../../cron/ownership/store';
import { TOOLKIT_ROOT } from '@/paths';
import { resolvePythonPath } from '../../../../../../cron/pythonPath';
import { makeProcCtl } from '../../../../../../cron/ownership/procctl';
import { checkStopped, isRunSnapshot } from '../../../../../../cron/ownership/recoverStopped';

// Delete = the atomic claim-free -> 'deleting' transition, which reserves the job's canonical folder
// (case-insensitive where the file system is, through junctions), then folder removal, then the row. The whole
// operation runs under the job's crash-released deletion lock (ownership/store.ts deleteJob): an interrupted or failed
// deletion is resumed by the next Delete, and a Delete while another is running is refused (409).
export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const out = await deleteJob(prisma, jobID, await getTrainingFolder());
  return NextResponse.json(out.body, { status: out.status });
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const seen = await request.json().catch(() => null);
  if (!isRunSnapshot(seen)) {
    return NextResponse.json({ error: 'A current job snapshot is required.' }, { status: 400 });
  }
  const { jobID } = await params;
  const check = await checkStopped(prisma, makeProcCtl(resolvePythonPath(), TOOLKIT_ROOT), jobID, seen);
  const out = check.status === 200 ? await deleteJob(prisma, jobID, await getTrainingFolder(), { seen }) : check;
  return NextResponse.json(out.body, { status: out.status });
}
