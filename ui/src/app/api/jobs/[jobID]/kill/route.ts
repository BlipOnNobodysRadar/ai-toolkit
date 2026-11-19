import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { TOOLKIT_ROOT } from '@/paths';
import { resolvePythonPath } from '../../../../../../cron/pythonPath';
import { makeProcCtl } from '../../../../../../cron/ownership/procctl';
import { forceStop } from '../../../../../../cron/ownership/forceStop';

// Force stop: terminates the run's process tree through handles that are checked against the recorded process
// identity, so a pid that was reused by another program is never touched. Legacy runs (no recorded identity) and
// platforms without such handles are refused explicitly; a failed or partial termination keeps the job's GPUs
// reserved and reports what is still running.
export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const out = await forceStop(prisma, makeProcCtl(resolvePythonPath(), TOOLKIT_ROOT), jobID);
  return NextResponse.json(out.body, { status: out.status });
}
