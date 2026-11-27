import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { TOOLKIT_ROOT } from '@/paths';
import { resolvePythonPath } from '../../../../../../cron/pythonPath';
import { makeProcCtl } from '../../../../../../cron/ownership/procctl';
import { forceStop } from '../../../../../../cron/ownership/forceStop';

// Force stop: terminates the run's process tree through handles that are checked against the recorded process
// identity. Old launchers identify the trainer and its workers through the job ID in their process environment.
export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const rawForce = request?.nextUrl?.searchParams?.get('force') ?? null;
  const force = rawForce === '1' || rawForce === 'true';
  const out = await forceStop(prisma, makeProcCtl(resolvePythonPath(), TOOLKIT_ROOT), jobID, 5000, { force });
  return NextResponse.json(out.body, { status: out.status });
}
