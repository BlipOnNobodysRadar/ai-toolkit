import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { requestStop } from '../../../../../../cron/ownership/store';

// Stop is a request on every platform: the job's stop watcher polls the `stop` flag and raises SIGINT inside its
// own process, which runs the normal KeyboardInterrupt shutdown (final DB write, closing log lines). Nothing is
// signalled from here, so a reused pid can never be hit. The run keeps its GPUs and folder until the worker has
// verified that its processes exited; a run still alive long after the request gets an identity-checked force
// stop from the worker (docs/STATE_MACHINES.md M10).
export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const out = await requestStop(prisma, jobID);
  return NextResponse.json(out.body, { status: out.status });
}
