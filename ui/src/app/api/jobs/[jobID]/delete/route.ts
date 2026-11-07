import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { deleteJob } from '../../../../../../cron/ownership/store';

// Delete = the atomic claim-free -> 'deleting' transition, which reserves the job's canonical folder
// (case-insensitive where the file system is, through junctions), then folder removal, then the row. The whole
// operation runs under the job's crash-released deletion lock (ownership/store.ts deleteJob): an interrupted or failed
// deletion is resumed by the next Delete, and a Delete while another is running is refused (409).
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const out = await deleteJob(prisma, jobID, await getTrainingFolder());
  return NextResponse.json(out.body, { status: out.status });
}
