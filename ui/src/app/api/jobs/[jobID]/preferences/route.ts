import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import {
  PreferenceFeedbackError,
  isUuidV4,
  listPreferenceFeedback,
} from '@/server/preferenceFeedback';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

const noStore = { 'Cache-Control': 'no-store, max-age=0' };

export async function GET(_request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  if (!isUuidV4(jobID)) {
    return NextResponse.json({ error: 'Invalid job id.' }, { status: 400, headers: noStore });
  }

  try {
    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) {
      return NextResponse.json({ error: 'Job not found.' }, { status: 404, headers: noStore });
    }
    const feedback = await listPreferenceFeedback(job, await getTrainingFolder());
    return NextResponse.json(feedback, { status: 200, headers: noStore });
  } catch (error) {
    if (error instanceof PreferenceFeedbackError) {
      return NextResponse.json({ error: error.message, code: error.code }, { status: error.status, headers: noStore });
    }
    console.error('Failed to load preference feedback:', error);
    return NextResponse.json({ error: 'Failed to load preference feedback.' }, { status: 500, headers: noStore });
  }
}
