import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import {
  PreferenceFeedbackError,
  castPreferenceVote,
  isUuidV4,
  isPreferenceID,
} from '@/server/preferenceFeedback';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

const noStore = { 'Cache-Control': 'no-store, max-age=0' };

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ jobID: string; preferenceID: string }> | { jobID: string; preferenceID: string } },
) {
  const { jobID, preferenceID } = await params;
  if (!isUuidV4(jobID)) {
    return NextResponse.json({ error: 'Invalid job id.' }, { status: 400, headers: noStore });
  }
  if (!isPreferenceID(preferenceID)) {
    return NextResponse.json({ error: 'Invalid preference id.' }, { status: 400, headers: noStore });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'A JSON body is required.' }, { status: 400, headers: noStore });
  }
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    return NextResponse.json({ error: 'A JSON object is required.' }, { status: 400, headers: noStore });
  }
  const choice = (body as { choice?: unknown }).choice;
  if (choice !== 'a' && choice !== 'b' && choice !== 'skip') {
    return NextResponse.json({ error: 'choice must be "a", "b" or "skip".' }, { status: 400, headers: noStore });
  }

  try {
    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) {
      return NextResponse.json({ error: 'Job not found.' }, { status: 404, headers: noStore });
    }
    const result = await castPreferenceVote(job, await getTrainingFolder(), preferenceID, choice);
    return NextResponse.json(result, { status: 200, headers: noStore });
  } catch (error) {
    if (error instanceof PreferenceFeedbackError) {
      return NextResponse.json({ error: error.message, code: error.code }, { status: error.status, headers: noStore });
    }
    console.error('Failed to record preference vote:', error);
    return NextResponse.json({ error: 'Failed to record preference vote.' }, { status: 500, headers: noStore });
  }
}
