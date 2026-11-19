import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import {
  PreferenceFeedbackError,
  isUuidV4,
  isPreferenceID,
  readPreferenceImage,
} from '@/server/preferenceFeedback';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

const noStore = { 'Cache-Control': 'no-store, max-age=0' };

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ jobID: string; preferenceID: string; arm: string }> },
) {
  const { jobID, preferenceID, arm } = await params;
  if (!isUuidV4(jobID)) {
    return NextResponse.json({ error: 'Invalid job id.' }, { status: 400, headers: noStore });
  }
  if (!isPreferenceID(preferenceID)) {
    return NextResponse.json({ error: 'Invalid preference id.' }, { status: 400, headers: noStore });
  }
  if (arm !== 'a' && arm !== 'b') {
    return NextResponse.json({ error: 'Invalid image arm.' }, { status: 400, headers: noStore });
  }

  try {
    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) {
      return NextResponse.json({ error: 'Job not found.' }, { status: 404, headers: noStore });
    }
    const image = await readPreferenceImage(job, await getTrainingFolder(), preferenceID, arm);
    return new NextResponse(image.bytes as any, {
      status: 200,
      headers: {
        ...noStore,
        'Content-Type': image.contentType,
        'Content-Length': String(image.bytes.length),
        'X-Content-Type-Options': 'nosniff',
      },
    });
  } catch (error) {
    if (error instanceof PreferenceFeedbackError) {
      return NextResponse.json({ error: error.message, code: error.code }, { status: error.status, headers: noStore });
    }
    console.error('Failed to read preference image:', error);
    return NextResponse.json({ error: 'Failed to read preference image.' }, { status: 500, headers: noStore });
  }
}
