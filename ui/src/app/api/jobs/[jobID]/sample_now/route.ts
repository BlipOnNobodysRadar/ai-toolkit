import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await prisma.job.update({
    where: { id: jobID },
    data: {
      sample_now: true,
    },
  });


  return NextResponse.json(job);
}
