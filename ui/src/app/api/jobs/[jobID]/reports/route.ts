import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { readJobReports, resolveExistingJobFolder } from '@/server/jobReports';

export const runtime = 'nodejs';

// GET /api/jobs/:id/reports
// Read-only summaries of training_extras/validation.jsonl and edm2-weighting.csv.
// The only paths read are fixed names inside this job's own training folder.
export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  try {
    const trainingFolder = await getTrainingFolder();
    const jobFolder = await resolveExistingJobFolder(trainingFolder, job.name);
    return NextResponse.json(await readJobReports(jobFolder));
  } catch (error: any) {
    if (error?.message === 'Job folder is outside the training root') {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    console.error('Error reading job reports:', error);
    return NextResponse.json({ error: 'Error reading job reports' }, { status: 500 });
  }
}
