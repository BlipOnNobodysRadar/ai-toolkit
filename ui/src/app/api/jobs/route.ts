import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { isMac } from '@/helpers/basic';
import { cached } from '@/server/apiCache';
import { getTrainingFolder } from '@/server/settings';
import { saveJob } from '../../../../cron/ownership/store';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');
  const only_active = searchParams.get('only_active');

  try {
    if (id) {
      const job = await prisma.job.findUnique({
        where: { id },
      });
      return NextResponse.json(job);
    }
    if (job_ref) {
      const job = await prisma.job.findFirst({
        where: { job_ref },
        orderBy: { updated_at: 'desc' },
      });
      return NextResponse.json(job);
    }

    const where: any = {};
    if (job_type) {
      where.job_type = job_type;
    }
    if (only_active === 'true') {
      where.status = { in: ['starting', 'running', 'queued', 'stopping'] };
      const jobs = await cached(
        'jobs-active',
        () =>
          prisma.job.findMany({
            where,
            orderBy: { created_at: 'desc' },
          }),
        5000,
        { job_type },
      );
      return NextResponse.json({ jobs: jobs });
    }

    const jobs = await prisma.job.findMany({
      where,
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json({ jobs: jobs });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

// Create (no id) or edit. Names are unique by the folder they map to (case-insensitive on Windows/macOS), a
// folder still being deleted cannot be reused, and an edit is refused while the job owns a process: an editable
// status alone is not enough, since Python can report completed/error while its process is still exiting.
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const out = await saveJob(prisma, body, await getTrainingFolder(), isMac() ? 'mps' : undefined);
    return NextResponse.json(out.body, { status: out.status });
  } catch (error: any) {
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error('Failed to save training data', error?.code || error?.name);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
