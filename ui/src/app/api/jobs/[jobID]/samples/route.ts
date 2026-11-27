import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { jobArtifactPath } from '../../../../../../cron/ownership/jobArtifacts';
import { getTrainingFolder } from '@/server/settings';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // setup the training
  const trainingFolder = await getTrainingFolder();

  const samplesFolder = await jobArtifactPath(trainingFolder, job, 'samples');
  try {
    await fs.promises.access(samplesFolder);
  } catch {
    return NextResponse.json({ samples: [] });
  }

  // find all img (png, jpg, jpeg) files in the samples folder. Thumbnails
  // live in the hidden .thumbs subfolder (and partial writes in .tmp) — the
  // isFile check keeps those directories out even if their names ever match.
  const names = (await fs.promises.readdir(samplesFolder, { withFileTypes: true }))
    .filter(entry => entry.isFile())
    .map(entry => entry.name);
  const isMedia = (file: string) => /\.(png|jpe?g|gif|webp|bmp|mp4|avi|mov|mkv|wmv|m4v|flv|mp3|wav|flac|ogg)$/i.test(file);
  const mediaStems = new Set(names.filter(isMedia).map(file => file.slice(0, file.lastIndexOf('.'))));
  const samples = names
    .filter(file => {
      if (isMedia(file)) return true;
      // text samples (LLM models); a .txt next to a media sample is its prompt sidecar, not a sample
      return file.toLowerCase().endsWith('.txt') && !mediaStems.has(file.slice(0, -4));
    })
    .map(file => {
      return path.join(samplesFolder, file);
    })
    .sort();

  return NextResponse.json({ samples });
}
