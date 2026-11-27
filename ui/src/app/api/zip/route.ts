/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import archiver from 'archiver';
import { resolveMutationPath, UnsafePathError } from '@/server/safePath';
import prisma from '@/server/prisma';
import { jobArtifactPath } from '../../../../cron/ownership/jobArtifacts';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs'; // ensure Node APIs are available
export const dynamic = 'force-dynamic'; // long-running, non-cached

type PostBody = {
  zipTarget: 'samples' | 'dataset' | 'dataset_captions';
  jobName?: string; // required for 'samples'
  datasetName?: string; // required for dataset targets
};

const CAPTION_EXTS = ['.txt', '.json', '.caption'];

async function resolveSafe(p: string) {
  // resolve symlinks + normalize
  return await fsp.realpath(p);
}

/**
 * Recursively collects files to zip, skipping hidden entries (.thumbs etc).
 * When captionsOnly is set, only caption files (.txt/.json/.caption) are kept.
 */
async function collectFiles(dir: string, captionsOnly: boolean): Promise<string[]> {
  const entries = await fsp.readdir(dir, { withFileTypes: true });
  let out: string[] = [];
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const itemPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out = out.concat(await collectFiles(itemPath, captionsOnly));
    } else if (entry.isFile()) {
      if (!captionsOnly || CAPTION_EXTS.includes(path.extname(entry.name).toLowerCase())) {
        out.push(itemPath);
      }
    }
  }
  return out;
}

async function zipDataset(datasetName: string, captionsOnly: boolean) {
  const datasetsRoot = await resolveSafe(await getDatasetsRoot());
  if (typeof datasetName !== 'string' || !datasetName || datasetName.startsWith('.') || path.basename(datasetName) !== datasetName) {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }
  const safeName = datasetName;
  let folderPath: string;
  try {
    folderPath = (await resolveMutationPath(datasetsRoot, path.join(datasetsRoot, safeName))).realPath;
    if (!(await fsp.stat(folderPath)).isDirectory()) throw new Error();
  } catch (error) {
    if (error instanceof UnsafePathError) return new NextResponse('Access denied', { status: 403 });
    return NextResponse.json({ error: `Dataset '${datasetName}' not found` }, { status: 404 });
  }

  const files = await collectFiles(folderPath, captionsOnly);
  if (files.length === 0) {
    return NextResponse.json({ error: captionsOnly ? 'No captions found' : 'Dataset is empty' }, { status: 404 });
  }

  // Hidden dir at the datasets root: skipped by the dataset list/image walkers,
  // but still under datasetsRoot so /api/files/[...filePath] is allowed to serve it.
  const zipDir = (await resolveMutationPath(datasetsRoot, path.join(datasetsRoot, '.zips'))).realPath;
  await fsp.mkdir(zipDir, { recursive: true });
  const outputPath = path.join(zipDir, `${safeName}_${captionsOnly ? 'captions' : 'dataset'}.zip`);
  await fsp.rm(outputPath, { force: true });

  await new Promise<void>((resolve, reject) => {
    const output = fs.createWriteStream(outputPath);
    const archive = archiver('zip', { zlib: { level: 9 } });

    output.on('close', () => resolve());
    output.on('error', reject);
    archive.on('error', reject);

    archive.pipe(output);

    for (const file of files) {
      archive.file(file, { name: path.join(safeName, path.relative(folderPath, file)) });
    }

    archive.finalize().catch(reject);
  });

  return NextResponse.json({
    ok: true,
    zipPath: outputPath,
    fileName: path.basename(outputPath),
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as PostBody;

    if (body?.zipTarget === 'dataset' || body?.zipTarget === 'dataset_captions') {
      if (!body.datasetName) {
        return NextResponse.json({ error: 'datasetName is required' }, { status: 400 });
      }
      return await zipDataset(body.datasetName, body.zipTarget === 'dataset_captions');
    }

    if (!body || !body.jobName) {
      return NextResponse.json({ error: 'jobName is required' }, { status: 400 });
    }

    const job = await prisma.job.findUnique({ where: { name: body.jobName } });
    if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    const trainingRoot = await getTrainingFolder();
    const folderPath = await jobArtifactPath(trainingRoot, job, 'samples');
    const outputPath = await jobArtifactPath(trainingRoot, job, 'samples.zip');

    // Must be a directory
    let stat: fs.Stats;
    try {
      stat = await fsp.stat(folderPath);
    } catch {
      return new NextResponse('Folder not found', { status: 404 });
    }
    if (!stat.isDirectory()) {
      return new NextResponse('Not a directory', { status: 400 });
    }

    // delete current one if it exists (force:true => no error if missing)
    await fsp.rm(outputPath, { force: true });

    // Create write stream & archive
    await new Promise<void>((resolve, reject) => {
      const output = fs.createWriteStream(outputPath);
      const archive = archiver('zip', { zlib: { level: 9 } });

      output.on('close', () => resolve());
      output.on('error', reject);
      archive.on('error', reject);

      archive.pipe(output);

      // Add the directory contents (place them under the folder's base name in the zip)
      const rootName = path.basename(folderPath);
      archive.directory(folderPath, rootName);

      archive.finalize().catch(reject);
    });

    // Return the absolute path so your existing /api/files/[...filePath] can serve it
    // Example download URL (client-side): `/api/files/${encodeURIComponent(resolvedOutPath)}`
    return NextResponse.json({
      ok: true,
      zipPath: outputPath,
      fileName: path.basename(outputPath),
    });
  } catch (err) {
    console.error('Zip error:', err);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
