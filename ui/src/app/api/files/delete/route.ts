/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import { historicalArtifactFolder } from '../../../../../cron/ownership/jobArtifacts';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { resolveAllowedMutationPath } from '@/server/safePath';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { filePath } = body;

    if (!filePath || typeof filePath !== 'string') {
      return new NextResponse('filePath is required', { status: 400 });
    }

    // Decode the path
    let decodedFilePath: string;
    try {
      decodedFilePath = path.isAbsolute(filePath) ? filePath : decodeURIComponent(filePath);
    } catch {
      return new NextResponse('Invalid file path', { status: 400 });
    }

    // Get allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();

    // Resolve physically: an intentionally linked configured root is allowed, but a nested link escaping it is not.
    // The unlink immediately below is the residual check-then-act boundary documented in server/safePath.ts.
    let target;
    try {
      const roots = [datasetRoot, trainingRoot];
      const resolved = path.resolve(decodedFilePath);
      if (!roots.some(root => resolved.startsWith(root + path.sep))) {
        const historical = await historicalArtifactFolder(prisma, trainingRoot, resolved);
        if (historical) roots.push(historical);
      }
      target = await resolveAllowedMutationPath(roots, decodedFilePath);
    } catch {
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check if file exists and grab file info in one stat
    let stat;
    try {
      stat = await fs.promises.stat(target.lexicalPath);
    } catch {
      return new NextResponse('File not found', { status: 404 });
    }
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    await fs.promises.unlink(target.lexicalPath);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting file');
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
