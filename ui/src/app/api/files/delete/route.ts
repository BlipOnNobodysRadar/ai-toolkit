/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
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
      decodedFilePath = decodeURIComponent(filePath);
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
      target = await resolveAllowedMutationPath([datasetRoot, trainingRoot], decodedFilePath);
    } catch {
      console.warn(`Access denied: ${decodedFilePath} not in ${[datasetRoot, trainingRoot].join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check if file exists and grab file info in one stat
    let stat;
    try {
      stat = await fs.promises.stat(target.lexicalPath);
    } catch {
      console.warn(`File not found: ${target.lexicalPath}`);
      return new NextResponse('File not found', { status: 404 });
    }
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    await fs.promises.unlink(target.lexicalPath);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting file:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
