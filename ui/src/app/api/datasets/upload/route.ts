// src/app/api/datasets/upload/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir, rm, stat, unlink } from 'fs/promises';
import { randomUUID } from 'crypto';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { publishFileNoClobber, resolveMutationPath } from '@/server/safePath';

const sanitizeFileName = (name: unknown): string | null => {
  if (typeof name !== 'string') return null;
  const sanitized = name.replace(/[^a-zA-Z0-9.-]/g, '_');
  // Preserve the existing sanitized display/name contract, but never let it collapse to a directory alias.
  if (sanitized === '' || sanitized === '.' || sanitized === '..') return null;
  return sanitized;
};

const statIdentity = async (file: string) => {
  const s = await stat(file, { bigint: true });
  return `${s.dev}:${s.ino}`;
};

const rollbackPublished = async (published: Array<{ path: string; identity: string }>) => {
  for (const entry of [...published].reverse()) {
    try {
      if ((await statIdentity(entry.path)) === entry.identity) await unlink(entry.path);
    } catch {
      // A concurrent change means this file is no longer provably ours; leave it rather than deleting someone else's.
    }
  }
};

export async function POST(request: NextRequest) {
  let stagingDir: string | null = null;
  try {
    const datasetsPath = await getDatasetsRoot();
    if (!datasetsPath) {
      return NextResponse.json({ error: 'Datasets path not found' }, { status: 500 });
    }
    const formData = await request.formData();
    const files = formData.getAll('files');
    const datasetName = formData.get('datasetName') as string;

    if (!files || files.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }

    if (typeof datasetName !== 'string' || datasetName.trim() === '') {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
    }
    const resolvedDatasetsRoot = path.resolve(datasetsPath);
    const uploadDir = path.resolve(resolvedDatasetsRoot, datasetName);
    // Must resolve to a direct child of the datasets root; rejects "..", absolute paths, and the root itself.
    if (path.dirname(uploadDir) !== resolvedDatasetsRoot || uploadDir === resolvedDatasetsRoot) {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
    }

    const fileNames: string[] = [];
    for (const raw of files) {
      const fileName = sanitizeFileName((raw as any)?.name);
      if (!fileName) {
        return NextResponse.json({ error: 'Invalid uploaded file name' }, { status: 400 });
      }
      fileNames.push(fileName);
    }

    // Reject collisions before creating or publishing anything. This is the normal path for `a b.png` vs `a_b.png`
    // and for two files whose non-ASCII names both sanitize to `_.png`.
    const duplicateNames = [...new Set(fileNames.filter((name, i) => fileNames.indexOf(name) !== i))];
    if (duplicateNames.length > 0) {
      return NextResponse.json(
        {
          error: `Uploaded file names collide after sanitization: ${duplicateNames.join(', ')}. Rename them and retry.`,
          conflicts: duplicateNames,
        },
        { status: 409 },
      );
    }

    // Validate the dataset folder and every final destination physically before any write. An intentionally linked
    // datasets root is allowed; a nested dataset-folder link leaving it is not.
    let uploadTarget;
    try {
      uploadTarget = await resolveMutationPath(resolvedDatasetsRoot, uploadDir);
      for (const fileName of fileNames) {
        await resolveMutationPath(resolvedDatasetsRoot, path.join(uploadTarget.lexicalPath, fileName));
      }
    } catch {
      return NextResponse.json({ error: 'Invalid dataset destination' }, { status: 400 });
    }

    // Also refuse an existing destination before staging. Publication below repeats this atomically for races.
    for (const fileName of fileNames) {
      try {
        await stat(path.join(uploadTarget.lexicalPath, fileName));
        return NextResponse.json(
          { error: `A file named "${fileName}" already exists in this dataset`, exists: true, conflicts: [fileName] },
          { status: 409 },
        );
      } catch (e: any) {
        if (e?.code !== 'ENOENT') throw e;
      }
    }

    await mkdir(uploadTarget.lexicalPath, { recursive: true });
    stagingDir = path.join(uploadTarget.lexicalPath, `.upload-${randomUUID()}`);
    await mkdir(stagingDir);
    const staged: Array<{ stagingPath: string; finalPath: string; name: string }> = [];

    // Process files sequentially to avoid overwhelming the system; a byte buffer is written to staging before any
    // destination name becomes visible.
    for (let i = 0; i < files.length; i++) {
      const file = files[i] as any;
      const bytes = await file.arrayBuffer();
      const stagingPath = path.join(stagingDir, String(i));
      await writeFile(stagingPath, Buffer.from(bytes));
      staged.push({ stagingPath, finalPath: path.join(uploadTarget.lexicalPath, fileNames[i]), name: fileNames[i] });
    }

    const published: Array<{ path: string; identity: string }> = [];
    try {
      for (const item of staged) {
        await publishFileNoClobber(item.stagingPath, item.finalPath);
        published.push({ path: item.finalPath, identity: await statIdentity(item.finalPath) });
      }
    } catch (e: any) {
      await rollbackPublished(published);
      if (e?.code === 'EEXIST') {
        return NextResponse.json(
          { error: 'A file with the same sanitized name was published concurrently; retry or rename it.', exists: true },
          { status: 409 },
        );
      }
      const unsupported = ['EPERM', 'ENOTSUP', 'EXDEV', 'EINVAL', 'ENOSYS', 'EOPNOTSUPP'].includes(e?.code);
      return NextResponse.json(
        {
          error: unsupported
            ? 'This file system cannot publish uploads without overwriting; no files were published.'
            : 'Uploaded files could not be published; no files were overwritten.',
        },
        { status: 500 },
      );
    }
    return NextResponse.json({ message: 'Files uploaded successfully', files: fileNames });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json({ error: 'Error uploading files' }, { status: 500 });
  } finally {
    if (stagingDir) {
      try {
        await rm(stagingDir, { recursive: true, force: true });
      } catch (e) {
        console.error('Upload staging cleanup error:', e);
      }
    }
  }
}

// Increase payload size limit (default is 4mb)
export const config = {
  api: {
    bodyParser: false,
    responseLimit: '50mb',
  },
};

