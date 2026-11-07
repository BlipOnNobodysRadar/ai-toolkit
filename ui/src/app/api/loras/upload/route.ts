import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { getModelsPath } from '@/server/settings';
import { cancelUpload, finishUpload, startUpload, validId, writeChunk } from '@/server/uploadSession';

export const dynamic = 'force-dynamic';

// Chunked upload of a .safetensors file into <MODELS_PATH>/loras.
//   POST ?action=start&fileName=x.safetensors&size=N[&overwrite=1]  -> { uploadId, fileName }
//   POST ?action=chunk&uploadId=..&offset=N   (raw body bytes)       -> { received }
//   POST ?action=finish&uploadId=..                                  -> { path, name, size }
//   POST ?action=cancel&uploadId=..                                  -> { ok }
// Name, size and overwrite are fixed at start; chunk/finish/cancel run under the upload's cross-process operation
// lock (server/uploadSession.ts). A retried chunk must carry the same bytes; finish publishes only a fully received
// file, without replacing an existing one unless overwrite was chosen, and a retried finish returns the first result.
// 503 = busy with another request for the same upload: retry.

const PART_DIR = '.uploads';

const err = (msg: string, status = 400) => NextResponse.json({ error: msg }, { status });

const getRoots = async () => {
  const lorasRoot = path.resolve(await getModelsPath(), 'loras');
  const partDir = path.join(lorasRoot, PART_DIR);
  return { lorasRoot, partDir };
};

// Basename only; must be a .safetensors file.
const cleanFileName = (raw: string | null): string | null => {
  if (!raw) return null;
  const name = raw
    .trim()
    .replace(/[\\/]/g, '_')
    .replace(/[^\w.\- ()\[\]]/g, '_');
  if (!name.toLowerCase().endsWith('.safetensors') || name.startsWith('.') || name.length < 13) return null;
  return name;
};

const parseSize = (raw: string | null): number | null => {
  const n = Number(raw);
  return Number.isSafeInteger(n) && n > 0 ? n : null;
};

export async function POST(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const action = sp.get('action');
  const { lorasRoot, partDir } = await getRoots();

  try {
    if (action === 'start') {
      const fileName = cleanFileName(sp.get('fileName'));
      if (!fileName) return err('File must be a .safetensors file');
      const size = parseSize(sp.get('size'));
      if (!size) return err('Invalid file size');
      const out = await startUpload(partDir, lorasRoot, fileName, size, sp.get('overwrite') === '1');
      return NextResponse.json(out.body, { status: out.status });
    }

    const uploadId = sp.get('uploadId');
    if (!validId(uploadId)) return err('Invalid uploadId');

    let out;
    if (action === 'chunk') {
      const offset = Number(sp.get('offset'));
      out = await writeChunk(partDir, uploadId, offset, Buffer.from(await request.arrayBuffer()));
    } else if (action === 'finish') {
      out = await finishUpload(partDir, lorasRoot, uploadId);
    } else if (action === 'cancel') {
      out = await cancelUpload(partDir, uploadId);
    } else {
      return err('Unknown action');
    }
    return NextResponse.json(out.body, { status: out.status });
  } catch (e: any) {
    console.error('LoRA upload error:', e);
    return err(e?.message || 'Upload failed', 500);
  }
}
