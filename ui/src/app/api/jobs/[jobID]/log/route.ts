import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import type { FileHandle } from 'fs/promises';
import { getTrainingFolder } from '@/server/settings';

// Bytes of buf[0..n) up to the end of the last complete character. Only a trailing sequence that is a valid UTF-8
// prefix still waiting for its continuation bytes (valid lead, and a second byte in the lead's allowed range) is
// held back (at most 3 bytes; the writer appends the rest);
// malformed bytes pass through and decode as U+FFFD as before. A log that ends for good inside a character keeps
// those last bytes unshown.
const completeUtf8Length = (buf: Buffer, n: number): number => {
  let i = n - 1;
  let continuation = 0;
  while (i >= 0 && continuation < 3 && (buf[i] & 0xc0) === 0x80) {
    i -= 1;
    continuation += 1;
  }
  if (i < 0) return n;
  const lead = buf[i];
  const need = lead >= 0xc2 && lead <= 0xdf ? 2 : lead >= 0xe0 && lead <= 0xef ? 3 : lead >= 0xf0 && lead <= 0xf4 ? 4 : 1;
  if (need === 1 || n - i >= need) return n;
  if (n - i >= 2) {
    // the second byte's range is restricted for some leads (RFC 3629): E0 overlong, ED surrogates, F0 overlong,
    // F4 above U+10FFFF. Such a tail is already malformed, so it is decoded (U+FFFD) now rather than withheld.
    const second = buf[i + 1];
    const [lo, hi] =
      lead === 0xe0 ? [0xa0, 0xbf] : lead === 0xed ? [0x80, 0x9f] : lead === 0xf0 ? [0x90, 0xbf] : lead === 0xf4 ? [0x80, 0x8f] : [0x80, 0xbf];
    if (second < lo || second > hi) return n;
  }
  return i;
};

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'log.txt');

  try {
    await fs.promises.access(logPath);
  } catch {
    return NextResponse.json({ log: '', offset: 0, reset: true });
  }

  // Cap on the initial payload. The client renders the log through a terminal
  // emulator that collapses \r/cursor-movement rewrites (progress bars), so a
  // small newline-counted tail would be mostly bar churn that collapses to a
  // few rendered lines — send a generous byte tail instead and let the
  // emulator (which caps its own scrollback) do the trimming.
  const MAX_TAIL_BYTES = 5 * 1024 * 1024;
  // Client sends the byte offset it has already consumed so we only return new
  // content. `offset` omitted (or NaN) => initial load / full tail.
  const offsetParam = request.nextUrl.searchParams.get('offset');
  const offset = offsetParam === null ? NaN : parseInt(offsetParam, 10);
  // The cursor's stream identity: which log file the offset counts into. Every launch rotates log.txt, so a new
  // run's log can be larger than the old offset; comparing sizes alone would hand the client the new file's tail as
  // if it continued the old one.
  const generationParam = request.nextUrl.searchParams.get('generation');

  const readRange = async (fh: FileHandle, start: number, end: number): Promise<{ log: string; offset: number }> => {
    const length = end - start;
    if (length <= 0) return { log: '', offset: start };
    const buffer = Buffer.alloc(length);
    const { bytesRead } = await fh.read(buffer, 0, length, start);
    // the cursor never splits a character: a trailing, still-incomplete UTF-8 sequence is left for the next read
    const complete = completeUtf8Length(buffer, bytesRead);
    return { log: buffer.subarray(0, complete).toString('utf-8'), offset: start + complete };
  };

  try {
    const fh = await fs.promises.open(logPath, 'r');
    try {
      // Bind the identity, size and read to one handle: the pathname can be rotated between awaits.
      const stats = await fh.stat({ bigint: true });
      const size = Number(stats.size);
      const generation = `${stats.ino}-${stats.birthtimeNs}`;
      // A different log file (new run), or an offset past the current end (truncated) — load a fresh tail.
      const isReset = Number.isNaN(offset) || offset > size || (generationParam !== null && generationParam !== generation);
      if (isReset) {
        // Read only the tail of very large files to bound memory/payload.
        const start = Math.max(0, size - MAX_TAIL_BYTES);
        const read = await readRange(fh, start, size);
        let log = read.log;
        // Drop a partial first line if we started mid-file.
        if (start > 0) {
          const newlineIdx = log.indexOf('\n');
          if (newlineIdx !== -1) {
            log = log.slice(newlineIdx + 1);
          }
        }
        return NextResponse.json({ log, offset: read.offset, reset: true, generation, run_id: job.run_id });
      }
      // Incremental: return only the bytes appended since the last offset.
      const read = await readRange(fh, offset, size);
      return NextResponse.json({ ...read, reset: false, generation, run_id: job.run_id });
    } finally {
      await fh.close();
    }
  } catch (error) {
    console.error('Error reading log file:', error);
    return NextResponse.json({ log: 'Error reading log file', offset: 0, reset: true });
  }
}
