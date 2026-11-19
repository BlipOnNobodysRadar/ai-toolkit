// Preference-feedback offline-DPO export contract: real route + helper on isolated Prisma and temporary files.
//   CUDA_VISIBLE_DEVICES=-1 node --test --test-concurrency=1 ui/tests/preferenceFeedbackExport.test.cjs
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { randomUUID } = require('crypto');
const { setupDb, makeJob } = require('./harness/fixture.cjs');
const ts = require('./harness/tsrequire.cjs');

class TestResponse {
  constructor(body, init = {}) {
    this._body = body;
    this.status = init.status ?? 200;
    this.headers = new Headers(init.headers || {});
  }
  static json(body, init = {}) {
    return new TestResponse(body, init);
  }
  static next() {
    return { next: true, status: 200 };
  }
  async json() {
    if (typeof this._body === 'string') return JSON.parse(this._body);
    return this._body;
  }
}

ts.mock('next/server', { NextRequest: class {}, NextResponse: TestResponse });

let F;
let exportRoute;
const roots = [];

before(async () => {
  F = await setupDb('preference-export');
  roots.push(F.trainingRoot);
  exportRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/preferences/export/route.ts'));
});

after(async () => {
  await F.db.$disconnect();
  for (const dir of roots.reverse()) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // best-effort temporary fixture cleanup
    }
  }
});

const makeFeedbackJob = async name => {
  const id = randomUUID();
  const folder = path.join(F.trainingRoot, name);
  const config = {
    job: 'extension',
    config: {
      name,
      process: [
        {
          type: 'ui_trainer',
          train: { steps: 100, dpo_beta: 100, dpo_feedback: { every: 10, start_after: 0 } },
        },
      ],
    },
  };
  const job = await makeJob(F.db, {
    id,
    name,
    run_dir: folder,
    job_config: JSON.stringify(config),
  });
  fs.mkdirSync(folder, { recursive: true });
  return job;
};

const writePair = (jobFolder, preferenceID, overrides = {}, bytes = null) => {
  const pairDir = path.join(jobFolder, 'preferences', preferenceID);
  fs.mkdirSync(pairDir, { recursive: true });
  const metadata = {
    version: 1,
    id: preferenceID,
    step: 10,
    prompt: 'a red fox in a snowy forest',
    seeds: [1234, 5678],
    width: 512,
    height: 512,
    reference_id: 'd'.repeat(64),
    created_at: '2026-11-18T12:34:56.000Z',
    ...overrides,
  };
  fs.writeFileSync(path.join(pairDir, 'pair.json'), JSON.stringify(metadata));
  fs.writeFileSync(path.join(pairDir, 'a.png'), bytes?.a ?? Buffer.from([0x89, 0x50, 0x4e, 0x47, 1]));
  fs.writeFileSync(path.join(pairDir, 'b.png'), bytes?.b ?? Buffer.from([0x89, 0x50, 0x4e, 0x47, 2]));
  return pairDir;
};

const voteRecord = (pairID, choice) => ({
  version: 1,
  id: pairID,
  choice,
  voted_at: '2026-11-18T13:00:00.000Z',
});

const writeVote = (pairDir, vote) => {
  fs.writeFileSync(path.join(pairDir, 'vote.json'), JSON.stringify(vote, null, 2) + '\n');
};

const writeDecision = (pairDir, vote) => {
  const decisionDir = path.join(pairDir, '.decision');
  fs.mkdirSync(decisionDir, { recursive: true });
  fs.writeFileSync(path.join(decisionDir, 'vote.json'), JSON.stringify(vote, null, 2) + '\n');
};

const callExport = (jobID, signal = new AbortController().signal) =>
  exportRoute.GET({ signal }, { params: Promise.resolve({ jobID }) });

const collectBody = async body => {
  const chunks = [];
  for await (const chunk of body) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks);
};

// Minimal central-directory ZIP reader for test fixtures. Archiver streams entries with data descriptors, so sizes
// and offsets are read from the central directory rather than local headers.
const readZipEntries = buffer => {
  const eocdSignature = Buffer.from([0x50, 0x4b, 0x05, 0x06]);
  const eocd = buffer.lastIndexOf(eocdSignature);
  assert.notEqual(eocd, -1, 'ZIP end-of-central-directory record is present');
  const count = buffer.readUInt16LE(eocd + 10);
  const centralOffset = buffer.readUInt32LE(eocd + 16);
  const entries = new Map();
  let cursor = centralOffset;
  for (let index = 0; index < count; index += 1) {
    assert.equal(buffer.readUInt32LE(cursor), 0x02014b50, 'central directory signature');
    const method = buffer.readUInt16LE(cursor + 10);
    const compressedSize = buffer.readUInt32LE(cursor + 20);
    const nameLength = buffer.readUInt16LE(cursor + 28);
    const extraLength = buffer.readUInt16LE(cursor + 30);
    const commentLength = buffer.readUInt16LE(cursor + 32);
    const localOffset = buffer.readUInt32LE(cursor + 42);
    const name = buffer.subarray(cursor + 46, cursor + 46 + nameLength).toString('utf8');

    assert.equal(buffer.readUInt32LE(localOffset), 0x04034b50, 'local file header signature');
    const localNameLength = buffer.readUInt16LE(localOffset + 26);
    const localExtraLength = buffer.readUInt16LE(localOffset + 28);
    const dataOffset = localOffset + 30 + localNameLength + localExtraLength;
    const compressed = buffer.subarray(dataOffset, dataOffset + compressedSize);
    if (method === 0) entries.set(name, Buffer.from(compressed));
    else if (method === 8) entries.set(name, zlib.inflateRawSync(compressed));
    else throw new Error(`Unsupported ZIP compression method ${method}`);

    cursor += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
};

test('winner inversion maps A/B choices to chosen/rejected and preserves Unicode multiline prompts', async () => {
  const job = await makeFeedbackJob('export-winner-inversion');
  const idA = randomUUID().replaceAll('-', '');
  const idB = randomUUID().replaceAll('-', '');
  const bytesA = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0xaa, 0x01]);
  const bytesB = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0xbb, 0x02]);
  const promptA = 'first line\nsecond line — 日本語 🌸\r\nliteral trailing spaces  ';
  const promptB = 'α beta\r\nZażółć gęślą jaźń';
  const pairA = writePair(job.run_dir, idA, { prompt: promptA, step: 20 }, { a: bytesA, b: bytesB });
  const pairB = writePair(job.run_dir, idB, { prompt: promptB, step: 10 }, { a: bytesB, b: bytesA });
  writeVote(pairA, voteRecord(idA, 'a'));
  writeVote(pairB, voteRecord(idB, 'b'));

  const response = await callExport(job.id);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('content-type'), 'application/zip');
  assert.equal(response.headers.get('cache-control'), 'no-store, max-age=0');
  assert.equal(response.headers.get('content-disposition'), 'attachment; filename="preference-feedback.zip"');

  const entries = readZipEntries(await collectBody(response._body));
  assert.deepEqual(
    [...entries.keys()].sort(),
    [
      `chosen/${idA}.png`,
      `chosen/${idA}.txt`,
      `rejected/${idA}.png`,
      `chosen/${idB}.png`,
      `chosen/${idB}.txt`,
      `rejected/${idB}.png`,
    ].sort(),
  );
  assert.deepEqual(entries.get(`chosen/${idA}.png`), bytesA, 'choice a puts original a.png in chosen');
  assert.deepEqual(entries.get(`rejected/${idA}.png`), bytesB, 'choice a puts original b.png in rejected');
  assert.deepEqual(entries.get(`chosen/${idB}.png`), bytesA, 'choice b puts original b.png in chosen');
  assert.deepEqual(entries.get(`rejected/${idB}.png`), bytesB, 'choice b puts original a.png in rejected');
  assert.equal(entries.get(`chosen/${idA}.txt`).toString('utf8'), promptA);
  assert.equal(entries.get(`chosen/${idB}.txt`).toString('utf8'), promptB);
  assert.equal([...entries.keys()].some(name => name.endsWith('.json')), false, 'no pair or vote metadata is exported');
});

test('pending, skipped, invalid and malformed pairs are omitted; a committed decision exports without writes', async () => {
  const job = await makeFeedbackJob('export-filtering');
  const ratedID = randomUUID().replaceAll('-', '');
  const pendingID = randomUUID().replaceAll('-', '');
  const skippedID = randomUUID().replaceAll('-', '');
  const invalidVoteID = randomUUID().replaceAll('-', '');
  const malformedMetadataID = randomUUID().replaceAll('-', '');
  const decisionID = randomUUID().replaceAll('-', '');

  const ratedDir = writePair(job.run_dir, ratedID, { prompt: 'rated prompt' });
  writeVote(ratedDir, voteRecord(ratedID, 'b'));

  writePair(job.run_dir, pendingID, { prompt: 'pending prompt' });

  const skippedDir = writePair(job.run_dir, skippedID, { prompt: 'skipped prompt' });
  writeVote(skippedDir, voteRecord(skippedID, 'skip'));

  const invalidVoteDir = writePair(job.run_dir, invalidVoteID, { prompt: 'invalid prompt' });
  fs.writeFileSync(path.join(invalidVoteDir, 'vote.json'), '{not valid json');

  const malformedDir = writePair(job.run_dir, malformedMetadataID, { prompt: 'bad metadata', version: 999 });
  writeVote(malformedDir, voteRecord(malformedMetadataID, 'a'));

  const decisionDir = writePair(job.run_dir, decisionID, { prompt: 'decision-only prompt' });
  writeDecision(decisionDir, voteRecord(decisionID, 'a'));
  fs.mkdirSync(path.join(job.run_dir, 'preferences', 'not-a-pair'), { recursive: true });

  const response = await callExport(job.id);
  assert.equal(response.status, 200);
  const entries = readZipEntries(await collectBody(response._body));
  assert.deepEqual(
    [...entries.keys()].sort(),
    [
      `chosen/${ratedID}.png`,
      `chosen/${ratedID}.txt`,
      `rejected/${ratedID}.png`,
      `chosen/${decisionID}.png`,
      `chosen/${decisionID}.txt`,
      `rejected/${decisionID}.png`,
    ].sort(),
  );
  assert.equal(entries.get(`chosen/${ratedID}.txt`).toString('utf8'), 'rated prompt');
  assert.equal(entries.get(`chosen/${decisionID}.txt`).toString('utf8'), 'decision-only prompt');
  assert.equal(fs.existsSync(path.join(decisionDir, 'vote.json')), false, 'export must not project a vote receipt');
});

test('zero eligible pairs returns a clear JSON empty-state response instead of an empty ZIP', async () => {
  const job = await makeFeedbackJob('export-empty');
  const pendingID = randomUUID().replaceAll('-', '');
  const skippedID = randomUUID().replaceAll('-', '');
  writePair(job.run_dir, pendingID);
  const skippedDir = writePair(job.run_dir, skippedID);
  writeVote(skippedDir, voteRecord(skippedID, 'skip'));

  const response = await callExport(job.id);
  assert.equal(response.status, 404);
  assert.equal(response.headers.get('cache-control'), 'no-store, max-age=0');
  const body = await response.json();
  assert.equal(body.code, 'NO_RATED_PREFERENCE_PAIRS');
  assert.match(body.error, /No rated preference pairs are available to export/i);
});

test('export source read failures reject the response stream rather than finishing a partial ZIP', async () => {
  const job = await makeFeedbackJob('export-read-error');
  const pairID = randomUUID().replaceAll('-', '');
  const pairDir = writePair(job.run_dir, pairID);
  writeVote(pairDir, voteRecord(pairID, 'a'));

  const originalCreateReadStream = fs.createReadStream;
  let first = true;
  fs.createReadStream = function patchedCreateReadStream(...args) {
    if (first) {
      first = false;
      const { PassThrough } = require('stream');
      const source = new PassThrough();
      process.nextTick(() => source.destroy(new Error('fixture image read failure')));
      return source;
    }
    return originalCreateReadStream.apply(this, args);
  };
  try {
    const response = await callExport(job.id);
    assert.equal(response.status, 200);
    await assert.rejects(() => collectBody(response._body), /fixture image read failure/);
  } finally {
    fs.createReadStream = originalCreateReadStream;
  }
});

test('disconnecting during the first image closes it without opening the remaining images', async () => {
  const job = await makeFeedbackJob('export-cancel');
  for (let index = 0; index < 4; ++index) {
    const id = randomUUID().replaceAll('-', '');
    writeVote(writePair(job.run_dir, id), voteRecord(id, 'a'));
  }
  const controller = new AbortController();
  const { PassThrough } = require('node:stream');
  const { once } = require('node:events');
  const stalled = new PassThrough();
  const closed = once(stalled, 'close');
  let started;
  const opened = new Promise(resolve => { started = resolve; });
  let openCount = 0;
  const original = fs.createReadStream;
  fs.createReadStream = () => {
    openCount++;
    started();
    return stalled;
  };
  try {
    const response = await callExport(job.id, controller.signal);
    await opened;
    const cancelled = assert.rejects(() => collectBody(response._body), /cancel|abort|stream was destroyed/i);
    controller.abort();
    await Promise.all([closed, cancelled]);
    assert.equal(openCount, 1);
  } finally {
    fs.createReadStream = original;
  }
});

test('existing bearer middleware protects the export route', () => {
  const middleware = require(ts.uiPath('src/middleware.ts')).middleware;
  const request = (token) => ({
    method: 'GET',
    headers: new Headers(token ? { Authorization: `Bearer ${token}` } : {}),
    nextUrl: { pathname: `/api/jobs/${randomUUID()}/preferences/export` },
  });
  const old = process.env.AI_TOOLKIT_AUTH;
  try {
    process.env.AI_TOOLKIT_AUTH = 'fixture-only-token';
    assert.equal(middleware(request()).status, 401);
    assert.equal(middleware(request('fixture-only-token')).next, true);
  } finally {
    if (old === undefined) delete process.env.AI_TOOLKIT_AUTH;
    else process.env.AI_TOOLKIT_AUTH = old;
  }
});
