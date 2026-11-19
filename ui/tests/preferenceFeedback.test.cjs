// Preference feedback contract: real route modules + helper on isolated Prisma and temporary files.
//   CUDA_VISIBLE_DEVICES=-1 node --test --test-concurrency=1 ui/tests/preferenceFeedback.test.cjs
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { randomUUID } = require('crypto');
const { setupDb, makeJob, ts } = require('./harness/fixture.cjs');
const { SCRATCH } = require('./harness/env.cjs');

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
let listRoute;
let voteRoute;
let imageRoute;
const roots = [];

before(async () => {
  F = await setupDb('preference-feedback');
  roots.push(F.trainingRoot);
  listRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/preferences/route.ts'));
  voteRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/preferences/[preferenceID]/vote/route.ts'));
  imageRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/preferences/[preferenceID]/image/[arm]/route.ts'));
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

const callList = jobID => listRoute.GET({}, { params: Promise.resolve({ jobID }) });
const callImage = (jobID, preferenceID, arm) =>
  imageRoute.GET({}, { params: Promise.resolve({ jobID, preferenceID, arm }) });
const callVote = (jobID, preferenceID, body) =>
  voteRoute.POST({ json: async () => body }, { params: Promise.resolve({ jobID, preferenceID }) });
const bodyOf = async res => res.json();

const makeFeedbackJob = async (name, { enabled = true, runDir = null, configOverride = null } = {}) => {
  const id = randomUUID();
  const folder = runDir || path.join(F.trainingRoot, name);
  const config = configOverride || {
    job: 'extension',
    config: {
      name,
      process: [
        {
          type: 'ui_trainer',
          train: {
            steps: 100,
            dpo_beta: 100,
            ...(enabled ? { dpo_feedback: { every: 10, start_after: 0 } } : {}),
          },
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

test('positive contract: enabled feedback lists one exact pair, serves its exact images, and accepts an immutable idempotent vote', async () => {
  const job = await makeFeedbackJob('positive-pairs');
  const pairID = randomUUID().replaceAll('-', '');
  const pairDir = writePair(job.run_dir, pairID);
  const bytesA = fs.readFileSync(path.join(pairDir, 'a.png'));

  const listed = await callList(job.id);
  assert.equal(listed.status, 200);
  assert.equal(listed.headers.get('cache-control'), 'no-store, max-age=0');
  const list = await bodyOf(listed);
  assert.equal(list.enabled, true);
  assert.equal(list.pairs.length, 1);
  assert.deepEqual(list.counts, { pending: 1, rated: 0, skipped: 0, invalid: 0 });
  assert.equal(list.pairs[0].id, pairID);
  assert.equal(list.pairs[0].status, 'pending');
  assert.deepEqual(list.pairs[0].seeds, [1234, 5678]);
  assert.equal(list.pairs[0].image_urls.a, `/api/jobs/${job.id}/preferences/${pairID}/image/a`);
  assert.equal(JSON.stringify(list).includes(F.trainingRoot), false, 'no filesystem root leaks');

  const image = await callImage(job.id, pairID, 'a');
  assert.equal(image.status, 200);
  assert.equal(image.headers.get('content-type'), 'image/png');
  assert.equal(image.headers.get('cache-control'), 'no-store, max-age=0');
  assert.deepEqual(Buffer.from(image._body), bytesA);

  const first = await callVote(job.id, pairID, { choice: 'a' });
  assert.equal(first.status, 200, JSON.stringify(await bodyOf(first)));
  const firstBody = await bodyOf(first);
  assert.equal(firstBody.created, true);
  assert.equal(firstBody.vote.choice, 'a');
  assert.match(firstBody.vote.voted_at, /^\d{4}-\d{2}-\d{2}T.*Z$/);

  const retry = await callVote(job.id, pairID, { choice: 'a' });
  assert.equal(retry.status, 200);
  const retryBody = await bodyOf(retry);
  assert.equal(retryBody.created, false);
  assert.deepEqual(retryBody.vote, firstBody.vote, 'same-choice retry returns the original vote');
  assert.equal(fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8'), JSON.stringify(firstBody.vote, null, 2) + '\n');

  const conflict = await callVote(job.id, pairID, { choice: 'b' });
  assert.equal(conflict.status, 409);
  assert.match((await bodyOf(conflict)).error, /different immutable vote/);
  assert.equal(JSON.parse(fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8')).choice, 'a');

  const after = await bodyOf(await callList(job.id));
  assert.deepEqual(after.counts, { pending: 0, rated: 1, skipped: 0, invalid: 0 });
  assert.equal(after.pairs[0].status, 'rated');
  assert.equal(after.pairs[0].vote.choice, 'a');
});

test('skip is an immutable decision and disabled feedback with a published pair still opens the gallery', async () => {
  const job = await makeFeedbackJob('disabled-with-pair', { enabled: false });
  const pairID = randomUUID().replaceAll('-', '');
  writePair(job.run_dir, pairID, { created_at: '2026-11-18T12:34:56+00:00' });

  let listed = await bodyOf(await callList(job.id));
  assert.equal(listed.enabled, false);
  assert.equal(listed.pairs.length, 1, 'published pairs remain visible when feedback is off');

  const skipped = await callVote(job.id, pairID, { choice: 'skip' });
  assert.equal(skipped.status, 200);
  assert.equal((await bodyOf(skipped)).vote.choice, 'skip');
  listed = await bodyOf(await callList(job.id));
  assert.equal(listed.pairs[0].status, 'skipped');
  assert.deepEqual(listed.counts, { pending: 0, rated: 0, skipped: 1, invalid: 0 });
});

test('ordinary jobs stay empty and invalid ids/arms/choices are rejected without touching paths', async () => {
  const job = await makeFeedbackJob('ordinary-empty', { enabled: false });
  const listed = await bodyOf(await callList(job.id));
  assert.deepEqual(listed, { enabled: false, pairs: [], counts: { pending: 0, rated: 0, skipped: 0, invalid: 0 } });

  assert.equal((await callList('not-a-uuid')).status, 400);
  assert.equal((await callList('../../etc')).status, 400);
  const pairID = randomUUID().replaceAll('-', '');
  assert.equal((await callVote(job.id, 'not-a-uuid', { choice: 'a' })).status, 400);
  assert.equal((await callVote(job.id, pairID, { choice: 'path/to/file' })).status, 400);
  assert.equal((await callImage(job.id, pairID, 'a.png')).status, 400, 'arm is not a filename');
  assert.equal((await callImage(job.id, 'not-a-uuid', 'a')).status, 400);
  assert.equal((await callVote(randomUUID(), pairID, { choice: 'a' })).status, 404);
});

test('malformed metadata and non-UUID directories are omitted; a malformed vote is surfaced as invalid and never overwritten', async () => {
  const job = await makeFeedbackJob('defensive-metadata');
  const goodID = randomUUID().replaceAll('-', '');
  writePair(job.run_dir, goodID);
  const badID = randomUUID().replaceAll('-', '');
  writePair(job.run_dir, badID, { version: 999 });
  const strayID = '../escape';
  fs.mkdirSync(path.join(job.run_dir, 'preferences', strayID), { recursive: true });

  let list = await bodyOf(await callList(job.id));
  assert.deepEqual(list.pairs.map(p => p.id), [goodID]);

  const votePath = path.join(job.run_dir, 'preferences', goodID, 'vote.json');
  fs.writeFileSync(votePath, '{not json');
  const beforeBytes = fs.readFileSync(votePath);
  list = await bodyOf(await callList(job.id));
  assert.equal(list.pairs[0].status, 'invalid');
  assert.deepEqual(list.counts, { pending: 0, rated: 0, skipped: 0, invalid: 1 });

  const result = await callVote(job.id, goodID, { choice: 'a' });
  assert.equal(result.status, 409);
  assert.match((await bodyOf(result)).error, /invalid|overwrite/i);
  assert.deepEqual(fs.readFileSync(votePath), beforeBytes, 'malformed vote is not overwritten');
});

test('concurrent different votes: exactly one immutable winner and one conflict', async () => {
  const job = await makeFeedbackJob('vote-race');
  const pairID = randomUUID().replaceAll('-', '');
  const pairDir = writePair(job.run_dir, pairID);

  const [ra, rb] = await Promise.all([
    callVote(job.id, pairID, { choice: 'a' }),
    callVote(job.id, pairID, { choice: 'b' }),
  ]);
  assert.deepEqual([ra.status, rb.status].sort(), [200, 409]);
  const stored = JSON.parse(fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8'));
  assert.ok(stored.choice === 'a' || stored.choice === 'b');
  const winner = ra.status === 200 ? 'a' : 'b';
  assert.equal(stored.choice, winner);
  const listed = await bodyOf(await callList(job.id));
  assert.equal(listed.pairs[0].status, 'rated');
  assert.equal(listed.pairs[0].vote.choice, winner);

  const noTempFiles = fs.readdirSync(pairDir).filter(name => name.startsWith('.vote.'));
  assert.deepEqual(noTempFiles, [], 'temporary vote files are cleaned up');
});

test('path defenses: outside run bindings, linked preference roots, and linked pair files are refused', async () => {
  const outside = fs.mkdtempSync(path.join(SCRATCH, 'preference-outside-'));
  roots.push(outside);
  const outsideJobFolder = path.join(outside, 'escape');
  fs.mkdirSync(outsideJobFolder, { recursive: true });

  const escaped = await makeFeedbackJob('escaped-root', { runDir: outsideJobFolder });
  const escapedList = await callList(escaped.id);
  assert.equal(escapedList.status, 409);
  assert.match((await bodyOf(escapedList)).error, /claimed job folder/i);

  const linkedJob = await makeFeedbackJob('linked-prefs');
  fs.symlinkSync(outside, path.join(linkedJob.run_dir, 'preferences'), 'junction');
  const linkedList = await callList(linkedJob.id);
  assert.equal(linkedList.status, 403);
  assert.equal(JSON.stringify(await bodyOf(linkedList)).includes(outside), false, 'error does not reveal the outside path');

  const linkedPairJob = await makeFeedbackJob('linked-pair-file');
  const pairID = randomUUID().replaceAll('-', '');
  const pairDir = writePair(linkedPairJob.run_dir, pairID);
  const outsideImage = path.join(outside, 'secret.png');
  fs.writeFileSync(outsideImage, 'secret image bytes');
  let linked = false;
  try {
    fs.unlinkSync(path.join(pairDir, 'a.png'));
    fs.symlinkSync(outsideImage, path.join(pairDir, 'a.png'), 'file');
    linked = true;
  } catch (error) {
    if (error?.code !== 'EPERM' && error?.code !== 'EACCES') throw error;
  }
  if (linked) {
    const image = await callImage(linkedPairJob.id, pairID, 'a');
    assert.equal(image.status, 403);
    assert.equal(JSON.stringify(await bodyOf(image)).includes(outside), false);
    const list = await bodyOf(await callList(linkedPairJob.id));
    assert.deepEqual(list.pairs, [], 'a linked image makes that pair ineligible');
  }
});

test('existing bearer middleware protects preference reads and votes', () => {
  const middleware = require(ts.uiPath('src/middleware.ts')).middleware;
  const request = (method, jobID, token) => ({
    method,
    headers: new Headers(token ? { Authorization: `Bearer ${token}` } : {}),
    nextUrl: { pathname: `/api/jobs/${jobID}/preferences` },
  });
  const old = process.env.AI_TOOLKIT_AUTH;
  try {
    process.env.AI_TOOLKIT_AUTH = 'fixture-only-token';
    const id = randomUUID().replaceAll('-', '');
    assert.equal(middleware(request('GET', id)).status, 401);
    assert.equal(middleware(request('POST', id)).status, 401);
    assert.equal(middleware(request('GET', id, 'fixture-only-token')).next, true);
    assert.equal(middleware(request('POST', id, 'fixture-only-token')).next, true);
  } finally {
    if (old === undefined) delete process.env.AI_TOOLKIT_AUTH;
    else process.env.AI_TOOLKIT_AUTH = old;
  }
});

test('Samples integration is opt-in and the browser component never handles filesystem paths', () => {
  const samples = fs.readFileSync(ts.uiPath('src/components/SampleImages.tsx'), 'utf8');
  assert.match(samples, /import PreferenceFeedback from '\.\/PreferenceFeedback'/);
  assert.match(samples, /<PreferenceFeedback job=\{job\} \/>/);

  const component = fs.readFileSync(ts.uiPath('src/components/PreferenceFeedback.tsx'), 'utf8');
  assert.match(component, /const shouldShow = configured \|\| !!feedback\?\.enabled \|\| \(feedback\?\.pairs\.length \?\? 0\) > 0/);
  assert.match(component, /\/api\/jobs\/\$\{encodeURIComponent\(job\.id\)\}\/preferences/);
  assert.doesNotMatch(component, /from ['"]fs['"]|from ['"]path['"]/);
  assert.match(component, /votingRef\.current !== null/, 'double-click guard exists before the network mutation');
});
