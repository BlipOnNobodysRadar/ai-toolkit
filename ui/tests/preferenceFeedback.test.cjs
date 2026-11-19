// Preference feedback contract: real route modules + helper on isolated Prisma and temporary files.
//   CUDA_VISIBLE_DEVICES=-1 node --test --test-concurrency=1 ui/tests/preferenceFeedback.test.cjs
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { randomUUID } = require('crypto');
const sqlite3 = require('sqlite3');
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

const voteRecord = (pairID, choice, votedAt = '2026-11-18T13:00:00.000Z') => ({
  version: 1,
  id: pairID,
  choice,
  voted_at: votedAt,
});

const writeVote = (pairDir, vote) => {
  fs.writeFileSync(path.join(pairDir, 'vote.json'), JSON.stringify(vote, null, 2) + '\n');
};

const writeDecision = (pairDir, vote) => {
  const decisionDir = path.join(pairDir, '.decision');
  fs.mkdirSync(decisionDir, { recursive: true });
  fs.writeFileSync(path.join(decisionDir, 'vote.json'), JSON.stringify(vote, null, 2) + '\n');
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

const sqliteRun = (db, sql, params = []) =>
  new Promise((resolve, reject) => db.run(sql, params, error => (error ? reject(error) : resolve())));
const sqliteOpen = file => new Promise((resolve, reject) => {
  const db = new sqlite3.Database(file, error => (error ? reject(error) : resolve(db)));
});
const sqliteClose = db => new Promise((resolve, reject) => db.close(error => (error ? reject(error) : resolve())));

const writeLossMetrics = async (jobFolder, rows) => {
  const file = path.join(jobFolder, 'loss_log.db');
  const db = await sqliteOpen(file);
  try {
    await sqliteRun(db, 'CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL);');
    await sqliteRun(db, 'CREATE TABLE metric_keys (key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER);');
    await sqliteRun(
      db,
      'CREATE TABLE metrics (step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL, value_text TEXT, PRIMARY KEY (step, key));',
    );
    await sqliteRun(db, 'CREATE INDEX idx_metrics_key_step ON metrics (key, step);');
    const bounds = new Map();
    for (const row of rows) {
      await sqliteRun(db, 'INSERT OR REPLACE INTO steps(step, wall_time) VALUES(?, ?);', [row.step, row.wall_time ?? row.step]);
      await sqliteRun(
        db,
        'INSERT OR REPLACE INTO metrics(step, key, value_real, value_text) VALUES(?, ?, ?, NULL);',
        [row.step, row.key, row.value],
      );
      const prior = bounds.get(row.key);
      bounds.set(row.key, {
        first: prior ? Math.min(prior.first, row.step) : row.step,
        last: prior ? Math.max(prior.last, row.step) : row.step,
      });
    }
    for (const [key, bound] of bounds) {
      await sqliteRun(
        db,
        'INSERT OR REPLACE INTO metric_keys(key, first_seen_step, last_seen_step) VALUES(?, ?, ?);',
        [key, bound.first, bound.last],
      );
    }
  } finally {
    await sqliteClose(db);
  }
  return file;
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


test('portable decisions work without hard links and persist genuine A and B choices', async () => {
  const job = await makeFeedbackJob('portable-ab');
  const pairA = randomUUID().replaceAll('-', '');
  const pairB = randomUUID().replaceAll('-', '');
  const dirA = writePair(job.run_dir, pairA);
  const dirB = writePair(job.run_dir, pairB);

  const originalLink = fs.promises.link;
  fs.promises.link = async () => {
    throw Object.assign(new Error('hard links are unavailable in this test'), { code: 'ENOSYS' });
  };
  let resultA;
  let resultB;
  try {
    [resultA, resultB] = await Promise.all([
      callVote(job.id, pairA, { choice: 'a' }),
      callVote(job.id, pairB, { choice: 'b' }),
    ]);
  } finally {
    fs.promises.link = originalLink;
  }

  assert.equal(resultA.status, 200, JSON.stringify(await bodyOf(resultA)));
  assert.equal(resultB.status, 200, JSON.stringify(await bodyOf(resultB)));
  const bodyA = await bodyOf(resultA);
  const bodyB = await bodyOf(resultB);
  assert.equal(bodyA.created, true);
  assert.equal(bodyB.created, true);
  assert.equal(bodyA.vote.choice, 'a');
  assert.equal(bodyB.vote.choice, 'b');

  for (const [pairDir, pairID, expectedChoice, body] of [
    [dirA, pairA, 'a', bodyA],
    [dirB, pairB, 'b', bodyB],
  ]) {
    const topText = fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8');
    const decisionText = fs.readFileSync(path.join(pairDir, '.decision', 'vote.json'), 'utf8');
    const top = JSON.parse(topText);
    assert.equal(topText, decisionText, 'the public receipt is the committed decision content');
    assert.deepEqual(Object.keys(top).sort(), ['choice', 'id', 'version', 'voted_at']);
    assert.equal(top.version, 1);
    assert.equal(top.id, pairID);
    assert.equal(top.choice, expectedChoice);
    assert.match(top.voted_at, /^\d{4}-\d{2}-\d{2}T.*Z$/);
    assert.deepEqual(body.vote, top);
    assert.equal(fs.statSync(path.join(pairDir, 'vote.json')).nlink, 1, 'top-level receipt is a regular independent file');
    assert.equal(fs.statSync(path.join(pairDir, '.decision', 'vote.json')).nlink, 1, 'decision file is a regular independent file');
  }
});

test('legacy top-level vote.json is authoritative before any portable decision exists', async () => {
  const job = await makeFeedbackJob('legacy-top-level');
  const pairID = randomUUID().replaceAll('-', '');
  const pairDir = writePair(job.run_dir, pairID);
  const legacy = voteRecord(pairID, 'a', '2026-11-18T12:00:00.000Z');
  writeVote(pairDir, legacy);
  const legacyText = fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8');

  const same = await callVote(job.id, pairID, { choice: 'a' });
  assert.equal(same.status, 200);
  const sameBody = await bodyOf(same);
  assert.equal(sameBody.created, false);
  assert.deepEqual(sameBody.vote, legacy);
  assert.equal(fs.existsSync(path.join(pairDir, '.decision')), false, 'legacy success does not create a second decision');

  const conflict = await callVote(job.id, pairID, { choice: 'b' });
  assert.equal(conflict.status, 409);
  assert.match((await bodyOf(conflict)).error, /different immutable vote/);
  assert.equal(fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8'), legacyText);
  assert.equal(fs.existsSync(path.join(pairDir, '.decision')), false, 'legacy conflict does not mutate the pair');
});

test('concurrent same-choice votes are idempotent with one complete winner', async () => {
  const job = await makeFeedbackJob('vote-race-same');
  const pairID = randomUUID().replaceAll('-', '');
  const pairDir = writePair(job.run_dir, pairID);

  const responses = await Promise.all(
    Array.from({ length: 6 }, () => callVote(job.id, pairID, { choice: 'a' })),
  );
  assert.ok(responses.every(response => response.status === 200));
  const bodies = await Promise.all(responses.map(bodyOf));
  const winners = bodies.filter(body => body.created);
  assert.equal(winners.length, 1, 'exactly one publisher created the immutable decision');
  assert.ok(bodies.every(body => body.vote.choice === 'a'));
  assert.ok(bodies.every(body => JSON.stringify(body.vote) === JSON.stringify(winners[0].vote)));

  const topText = fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8');
  const decisionText = fs.readFileSync(path.join(pairDir, '.decision', 'vote.json'), 'utf8');
  assert.equal(topText, decisionText);
  assert.deepEqual(JSON.parse(topText), winners[0].vote);
  assert.deepEqual(fs.readdirSync(pairDir).filter(name => name.startsWith('.vote.')), []);
});

test('interrupted projection heals from .decision and retries recover the committed winner', async () => {
  const job = await makeFeedbackJob('interrupted-projection');
  const pairID = randomUUID().replaceAll('-', '');
  const pairDir = writePair(job.run_dir, pairID);
  const committed = voteRecord(pairID, 'b', '2026-11-18T14:00:00.000Z');
  writeDecision(pairDir, committed);
  const decisionText = fs.readFileSync(path.join(pairDir, '.decision', 'vote.json'), 'utf8');

  const listed = await bodyOf(await callList(job.id));
  assert.equal(listed.pairs[0].status, 'rated');
  assert.deepEqual(listed.pairs[0].vote, committed);
  assert.deepEqual(listed.counts, { pending: 0, rated: 1, skipped: 0, invalid: 0 });
  assert.equal(fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8'), decisionText, 'read healed the receipt');

  fs.unlinkSync(path.join(pairDir, 'vote.json'));
  const same = await callVote(job.id, pairID, { choice: 'b' });
  assert.equal(same.status, 200);
  const sameBody = await bodyOf(same);
  assert.equal(sameBody.created, false);
  assert.deepEqual(sameBody.vote, committed);
  assert.equal(fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8'), decisionText);

  fs.unlinkSync(path.join(pairDir, 'vote.json'));
  const conflict = await callVote(job.id, pairID, { choice: 'a' });
  assert.equal(conflict.status, 409);
  assert.match((await bodyOf(conflict)).error, /different immutable vote/);
  assert.equal(fs.readFileSync(path.join(pairDir, 'vote.json'), 'utf8'), decisionText);
  assert.equal(fs.readFileSync(path.join(pairDir, '.decision', 'vote.json'), 'utf8'), decisionText, 'the real decision was never reversed');
  assert.deepEqual(fs.readdirSync(pairDir).filter(name => name.startsWith('.vote.')), []);
});

test('linked .decision directories are rejected without reading outside or writing a receipt', async () => {
  const outside = fs.mkdtempSync(path.join(SCRATCH, 'preference-decision-outside-'));
  roots.push(outside);
  const job = await makeFeedbackJob('linked-decision');
  const pairID = randomUUID().replaceAll('-', '');
  const pairDir = writePair(job.run_dir, pairID);
  const outsideDecision = path.join(outside, 'decision');
  fs.mkdirSync(outsideDecision, { recursive: true });
  fs.writeFileSync(path.join(outsideDecision, 'vote.json'), JSON.stringify(voteRecord(pairID, 'a'), null, 2) + '\n');

  let linked = false;
  try {
    fs.symlinkSync(outsideDecision, path.join(pairDir, '.decision'), 'junction');
    linked = true;
  } catch (error) {
    if (error?.code !== 'EPERM' && error?.code !== 'EACCES') throw error;
  }
  if (!linked) return;

  const listed = await bodyOf(await callList(job.id));
  assert.equal(listed.pairs[0].status, 'invalid');
  assert.deepEqual(listed.counts, { pending: 0, rated: 0, skipped: 0, invalid: 1 });
  assert.equal(JSON.stringify(listed).includes(outside), false, 'no outside path leaks');

  const vote = await callVote(job.id, pairID, { choice: 'a' });
  assert.equal(vote.status, 409);
  assert.equal(fs.existsSync(path.join(pairDir, 'vote.json')), false, 'linked decision cannot create a top-level receipt');
});

test('training summary uses latest real feedback metrics, keeps exact own steps, and preserves actual zeros', async () => {
  const job = await makeFeedbackJob('training-metrics-latest');
  const pairID = randomUUID().replaceAll('-', '');
  writePair(job.run_dir, pairID);
  await writeLossMetrics(job.run_dir, [
    { step: 3, key: 'feedback/updates', value: 0 },
    { step: 9, key: 'feedback/updates', value: 0 },
    { step: 4, key: 'feedback/optimizer_updates', value: 12 },
    { step: 8, key: 'feedback/optimizer_updates', value: 41 },
    { step: 6, key: 'feedback/foreign_votes', value: 0 },
    { step: 10, key: 'feedback/foreign_votes', value: 2 },
    { step: 12, key: 'loss/total', value: 0.5 },
  ]);

  const listed = await callList(job.id);
  assert.equal(listed.status, 200);
  const list = await bodyOf(listed);
  assert.deepEqual(list.training, {
    preference_updates: { value: 0, step: 9 },
    optimizer_updates: { value: 41, step: 8 },
    foreign_votes: { value: 2, step: 10 },
  });
  assert.equal(list.pairs.length, 1, 'metrics and pair listing coexist');

  // A rewind/stale-job display must keep the log row's own steps rather than borrowing the job's current step.
  await F.db.job.update({ where: { id: job.id }, data: { step: 1 } });
  const afterRewind = await bodyOf(await callList(job.id));
  assert.deepEqual(afterRewind.training, list.training);
});


test('missing feedback metrics stay absent while a present zero is not treated as missing', async () => {
  const partial = await makeFeedbackJob('training-metrics-partial');
  writePair(partial.run_dir, randomUUID().replaceAll('-', ''));
  await writeLossMetrics(partial.run_dir, [{ step: 7, key: 'feedback/updates', value: 0 }]);
  const partialList = await bodyOf(await callList(partial.id));
  assert.deepEqual(partialList.training, {
    preference_updates: { value: 0, step: 7 },
    optimizer_updates: null,
    foreign_votes: null,
  });

  const ordinaryLog = await makeFeedbackJob('training-metrics-unrelated');
  writePair(ordinaryLog.run_dir, randomUUID().replaceAll('-', ''));
  await writeLossMetrics(ordinaryLog.run_dir, [{ step: 5, key: 'loss/total', value: 1 }]);
  const ordinaryList = await bodyOf(await callList(ordinaryLog.id));
  assert.equal(Object.hasOwn(ordinaryList, 'training'), false, 'no fabricated summary without feedback metrics');
});

test('corrupt training log fails soft: list and voting still work', async () => {
  const job = await makeFeedbackJob('training-metrics-corrupt');
  const pairID = randomUUID().replaceAll('-', '');
  writePair(job.run_dir, pairID);
  fs.writeFileSync(path.join(job.run_dir, 'loss_log.db'), 'not a sqlite database');

  const listed = await callList(job.id);
  assert.equal(listed.status, 200);
  const list = await bodyOf(listed);
  assert.equal(list.pairs.length, 1);
  assert.equal(Object.hasOwn(list, 'training'), false);

  const vote = await callVote(job.id, pairID, { choice: 'a' });
  assert.equal(vote.status, 200, JSON.stringify(await bodyOf(vote)));
  assert.equal((await bodyOf(vote)).vote.choice, 'a');
});

test('linked loss logs and unsafe sidecars are not followed', async () => {
  const outside = fs.mkdtempSync(path.join(SCRATCH, 'training-metrics-outside-'));
  roots.push(outside);
  await writeLossMetrics(outside, [{ step: 11, key: 'feedback/updates', value: 7 }]);
  const outsideDb = path.join(outside, 'loss_log.db');

  const linkedJob = await makeFeedbackJob('training-metrics-linked');
  writePair(linkedJob.run_dir, randomUUID().replaceAll('-', ''));
  let linked = false;
  try {
    fs.symlinkSync(outsideDb, path.join(linkedJob.run_dir, 'loss_log.db'), 'file');
    linked = true;
  } catch (error) {
    if (error?.code !== 'EPERM' && error?.code !== 'EACCES') throw error;
  }
  if (!linked) return;
  const linkedList = await bodyOf(await callList(linkedJob.id));
  assert.equal(linkedList.pairs.length, 1);
  assert.equal(Object.hasOwn(linkedList, 'training'), false, 'linked DB is refused without hiding the gallery');

  const sidecarJob = await makeFeedbackJob('training-metrics-sidecar');
  writePair(sidecarJob.run_dir, randomUUID().replaceAll('-', ''));
  await writeLossMetrics(sidecarJob.run_dir, [{ step: 4, key: 'feedback/updates', value: 3 }]);
  try {
    fs.symlinkSync(outsideDb, path.join(sidecarJob.run_dir, 'loss_log.db-wal'), 'file');
  } catch (error) {
    if (error?.code !== 'EPERM' && error?.code !== 'EACCES') throw error;
    return;
  }
  const sidecarList = await bodyOf(await callList(sidecarJob.id));
  assert.equal(sidecarList.pairs.length, 1);
  assert.equal(Object.hasOwn(sidecarList, 'training'), false, 'linked WAL sidecar is refused');
});

test('a locked training log does not break the preference gallery', async () => {
  const job = await makeFeedbackJob('training-metrics-locked');
  writePair(job.run_dir, randomUUID().replaceAll('-', ''));
  const dbFile = await writeLossMetrics(job.run_dir, [{ step: 3, key: 'feedback/updates', value: 1 }]);
  const lock = await sqliteOpen(dbFile);
  try {
    await sqliteRun(lock, 'BEGIN EXCLUSIVE;');
    const listed = await callList(job.id);
    assert.equal(listed.status, 200);
    const list = await bodyOf(listed);
    assert.equal(list.pairs.length, 1);
    assert.equal(Object.hasOwn(list, 'training'), false);
  } finally {
    await sqliteRun(lock, 'ROLLBACK;').catch(() => {});
    await sqliteClose(lock);
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
