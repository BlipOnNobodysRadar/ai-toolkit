// Job deletion across processes (docs/STATE_MACHINES.md M35): the actual deleteJob/delete route on an isolated Prisma
// client + temporary SQLite, with REAL child Node processes for the crashed and the overlapping deleter.
//   node --test --test-concurrency=1 ui/tests/deletion.test.cjs
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { setupDb, makeJob, routeCall, ts } = require('./harness/fixture.cjs');
const { SCRATCH } = require('./harness/env.cjs');

let F, store, deleteRoute, jobsRoute;
before(async () => {
  F = await setupDb('deletion');
  store = require(ts.uiPath('cron/ownership/store.ts'));
  deleteRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/delete/route.ts'));
  jobsRoute = require(ts.uiPath('src/app/api/jobs/route.ts'));
});
after(async () => F.db.$disconnect());

const child = (jobId, mode, gate) =>
  new Promise((resolve, reject) => {
    const c = spawn(process.execPath, [path.join(__dirname, 'fixtures', 'deleteChild.cjs'), F.file, F.trainingRoot, jobId, mode, gate ?? ''], {
      env: process.env,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    let err = '';
    c.stdout.on('data', d => (out += d));
    c.stderr.on('data', d => (err += d));
    c.on('error', reject);
    c.on('exit', code => resolve({ code, out, err }));
  });
const folderWithBytes = name => {
  const dir = path.join(F.trainingRoot, name);
  fs.mkdirSync(path.join(dir, 'samples'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'samples', 'a.bin'), Buffer.alloc(1024, 7));
  return dir;
};
const row = id => F.db.job.findUnique({ where: { id } });

test('a deleter process that dies after committing the reservation is resumed by the next Delete (Codex delete-crash-retry-probe)', async () => {
  const j = await makeJob(F.db, { name: 'crashdel', status: 'stopped' });
  const dir = folderWithBytes('crashdel');
  const r = await child(j.id, 'exit-after-begin');
  assert.equal(r.code, 0, r.err);
  assert.equal((await row(j.id)).status, 'deleting', 'the crashed deleter left its reservation');
  assert.equal(await F.db.runClaim.count({ where: { kind: 'delete' } }), 1);
  assert.equal(fs.existsSync(dir), true);
  // meanwhile the name/folder stay reserved
  assert.equal((await jobsRoute.POST({ json: async () => ({ name: 'CrashDel', gpu_ids: '0', job_config: {} }) })).status, 409);
  const out = await routeCall(deleteRoute, 'GET', j.id);
  assert.equal(out.status, 200, JSON.stringify(out.body));
  assert.equal(await row(j.id), null);
  assert.equal(fs.existsSync(dir), false);
  assert.equal(await F.db.runClaim.count(), 0);
});

test('an overlapping Delete while another process holds the deletion lock is refused, and the first one completes', async () => {
  const j = await makeJob(F.db, { name: 'overlap', status: 'stopped' });
  const dir = folderWithBytes('overlap');
  const gate = path.join(fs.mkdtempSync(path.join(SCRATCH, 'delgate-')), 'go');
  const first = child(j.id, 'hold-until-gate', gate);
  while (!fs.existsSync(`${gate}.holding`)) await new Promise(r => setTimeout(r, 20));
  const second = await routeCall(deleteRoute, 'GET', j.id);
  assert.equal(second.status, 409);
  assert.match(second.body.error, /being deleted by another request/);
  fs.writeFileSync(gate, '');
  const r = await first;
  assert.equal(r.code, 0, r.err);
  assert.equal(JSON.parse(r.out).status, 200);
  assert.equal(await row(j.id), null);
  assert.equal(fs.existsSync(dir), false);
});

test('a failed final DB write leaves a resumable deleting row; the retry finishes, the folder reserved in between', async () => {
  const j = await makeJob(F.db, { name: 'dbfail', status: 'stopped' });
  folderWithBytes('dbfail');
  const failJob = tt => new Proxy(tt.job, { get: (jj, m) => (m === 'deleteMany' ? () => Promise.reject(new Error('injected DB failure')) : jj[m]) });
  const failing = new Proxy(F.db, {
    get(t, k) {
      if (k !== '$transaction') return t[k];
      return (fn, opts) => t.$transaction(tx => fn(new Proxy(tx, { get: (tt, kk) => (kk === 'job' ? failJob(tt) : tt[kk]) })), opts);
    },
  });
  const firstTry = await store.deleteJob(failing, j.id, F.trainingRoot);
  assert.equal(firstTry.status, 500);
  assert.match(firstTry.body.error, /retry Delete/);
  assert.equal((await row(j.id)).status, 'deleting');
  assert.equal(await F.db.runClaim.count({ where: { kind: 'delete' } }), 1, 'folder still reserved');
  assert.equal((await jobsRoute.POST({ json: async () => ({ name: 'dbfail', gpu_ids: '0', job_config: {} }) })).status, 409);
  const retry = await store.deleteJob(F.db, j.id, F.trainingRoot);
  assert.equal(retry.status, 200);
  assert.equal(await row(j.id), null);
  assert.equal(await F.db.runClaim.count(), 0);
});

test('a folder removal failure leaves delete_failed (still reserved); Delete can be retried', async () => {
  const j = await makeJob(F.db, { name: 'rmfail', status: 'stopped' });
  folderWithBytes('rmfail');
  const out = await store.deleteJob(F.db, j.id, F.trainingRoot, { rm: async () => Promise.reject(new Error('EBUSY: injected')) });
  assert.equal(out.status, 500);
  assert.equal((await row(j.id)).status, 'delete_failed');
  assert.equal(await F.db.runClaim.count({ where: { kind: 'delete' } }), 1);
  assert.equal((await store.deleteJob(F.db, j.id, F.trainingRoot)).status, 200);
  assert.equal(await row(j.id), null);
});
