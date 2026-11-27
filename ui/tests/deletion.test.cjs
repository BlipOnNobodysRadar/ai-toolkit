// Job deletion across processes (docs/STATE_MACHINES.md M35): the actual deleteJob/delete route on an isolated Prisma
// client + temporary SQLite, with REAL child Node processes for the crashed and the overlapping deleter.
//   node --test --test-concurrency=1 ui/tests/deletion.test.cjs
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { once } = require('node:events');
const { spawn } = require('child_process');
const { setupDb, makeJob, routeCall, fakeProcCtl, ts } = require('./harness/fixture.cjs');
const { SCRATCH, TOOLKIT_ROOT } = require('./harness/env.cjs');

let F, store, deleteRoute, jobsRoute;
before(async () => {
  F = await setupDb('deletion');
  ts.mock('@/paths', { ...require(ts.uiPath('src/paths.ts')), TOOLKIT_ROOT });
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
  assert.equal((await jobsRoute.POST({ json: async () => ({ name: process.platform === 'win32' ? 'CrashDel' : j.name, gpu_ids: '0', job_config: {} }) })).status, 409);
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

test('a retry after changing Training Folder removes the original run only', async () => {
  const j = await makeJob(F.db, { name: 'root-change' });
  const original = folderWithBytes(j.name);
  const failed = await store.deleteJob(F.db, j.id, F.trainingRoot, { rm: async () => { throw Error('busy'); } });
  assert.equal(failed.status, 500);
  const newRoot = fs.mkdtempSync(path.join(SCRATCH, 'new-training-'));
  fs.mkdirSync(path.join(newRoot, j.name));
  fs.writeFileSync(path.join(newRoot, j.name, 'unrelated'), 'keep');
  const retry = await store.deleteJob(F.db, j.id, newRoot);
  assert.equal(retry.status, 200, JSON.stringify(retry.body));
  assert.equal(fs.existsSync(original), false);
  assert.equal(fs.readFileSync(path.join(newRoot, j.name, 'unrelated'), 'utf8'), 'keep');
});

test('renamed runs retain their launch folder, and deleting one of two references keeps the other checkpoints', async () => {
  const runDir = folderWithBytes('before-rename');
  const renamed = await makeJob(F.db, { name: 'after-rename', run_id: 'old-run', run_dir: runDir, launch_phase: 'exited' });
  fs.writeFileSync(path.join(runDir, '.job_config.old-run.json'), '{}');
  const alias = await jobsRoute.POST({ json: async () => ({ name: 'before-rename', gpu_ids: '0', job_config: {} }) });
  assert.equal(alias.status, 409);
  const existing = await makeJob(F.db, { name: 'before-rename' });
  const deleted = await routeCall(deleteRoute, 'POST', existing.id, existing);
  assert.equal(deleted.status, 200, JSON.stringify(deleted.body));
  assert.match(deleted.body.keptOutput, /after-rename/);
  assert.equal(await row(existing.id), null);
  assert.equal(fs.existsSync(path.join(runDir, 'samples', 'a.bin')), true);
  assert.equal((await routeCall(deleteRoute, 'POST', renamed.id, renamed)).status, 200);
  assert.equal(fs.existsSync(runDir), false);
});

test('a stale confirmation cannot delete a renamed job, and a failed process check cannot delete checkpoints', async () => {
  const j = await makeJob(F.db, { name: 'was-called-this' });
  await F.db.job.update({ where: { id: j.id }, data: { name: 'renamed-since-confirmation' } });
  assert.equal((await routeCall(deleteRoute, 'POST', j.id, j)).status, 409);
  const legacy = await makeJob(F.db, { name: 'unverified-legacy', status: 'running', pid: 123 });
  const check = require(ts.uiPath('cron/ownership/recoverStopped.ts')).checkStopped;
  assert.equal((await check(F.db, fakeProcCtl({ down: true }), legacy.id, legacy, 'delete')).status, 409);
  assert.equal((await check(F.db, fakeProcCtl({ describe: { 123: { state: 'gone' } } }), legacy.id, legacy, 'delete')).status, 200);
});

test('a legacy row with an invalid folder can be removed without touching any files', async () => {
  const j = await makeJob(F.db, { name: '..' });
  const kept = folderWithBytes('untouched-run');
  const out = await routeCall(deleteRoute, 'POST', j.id, j);
  assert.equal(out.status, 200, JSON.stringify(out.body));
  assert.match(out.body.keptOutput, /no valid output folder/);
  assert.equal(await row(j.id), null);
  assert.equal(fs.existsSync(path.join(kept, 'samples', 'a.bin')), true);
});

test('a historical root junction resolves the original run and preserves files when retargeted', async () => {
  const dir = fs.mkdtempSync(path.join(SCRATCH, 'delete-root-link-'));
  const original = path.join(dir, 'original');
  const replacement = path.join(dir, 'replacement');
  const link = path.join(dir, 'old-root');
  const name = 'linked-job';
  for (const root of [original, replacement]) fs.mkdirSync(path.join(root, name), { recursive: true });
  fs.symlinkSync(original, link, 'junction');
  const j = await makeJob(F.db, { name, status: 'stopped', run_dir: path.join(link, name), run_id: 'last-run', launch_phase: 'exited' });
  fs.writeFileSync(path.join(original, name, '.job_config.last-run.json'), '{}');
  const { jobOutputFolder } = require(ts.uiPath('cron/ownership/resources.ts'));
  assert.equal(jobOutputFolder(F.trainingRoot, j), j.run_dir);
  assert.equal(jobOutputFolder(F.trainingRoot, j, 'delete'), path.join(fs.realpathSync(original), name));
  fs.unlinkSync(link);
  fs.symlinkSync(replacement, link, 'junction');
  const checkpoint = path.join(replacement, name, 'keep.safetensors');
  fs.writeFileSync(checkpoint, 'another run');
  assert.throws(() => jobOutputFolder(F.trainingRoot, j), /historical output link/);
  const result = await routeCall(deleteRoute, 'POST', j.id, j);
  assert.equal(result.status, 200);
  assert.ok(result.body.keptOutput);
  assert.equal(fs.readFileSync(checkpoint, 'utf8'), 'another run');
});

test('a delayed launch of a deleted claim leaves the removed folder gone', async () => {
  const j = await makeJob(F.db, { name: 'delayed-start', status: 'queued', gpu_ids: '-1' });
  const { admitted } = await store.admit(F.db, j.id, F.trainingRoot, { pid: process.pid, identity: '123@start' });
  assert.ok(admitted);
  assert.equal((await routeCall(deleteRoute, 'POST', j.id, admitted)).status, 200);
  const { launchRun } = require(ts.uiPath('cron/ownership/launch.ts'));
  await launchRun({ db: F.db, inFlight: new Set(), sqliteDbPath: F.file, runPy: 'no-such-program' }, admitted);
  assert.equal(fs.existsSync(admitted.run_dir), false, 'stale launch recreated the job folder');
});

test('Delete from an older job row keeps a real live process, then succeeds after it exits', async () => {
  const name = 'old-process';
  const dir = folderWithBytes(name);
  const worker = spawn(process.execPath, ['-e', 'process.stdin.resume()', path.join(dir, '.job_config.json')], {
    stdio: ['pipe', 'ignore', 'ignore'],
  });
  const ended = once(worker, 'exit');
  try {
    await once(worker, 'spawn');
    const j = await makeJob(F.db, { name, status: 'completed', pid: worker.pid });
    const live = await routeCall(deleteRoute, 'POST', j.id, j);
    assert.equal(live.status, 409);
    assert.match(live.body.error, /running process/);
    assert.equal(fs.existsSync(dir), true);
    worker.stdin.end();
    await ended;
    const stopped = await routeCall(deleteRoute, 'POST', j.id, j);
    assert.equal(stopped.status, 200, JSON.stringify(stopped.body));
    assert.equal(await row(j.id), null);
    assert.equal(fs.existsSync(dir), false);
  } finally {
    worker.stdin.end();
    await ended;
  }
});

test('deletion retains both verified copies and protects a renamed relocated owner from another job', async () => {
  const oldRoot = fs.mkdtempSync(path.join(SCRATCH, 'moved-from-'));
  const old = path.join(oldRoot, 'output-name');
  const moved = folderWithBytes('output-name');
  const marker = '.job_config.rel-generation.json';
  fs.writeFileSync(path.join(moved, marker), '{}');
  const j = await makeJob(F.db, { name: 'display-name', run_id: 'rel-generation', run_dir: old, launch_phase: 'exited', step: 800 });
  const alias = await makeJob(F.db, { name: 'output-name' });
  const shared = await store.deleteJob(F.db, alias.id, F.trainingRoot);
  assert.equal(shared.status, 200);
  assert.match(shared.body.keptOutput, /display-name/);
  assert.equal(fs.existsSync(path.join(moved, marker)), true);
  fs.mkdirSync(old);
  fs.writeFileSync(path.join(old, marker), '{}');
  const ambiguous = await store.deleteJob(F.db, j.id, F.trainingRoot);
  assert.equal(ambiguous.status, 200);
  assert.ok(ambiguous.body.keptOutput);
  assert.equal(fs.existsSync(path.join(old, marker)), true);
  assert.equal(fs.existsSync(path.join(moved, marker)), true);
});

test('deleting a missing historical run leaves an unrelated same-name folder in the configured root alone', async () => {
  const oldRoot = fs.mkdtempSync(path.join(SCRATCH, 'missing-old-'));
  const name = 'unrelated-output';
  const unrelated = folderWithBytes(name);
  fs.writeFileSync(path.join(unrelated, '.job_config.other-generation.json'), '{}');
  const j = await makeJob(F.db, { name, run_id: 'lost-generation', run_dir: path.join(oldRoot, name), launch_phase: 'exited', step: 800 });
  assert.equal((await store.deleteJob(F.db, j.id, F.trainingRoot)).status, 200);
  assert.equal(fs.existsSync(path.join(unrelated, 'samples', 'a.bin')), true);
  const replaced = path.join(oldRoot, name);
  fs.mkdirSync(replaced);
  fs.writeFileSync(path.join(replaced, 'foreign'), 'keep');
  const old = await makeJob(F.db, { name, run_id: 'lost-generation', run_dir: replaced, launch_phase: 'exited', step: 800 });
  assert.equal((await store.requestStart(F.db, old.id, F.trainingRoot)).status, 409, 'foreign files do not identify the original run');
  fs.writeFileSync(path.join(replaced, name + '_other_000000800.safetensors'), 'foreign');
  assert.equal((await store.requestStart(F.db, old.id, F.trainingRoot)).status, 409, 'a different checkpoint prefix does not identify the run');
  fs.writeFileSync(path.join(replaced, name + '_000000800.safetensors'), 'weights');
  assert.equal((await store.requestStart(F.db, old.id, F.trainingRoot)).status, 200, 'the original named checkpoint can resume without its UI config');
  assert.ok((await store.deleteJob(F.db, old.id, F.trainingRoot)).body.keptOutput);
  assert.equal(fs.readFileSync(path.join(replaced, 'foreign'), 'utf8'), 'keep');
});
