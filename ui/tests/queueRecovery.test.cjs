// Exercise an upgrade and the UI recovery actions against real OS processes and a temporary SQLite database.
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const fs = require('node:fs');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');
const { setupDb, makeJob, routeCall, ts, jobConfig } = require('./harness/fixture.cjs');
const { PYTHON, TOOLKIT_ROOT, UI_ROOT } = require('./harness/env.cjs');

const supported = ['linux', 'win32'].includes(process.platform);
const originalDir = process.cwd();
let F, pc, worker, reconcile, queue, launchRun, runFiles, mark, remove;

before(async () => {
  if (!supported) return;
  process.chdir(UI_ROOT); // Same working directory as npm start.
  F = await setupDb('queue-recovery');
  pc = require(ts.uiPath('cron/ownership/procctl.ts')).makeProcCtl(PYTHON, TOOLKIT_ROOT);
  worker = { ...(await pc.self(process.pid)), pid: process.pid };
  ({ reconcile } = require(ts.uiPath('cron/ownership/reconcile.ts')));
  ({ processQueueWith: queue } = require(ts.uiPath('cron/actions/processQueue.ts')));
  ({ launchRun, runFiles } = require(ts.uiPath('cron/ownership/launch.ts')));
  mark = require(ts.uiPath('src/app/api/jobs/[jobID]/mark_stopped/route.ts'));
  remove = require(ts.uiPath('src/app/api/jobs/[jobID]/delete/route.ts'));
});

after(async () => {
  if (F) await F.db.$disconnect();
  process.chdir(originalDir);
});

const row = id => F.db.job.findUnique({ where: { id } });
const snapshot = ({ run_id, launch_phase, status, pid }) => ({ run_id, launch_phase, status, pid });
const poll = () => reconcile({ db: F.db, procctl: pc, self: worker, inFlight: new Set(), trainingRoot: F.trainingRoot, tick: 0 });

test('recover pre-update jobs, delete a finished run, and complete the next launch', { skip: !supported }, async t => {
  const name = 'previous run';
  const dir = path.join(F.trainingRoot, name);
  fs.mkdirSync(dir);
  const checkpoint = path.join(dir, 'checkpoint.safetensors');
  fs.writeFileSync(checkpoint, 'previous checkpoint');
  const trainer = spawn(PYTHON, ['-c', 'import sys; sys.stdin.buffer.read()', path.join(dir, '.job_config.json')], {
    stdio: ['pipe', 'ignore', 'inherit'], windowsHide: true,
  });
  t.after(() => trainer.stdin.end());
  await once(trainer, 'spawn');
  const legacy = await makeJob(F.db, { name, pid: trainer.pid, status: 'running', step: 97 });
  const update = data => F.db.job.update({ where: { id: legacy.id }, data });

  for (const pid of [trainer.pid, null]) {
    const observed = await update({ pid, updated_at: pid == null ? new Date(Date.now() - 60_000) : new Date() });
    for (const action of [mark, remove]) {
      const result = await routeCall(action, 'POST', observed.id, snapshot(observed));
      assert.equal(result.status, 409);
      assert.match(result.body.error, /running process/);
    }
    await poll();
    assert.equal((await row(legacy.id)).status, 'running');
  }

  const exit = once(trainer, 'exit');
  trainer.stdin.end();
  assert.equal((await exit)[0], 0);
  const observed = await update({ pid: trainer.pid });
  assert.equal((await routeCall(mark, 'POST', legacy.id, snapshot(observed))).status, 200);
  assert.equal((await row(legacy.id)).step, 97);
  assert.equal(fs.readFileSync(checkpoint, 'utf8'), 'previous checkpoint');

  const archived = await makeJob(F.db, { name: 'remove-me', status: 'running', pid: trainer.pid });
  const archivedDir = path.join(F.trainingRoot, archived.name);
  fs.mkdirSync(archivedDir);
  fs.writeFileSync(path.join(archivedDir, 'checkpoint'), 'discard');
  assert.equal((await routeCall(remove, 'POST', archived.id, snapshot(archived))).status, 200);
  assert.equal(await row(archived.id), null);
  assert.equal(fs.existsSync(archivedDir), false);
  assert.equal(fs.readFileSync(checkpoint, 'utf8'), 'previous checkpoint');

  const stale = await makeJob(F.db, { name: 'already-finished', status: 'completed', pid: trainer.pid, step: 100 });
  const config = jobConfig('next-job');
  config.config.process[0].test_steps = ['status:running', 'status:completed'];
  const next = await makeJob(F.db, { name: config.config.name, status: 'queued', job_config: JSON.stringify(config) });
  await F.db.queue.create({ data: { gpu_ids: '0', is_running: true } });
  await poll();
  assert.equal((await row(stale.id)).pid, null);
  assert.equal((await row(stale.id)).status, 'completed');
  let launch;
  await queue({ db: F.db, trainingRoot: F.trainingRoot, launcher: worker, launch: job => {
    assert.equal(job.id, next.id);
    launch = launchRun({
      db: F.db, toolkitRoot: TOOLKIT_ROOT, helperPath: pc.helperPath, helperPython: PYTHON, jobPython: PYTHON,
      runPy: path.join(__dirname, 'fixtures', 'fake_run.py'), sqliteDbPath: F.file,
      env: { CUDA_VISIBLE_DEVICES: '-1' }, inFlight: new Set(),
    }, job);
  } });
  assert.equal((await launch)?.result, 'attached');
  const current = await row(next.id);
  const receipt = runFiles(current.run_dir, current.run_id).exit;
  const deadline = Date.now() + 15_000;
  while (!fs.existsSync(receipt) && Date.now() < deadline) await delay(50);
  assert.ok(fs.existsSync(receipt), 'next job finished through the normal launcher');
  await poll();
  assert.equal((await row(next.id)).status, 'completed');
  assert.equal(await F.db.runClaim.count({ where: { job_id: next.id } }), 0);
});

test('Force Stop ends only the pre-update trainer and its workers', { skip: !supported }, async t => {
  const kill = require(ts.uiPath('src/app/api/jobs/[jobID]/kill/route.ts'));
  const legacy = await makeJob(F.db, { name: 'old-stuck', status: 'running', step: 87 });
  const config = path.join(F.trainingRoot, legacy.name, '.job_config.json');
  const gate = path.join(F.trainingRoot, 'legacy-child.pid');
  const script = [
    'import pathlib,subprocess,sys,time',
    'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])',
    'pathlib.Path(sys.argv[1]).write_text(str(child.pid))',
    'child.wait()',
  ].join('\n');
  const trainer = spawn(PYTHON, ['-c', script, gate, config], {
    stdio: 'ignore', windowsHide: true, env: { ...process.env, AITK_JOB_ID: legacy.id, AITK_RUN_ID: '' },
  });
  const other = spawn(PYTHON, ['-c', 'import time; time.sleep(30)', config], {
    stdio: 'ignore', windowsHide: true, env: { ...process.env, AITK_JOB_ID: 'some-other-job' },
  });
  t.after(() => { trainer.kill(); other.kill(); });
  const deadline = Date.now() + 5000;
  while (!fs.existsSync(gate) && Date.now() < deadline) await delay(50);
  const childPid = Number(fs.readFileSync(gate, 'utf8'));
  const setPid = pid => F.db.job.update({ where: { id: legacy.id }, data: { pid } });
  await setPid(other.pid);
  assert.equal((await routeCall(kill, 'GET', legacy.id)).status, 409);
  await setPid(trainer.pid);
  const result = await routeCall(kill, 'GET', legacy.id);
  assert.equal(result.status, 200, JSON.stringify(result.body));
  assert.equal(result.body.killed, true);
  const observed = await pc.probe({ describe: [other.pid, trainer.pid, childPid] });
  assert.equal(observed.describe[other.pid].state, 'alive');
  assert.equal(observed.describe[trainer.pid].state, 'gone');
  assert.equal(observed.describe[childPid].state, 'gone');
  const recovered = await row(legacy.id);
  assert.equal(recovered.status, 'stopped');
  assert.equal(recovered.step, 87);
});

test('Linux Force Stop and overdue graceful stop both release the run', { skip: process.platform !== 'linux' }, async t => {
  const store = require(ts.uiPath('cron/ownership/store.ts'));
  const kill = require(ts.uiPath('src/app/api/jobs/[jobID]/kill/route.ts'));
  const stop = require(ts.uiPath('src/app/api/jobs/[jobID]/stop/route.ts'));
  for (const action of ['force', 'graceful']) {
    const name = 'linux-stop-' + action;
    const gate = path.join(F.trainingRoot, name + '.gate');
    t.after(() => fs.writeFileSync(gate, ''));
    const config = jobConfig(name);
    config.config.process[0].test_steps = ['wait:' + gate];
    const j = await makeJob(F.db, { name, status: 'queued', job_config: JSON.stringify(config) });
    const { admitted } = await store.admit(F.db, j.id, F.trainingRoot, worker);
    assert.ok(admitted);
    const launch = await launchRun({
      db: F.db, toolkitRoot: TOOLKIT_ROOT, helperPath: pc.helperPath, helperPython: PYTHON, jobPython: PYTHON,
      runPy: path.join(__dirname, 'fixtures', 'fake_run.py'), sqliteDbPath: F.file, inFlight: new Set(),
    }, admitted);
    assert.equal(launch.result, 'attached');
    const job = await row(j.id);
    if (action === 'force') {
      const result = await routeCall(kill, 'GET', j.id);
      assert.ok([200, 202].includes(result.status), JSON.stringify(result.body));
    } else {
      await routeCall(stop, 'GET', j.id);
      await reconcile({ db: F.db, procctl: pc, self: worker, inFlight: new Set(), trainingRoot: F.trainingRoot, tick: 0,
        now: () => Date.now() + 120_000 });
    }
    const exit = runFiles(job.run_dir, job.run_id).exit;
    const deadline = Date.now() + 5000;
    while (!fs.existsSync(exit) && Date.now() < deadline) await delay(50);
    assert.ok(fs.existsSync(exit));
    await poll();
    assert.equal((await row(j.id)).status, 'stopped');
    assert.equal(await F.db.runClaim.count({ where: { job_id: j.id } }), 0);
  }
});
