// Launch/reconcile/force-stop with REAL processes (Windows): the actual launch.ts, reconcile.ts, forceStop.ts and kill
// route drive the actual toolkit/ui_procctl.py relay and job-object supervisor, which run ui/tests/fixtures/fake_run.py
// (CPU only, hidden, every process waits on a gate file owned by the test). Database: isolated Prisma + temporary
// SQLite. Every process a test starts is in its run's job object; cleanup terminates only those job objects.
//   node --test --test-concurrency=1 ui/tests/launch.test.cjs   (environment: docs/STATE_MACHINES.md section 18)
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const { setupDb, makeJob, ts, jobConfig } = require('./harness/fixture.cjs');
const { PYTHON, TOOLKIT_ROOT, SCRATCH } = require('./harness/env.cjs');

const windows = process.platform === 'win32';
let F, store, launch, recon, procctlMod, forceStop, killRoute, pc, me;
const startedRuns = new Set();
const identityLog = path.join(SCRATCH, 'launch-test-identities.log');
const gateDir = () => fs.mkdtempSync(path.join(SCRATCH, 'gates-'));
const open = gate => fs.writeFileSync(gate, '');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const PYTHONW = path.join(path.dirname(PYTHON), 'pythonw.exe');

before(async () => {
  if (!windows) return;
  F = await setupDb('launch');
  store = require(ts.uiPath('cron/ownership/store.ts'));
  launch = require(ts.uiPath('cron/ownership/launch.ts'));
  recon = require(ts.uiPath('cron/ownership/reconcile.ts'));
  procctlMod = require(ts.uiPath('cron/ownership/procctl.ts'));
  forceStop = require(ts.uiPath('cron/ownership/forceStop.ts')).forceStop;
  pc = procctlMod.makeProcCtl(PYTHON, TOOLKIT_ROOT);
  me = await pc.self(process.pid);
  assert.ok(me?.identity && me.boot_id, 'own identity and boot id');
});

after(async () => {
  if (!windows) return;
  // terminate only this file's run boundaries, then confirm with a complete scan that nothing carries their run ids
  for (const runId of startedRuns) {
    const t = await pc.terminate(runId, null);
    fs.appendFileSync(identityLog, `${new Date().toISOString()} cleanup ${runId} ${JSON.stringify(t)}\n`);
  }
  for (const runId of startedRuns) {
    const res = await pc.probe({ scans: { s: { needle: runId, since_ms: 0 } } });
    assert.equal(res.scans.s.matches.length, 0, `processes of ${runId} left behind`);
  }
  await F.db.$disconnect();
});

const self = () => ({ pid: process.pid, identity: me.identity, boot_id: me.boot_id });
let gpu = 0;
const admitted = async (steps, name = `L${gpu}`) => {
  const cfg = jobConfig(name);
  cfg.config.process[0].test_steps = steps;
  const j = await makeJob(F.db, { name, gpu_ids: String(gpu++), status: 'queued', job_config: JSON.stringify(cfg) });
  const r = await store.admit(F.db, j.id, F.trainingRoot, self());
  assert.ok(r.admitted, r.reason);
  startedRuns.add(r.admitted.run_id);
  return r.admitted;
};
const ctx = (extra = {}) => ({
  db: F.db,
  toolkitRoot: TOOLKIT_ROOT,
  helperPath: path.join(TOOLKIT_ROOT, 'toolkit', 'ui_procctl.py'),
  helperPython: PYTHON,
  jobPython: PYTHONW,
  runPy: path.join(__dirname, 'fixtures', 'fake_run.py'),
  sqliteDbPath: F.file,
  env: { CUDA_VISIBLE_DEVICES: '-1' },
  inFlight: new Set(),
  ...extra,
});
const reconcile = (extra = {}) =>
  recon.reconcile({ db: F.db, procctl: pc, self: self(), inFlight: new Set(), trainingRoot: F.trainingRoot, tick: 0, ...extra });
const row = id => F.db.job.findUnique({ where: { id } });
const claims = id => F.db.runClaim.count({ where: { job_id: id, kind: 'run' } });
const logIds = async (label, j) => {
  const r = await row(j.id);
  fs.appendFileSync(identityLog, `${new Date().toISOString()} ${label} run=${r.run_id} root=${r.pid_identity} supervisor=${r.supervisor_identity}\n`);
};
// waits (bounded only so a broken test fails instead of hanging) for the reconciler to reach a state
const until = async (fn, label, ms = 30000) => {
  const end = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > end) throw new Error(`timed out waiting for ${label}`);
    await sleep(100);
  }
};

test('real launch: attached from the receipt; an early Python "completed" keeps the GPU until the supervisor proves the job object empty', { skip: !windows }, async () => {
  const g = gateDir();
  const j = await admitted(['status:running', 'status:completed', `wait:${g}/exit`]);
  const res = await launch.launchRun(ctx(), j);
  assert.equal(res.result, 'attached', JSON.stringify(res));
  await logIds('early-completed', j);
  let r = await row(j.id);
  assert.equal(r.launch_phase, 'attached');
  assert.equal(r.run_boundary, 'win-job');
  assert.match(r.supervisor_identity, /^\d+@\d+$/);
  await until(async () => (await row(j.id)).status === 'completed', 'fenced completed write');
  const next = await makeJob(F.db, { name: 'L-next', gpu_ids: r.gpu_ids, status: 'queued' });
  for (let i = 0; i < 3; i++) await reconcile({ tick: i });
  assert.equal(await claims(j.id), 2, 'claims held while the process lives');
  assert.equal((await store.admit(F.db, next.id, F.trainingRoot, self())).admitted, null);
  open(`${g}/exit`);
  await until(async () => fs.existsSync(path.join(r.run_dir, `.run.${r.run_id}.exit.json`)), 'exit receipt');
  const rep = await reconcile();
  assert.match(rep[r.run_id], /released: exit receipt -> completed/);
  r = await row(j.id);
  assert.equal(r.status, 'completed');
  assert.equal(await claims(j.id), 0);
  const nextAdmit = await store.admit(F.db, next.id, F.trainingRoot, self());
  assert.ok(nextAdmit.admitted);
  await store.releaseRun(F.db, next.id, nextAdmit.admitted.run_id, { phase: 'claimed' }, { kind: 'not_spawned', info: 'test' });
});

test('intermediate exits before the first observation, live grandchild: the job object still owns it (Codex counterexample 2, real processes)', { skip: !windows }, async () => {
  const g = gateDir();
  const j = await admitted([`orphan:${g}/gc`]); // root starts intermediate -> grandchild, intermediate exits, root exits
  assert.equal((await launch.launchRun(ctx(), j)).result, 'attached');
  await logIds('orphan', j);
  const r = await row(j.id);
  // wait until the root itself is gone, then reconcile for the first time
  await until(async () => !procctlMod.pidExists(r.pid), 'root exit');
  const rep = await reconcile();
  assert.match(rep[r.run_id], /kept: open boundary, [1-9]\d* active, supervisor alive/);
  assert.equal(await claims(j.id), 2);
  open(`${g}/gc`);
  await until(async () => fs.existsSync(path.join(r.run_dir, `.run.${r.run_id}.exit.json`)), 'exit receipt');
  assert.match((await reconcile())[r.run_id], /released: exit receipt -> error/, 'no Python ending: died');
});

test('a child created after the first observation is still in the run (delayed child creation)', { skip: !windows }, async () => {
  const g = gateDir();
  const j = await admitted([`wait:${g}/spawn`, `child:${g}/child`]);
  assert.equal((await launch.launchRun(ctx(), j)).result, 'attached');
  await logIds('delayed-child', j);
  const r = await row(j.id);
  await reconcile();
  open(`${g}/spawn`);
  await until(async () => !procctlMod.pidExists(r.pid), 'root exit after spawning');
  const st = (await pc.probe({ runs: { k: { run_id: r.run_id, supervisor: r.supervisor_identity, members: [r.pid_identity] } } })).runs.k;
  assert.equal(st.boundary, 'open');
  assert.ok(st.active >= 1, JSON.stringify(st));
  assert.equal(st.members[r.pid_identity], 'exited');
  assert.match((await reconcile())[r.run_id], /kept/);
  open(`${g}/child`);
  await until(async () => fs.existsSync(path.join(r.run_dir, `.run.${r.run_id}.exit.json`)), 'exit receipt');
  assert.match((await reconcile())[r.run_id], /released/);
});

test('receipt publication failure: the supervisor terminates the whole job object (incl. an existing child) and verifies it empty', { skip: !windows }, async () => {
  const g = gateDir();
  const j = await admitted([`child:${g}/never`, 'sleep:60']);
  const res = await launch.launchRun(ctx({ hooks: { relayEnv: { AITK_PROCCTL_FAULT: 'receipt_write' } } }), j);
  assert.equal(res.result, 'not_spawned', JSON.stringify(res));
  const r = await row(j.id);
  assert.equal(r.status, 'error');
  assert.equal(await claims(j.id), 0);
  // observation only (a scan is never proof, and with unqueryable system processes it is incomplete): the
  // supervisor itself saw the job object's active count reach 0 before reporting EXIT_RECEIPT_FAILED
  const scan = (await pc.probe({ scans: { s: { needle: r.run_id, since_ms: 0 } } })).scans.s;
  assert.deepEqual(scan.matches.filter(m => !/bash\.exe/.test(m.cmdline)), [], 'nothing of the run survives');
});

test('known pre-spawn failures release: program cannot start (failed receipt), run.py missing, relay cannot start', { skip: !windows }, async () => {
  const a = await admitted([]);
  assert.equal((await launch.launchRun(ctx({ hooks: { relayEnv: { AITK_PROCCTL_FAULT: 'before_spawn' } } }), a)).result, 'not_spawned');
  assert.equal(await claims(a.id), 0);
  assert.match((await row(a.id)).info, /Error launching job/);
  const b = await admitted([]);
  assert.equal((await launch.launchRun(ctx({ runPy: path.join(SCRATCH, 'no-run.py') }), b)).result, 'not_spawned');
  assert.equal((await row(b.id)).launch_phase, 'exited');
  const c = await admitted([]);
  assert.equal((await launch.launchRun(ctx({ helperPython: path.join(SCRATCH, 'no-python.exe') }), c)).result, 'not_spawned');
  assert.equal(await claims(c.id), 0);
});

test('a live relay delayed beyond the former 30 s timeout keeps ownership; a restarted worker cannot release it; it attaches when the receipt appears', { skip: !windows, timeout: 120000 }, async () => {
  const g = gateDir();
  const j = await admitted([`wait:${g}/exit`]);
  const inFlight = new Set([j.run_id]);
  const pending = launch.launchRun(ctx({ inFlight, hooks: { relayEnv: { AITK_PROCCTL_FAULT: 'hang_after_spawn', AITK_PROCCTL_GATE: `${g}/receipt` } } }), j);
  let settled = false;
  pending.then(() => (settled = true));
  await until(async () => (await row(j.id)).launch_phase === 'spawn_requested' && (await row(j.id)).relay_pid != null, 'relay spawned');
  // this worker: in flight
  assert.equal((await reconcile({ inFlight }))[j.run_id], 'launch in flight');
  // the worker "dies" before anything is recorded: admitted by a now-dead worker, relay pid unknown
  const dead = spawnSync(PYTHON, ['-c', 'import os; print(os.getpid())'], { windowsHide: true });
  const deadPid = Number(String(dead.stdout).trim());
  await F.db.job.update({ where: { id: j.id }, data: { launcher_pid: deadPid, launcher_identity: `${deadPid}@1`, relay_pid: null } });
  const newWorker = { inFlight: new Set(), self: { pid: 1, identity: 'other-worker@1', boot_id: me.boot_id } };
  assert.match((await reconcile(newWorker))[j.run_id], /kept: supervisor alive/);
  await sleep(31000); // one hold past the old RELAY_PID_TIMEOUT_MS; nothing here is time-based any more
  assert.match((await reconcile(newWorker))[j.run_id], /kept: supervisor alive/);
  assert.equal(settled, false, 'the launch is still waiting for the real relay exit');
  assert.equal(await claims(j.id), 2);
  open(`${g}/receipt`);
  const res = await pending;
  assert.equal(res.result, 'attached');
  await logIds('delayed-relay', j);
  open(`${g}/exit`);
  const r = await row(j.id);
  await until(async () => fs.existsSync(path.join(r.run_dir, `.run.${r.run_id}.exit.json`)), 'exit receipt');
  assert.match((await reconcile())[r.run_id], /released/);
});

test('DB write failure after spawn (the attach is lost): launch reports uncertain, claims kept; the reconciler attaches from the durable receipt', { skip: !windows }, async () => {
  const g = gateDir();
  const j = await admitted([`wait:${g}/exit`]);
  // the launching worker's attach transaction fails after the child exists
  const failAttach = new Proxy(F.db, {
    get(t, p) {
      if (p !== '$transaction') return t[p];
      return (fn, opts) =>
        t.$transaction(
          tx =>
            fn(
              new Proxy(tx, {
                get(tt, pp) {
                  if (pp !== 'job') return tt[pp];
                  return new Proxy(tt.job, {
                    get(jj, k) {
                      if (k !== 'updateMany') return jj[k];
                      return args => (args?.data?.launch_phase === 'attached' ? Promise.reject(new Error('injected DB failure after spawn')) : jj.updateMany(args));
                    },
                  });
                },
              }),
            ),
          opts,
        );
    },
  });
  const res = await launch.launchRun(ctx({ db: failAttach }), j);
  assert.equal(res.result, 'uncertain', JSON.stringify(res));
  let r = await row(j.id);
  assert.equal(r.launch_phase, 'spawn_requested');
  assert.equal(await claims(j.id), 2, 'claims kept while the outcome is unknown');
  assert.match((await reconcile())[j.run_id], /attached from receipt/);
  r = await row(j.id);
  assert.equal(r.launch_phase, 'attached');
  await logIds('receipt-attach', j);
  open(`${g}/exit`);
  await until(async () => fs.existsSync(path.join(r.run_dir, `.run.${r.run_id}.exit.json`)), 'exit receipt');
  assert.match((await reconcile())[r.run_id], /released/);
});

test('worker death before spawn (claimed by a real exited worker process): released, nothing spawned', { skip: !windows }, async () => {
  const cfg = jobConfig('L-claimed');
  const j = await makeJob(F.db, { name: 'L-claimed', gpu_ids: '90', status: 'queued', job_config: JSON.stringify(cfg) });
  const dead = spawnSync(PYTHON, ['-c', 'import os; print(os.getpid())'], { windowsHide: true });
  const deadPid = Number(String(dead.stdout).trim());
  const r = await store.admit(F.db, j.id, F.trainingRoot, { pid: deadPid, identity: `${deadPid}@133000000000000000` });
  assert.match((await reconcile())[r.admitted.run_id], /released: admitting worker gone before spawn/);
});

test('force stop through the actual kill route: TerminateJobObject, exit receipt, released as stopped; a stale identity never terminates anything', { skip: !windows }, async () => {
  const g = gateDir();
  const j = await admitted([`child:${g}/never`, `wait:${g}/never`]);
  assert.equal((await launch.launchRun(ctx(), j)).result, 'attached');
  await logIds('kill-route', j);
  // a terminate request for a run id with no job object touches nothing
  const none = await pc.terminate(`no-such-run-${Date.now()}`, null);
  assert.equal(none.verified, false);
  assert.equal(none.boundary, 'absent');
  // identity check of the live root with a wrong creation time: 'exited' semantics (a different process), no signal
  const r = await row(j.id);
  const wrong = `${r.pid}@1`;
  const st = (await pc.probe({ runs: { k: { run_id: r.run_id, supervisor: r.supervisor_identity, members: [wrong, r.pid_identity] } } })).runs.k;
  assert.equal(st.members[wrong], 'exited');
  assert.equal(st.members[r.pid_identity], 'alive');
  // the route's production path resolution (TOOLKIT_ROOT from cwd, .venv python) points elsewhere in a test worktree
  ts.mock('@/paths', { TOOLKIT_ROOT });
  ts.mock('../../../../../../cron/pythonPath', { resolvePythonPath: () => PYTHON });
  killRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/kill/route.ts'));
  const res = await killRoute.GET({}, { params: Promise.resolve({ jobID: j.id }) });
  const body = await res.json();
  assert.equal(res.status, 200, JSON.stringify(body));
  assert.equal(body.killed, true);
  await until(async () => (await claims(j.id)) === 0 || (await reconcile()) && (await claims(j.id)) === 0, 'release');
  const after = await row(j.id);
  assert.equal(after.status, 'stopped');
  assert.equal(after.launch_phase, 'exited');
});

test('supervisor dies after creating the root, before any receipt, with a tokenless grandchild alive: claims KEPT (no absence proof, decision rev2)', { skip: !windows }, async () => {
  const g = gateDir();
  const j = await admitted([`child:${g}/never`, `wait:${g}/never`]); // the child's command line carries no run id
  const pending = launch.launchRun(ctx({ hooks: { relayEnv: { AITK_PROCCTL_FAULT: 'crash_before_receipt', AITK_PROCCTL_GATE: `${g}/crash` } } }), j);
  // the launch waits on the relay, which waits on the supervisor: wait until the root has started its child and that
  // child is observed as a live member of the run's job object, then crash the supervisor
  const childPid = Number(await until(async () => fs.existsSync(`${g}/never.started`) && fs.readFileSync(`${g}/never.started`, 'utf8'), 'child started'));
  const r1 = await row(j.id);
  const claimed = JSON.parse(fs.readFileSync(launch.runFiles(r1.run_dir, r1.run_id).boundary, 'utf8'));
  const st = (await pc.probe({ runs: { k: { run_id: r1.run_id, supervisor: claimed.supervisor, members: [] } } })).runs.k;
  assert.equal(st.boundary, 'open');
  assert.ok(st.live.some(m => m.startsWith(`${childPid}@`)), `child ${childPid} is in the job: ${JSON.stringify(st.live)}`);
  open(`${g}/crash`);
  const res = await pending;
  assert.equal(res.result, 'uncertain', JSON.stringify(res));
  const r0 = await until(async () => {
    const row = await F.db.job.findUnique({ where: { id: j.id } });
    return row.relay_exited_at ? row : null;
  }, 'relay exit');
  const files = launch.runFiles(r0.run_dir, r0.run_id);
  assert.equal(launch.readReceipt(files.receipt, r0.run_id), null, 'no receipt was written');
  const claim = JSON.parse(fs.readFileSync(files.boundary, 'utf8'));
  assert.equal(claim.kind, 'supervisor');
  const rep = await reconcile();
  assert.match(rep[r0.run_id], /kept: supervisor exited/);
  assert.equal(await claims(j.id), 2);
  assert.match((await row(j.id)).ownership_note, /restart of this computer/);
  // the kernel did terminate the run (KILL_ON_JOB_CLOSE) — observed here for the record, never used as proof
  const scan = (await pc.probe({ scans: { s: { needle: r0.run_id, since_ms: 0 } } })).scans.s;
  fs.appendFileSync(identityLog, `${new Date().toISOString()} crash-before-receipt run=${r0.run_id} remaining-token-matches=${scan.matches.length}\n`);
});

test('a supervisor delayed before its boundary claim loses to the reconciler fence and starts nothing', { skip: !windows }, async () => {
  const g = gateDir();
  const j = await admitted([`wait:${g}/never`]);
  const pending = launch.launchRun(ctx({ hooks: { relayEnv: { AITK_PROCCTL_FAULT: 'delay_before_claim', AITK_PROCCTL_GATE: `${g}/claim` } } }), j);
  await until(async () => (await row(j.id)).relay_pid != null, 'relay spawned');
  // the admitting worker "died": a new worker reconciles, finds no claim, proves the launcher gone, fences
  const dead = spawnSync(PYTHON, ['-c', 'import os; print(os.getpid())'], { windowsHide: true });
  const deadPid = Number(String(dead.stdout).trim());
  await F.db.job.update({ where: { id: j.id }, data: { launcher_pid: deadPid, launcher_identity: `${deadPid}@1` } });
  const rep = await reconcile({ self: { pid: 1, identity: 'other-worker@1', boot_id: me.boot_id } });
  assert.match(rep[j.run_id], /released: fenced/);
  assert.equal(await claims(j.id), 0);
  open(`${g}/claim`); // the slow supervisor now tries to claim
  const res = await pending;
  assert.equal(res.result, 'not_spawned', JSON.stringify(res));
  const r = await row(j.id);
  const receipt = launch.readReceipt(launch.runFiles(r.run_dir, r.run_id).receipt, r.run_id);
  assert.equal(receipt.state, 'failed');
  assert.match(receipt.error, /already claimed \(fence\)/);
  const scan = (await pc.probe({ scans: { s: { needle: r.run_id, since_ms: 0 } } })).scans.s;
  assert.deepEqual(scan.matches.filter(m => !/ui_procctl\.py/.test(m.cmdline)), [], 'no root was ever started');
});

test('unsupported platform: refused before anything is spawned', { skip: !windows }, async () => {
  const j = await admitted([]);
  const res = await launch.launchRun(ctx({ platform: 'darwin' }), j);
  assert.equal(res.result, 'not_spawned');
  assert.match(res.info, /not supported on darwin/);
  const r = await row(j.id);
  assert.equal(r.relay_pid, null);
  assert.equal(r.launch_phase, 'exited');
  assert.equal(await claims(j.id), 0);
});
