// Run ownership on a real temporary SQLite database with an isolated generated Prisma client
// (docs/STATE_MACHINES.md M10, regression plan U-R1..U-R9). The actual route modules and worker modules run;
// the OS process layer is scripted here (fakeProcCtl). Real processes are exercised in launch.test.cjs.
//   node --test --test-concurrency=1 ui/tests/ownership.test.cjs   (environment: docs/STATE_MACHINES.md section 18)
const { test, before, beforeEach, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { setupDb, reset, makeJob, fakeProcCtl, routeCall, ts } = require('./harness/fixture.cjs');

let F, store, recon, startRoute, stopRoute, markRoute, deleteRoute, jobsRoute, queue, forceStop, resources;
const self = { pid: process.pid, identity: 'self@1', boot_id: 'boot-test-1' };

before(async () => {
  F = await setupDb('ownership');
  store = require(ts.uiPath('cron/ownership/store.ts'));
  recon = require(ts.uiPath('cron/ownership/reconcile.ts'));
  resources = require(ts.uiPath('cron/ownership/resources.ts'));
  forceStop = require(ts.uiPath('cron/ownership/forceStop.ts')).forceStop;
  queue = require(ts.uiPath('cron/actions/processQueue.ts'));
  startRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/start/route.ts'));
  stopRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/stop/route.ts'));
  markRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/mark_stopped/route.ts'));
  deleteRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/delete/route.ts'));
  jobsRoute = require(ts.uiPath('src/app/api/jobs/route.ts'));
});
beforeEach(async () => reset(F.db));
after(async () => F.db.$disconnect());

const job = id => F.db.job.findUnique({ where: { id } });
const claimsOf = jobId => F.db.runClaim.findMany({ where: { job_id: jobId }, orderBy: { resource: 'asc' } });
const admit = (j, db = F.db, hooks) => store.admit(db, j.id, F.trainingRoot, self, hooks);
const reconcile = (procctl, extra = {}) =>
  recon.reconcile({ db: F.db, procctl, self, inFlight: new Set(), trainingRoot: F.trainingRoot, tick: 1, ...extra });

// scripted run-boundary states, keyed by supervisor identity
const alive = (root, live = [root]) => ({ supervisor: 'alive', boundary: 'open', active: live.length, live, members: { [root]: 'alive' } });
const goneAbsent = (...ids) => ({ supervisor: 'exited', boundary: 'absent', live: [], members: Object.fromEntries(ids.map(i => [i, 'exited'])) });
const writeExitReceipt = (j, extra = {}) => {
  fs.mkdirSync(j.run_dir, { recursive: true });
  fs.writeFileSync(
    path.join(j.run_dir, `.run.${j.run_id}.exit.json`),
    JSON.stringify({ run_id: j.run_id, supervisor: j.supervisor_identity, state: 'exited', evidence: 'job object ActiveProcesses == 0', ...extra }),
  );
};

// admitted + attached run: root rootOf(name) (a numeric pid identity), supervisor 'name-sup@1'
const rootOf = name => `${[...name].reduce((a, c) => a * 31 + c.charCodeAt(0), 7) % 100000}@1`;
const runningJob = async (name, gpu_ids = '0') => {
  const j = await makeJob(F.db, { name, gpu_ids, status: 'queued' });
  const { admitted } = await admit(j);
  assert.ok(admitted, `admit ${name}`);
  await F.db.job.update({ where: { id: j.id }, data: { launch_phase: 'spawn_requested', spawn_requested_at: new Date() } });
  assert.equal(await store.attachRun(F.db, j.id, admitted.run_id, { identity: rootOf(name), supervisor: `${name}-sup@1`, boundary: 'win-job' }), true);
  return job(j.id);
};

test('U-R1 Start on starting/running/stopping/deleting/delete_failed is refused; a repeated Start of a queued job is a no-op', async () => {
  for (const status of ['starting', 'running', 'stopping', 'deleting', 'delete_failed']) {
    const j = await makeJob(F.db, { name: `s-${status}`, status, pid: status === 'running' ? 123 : null });
    const before = await job(j.id);
    const r = await routeCall(startRoute, 'GET', j.id);
    assert.equal(r.status, 409, status);
    assert.deepEqual(await job(j.id), before, `${status} row unchanged`);
  }
  const q = await makeJob(F.db, { name: 'q', status: 'queued', queue_position: 5 });
  const r = await routeCall(startRoute, 'GET', q.id);
  assert.equal(r.status, 200);
  assert.equal((await job(q.id)).queue_position, 5, 'no re-queue');
});

test('U-R1 terminal-but-live: Python wrote completed but the run still holds claims -> Start/Edit/Delete/Mark stopped refused', async () => {
  const j = await runningJob('tbl');
  await F.db.job.updateMany({ where: { id: j.id, run_id: j.run_id }, data: { status: 'completed' } }); // fenced Python write
  const start = await routeCall(startRoute, 'GET', j.id);
  assert.equal(start.status, 409);
  assert.match(start.body.error, /not exited yet/);
  const edit = await jobsRoute.POST({ json: async () => ({ id: j.id, name: 'tbl', gpu_ids: '0', job_config: {} }) });
  assert.equal(edit.status, 409);
  assert.equal((await routeCall(deleteRoute, 'GET', j.id)).status, 409);
  assert.equal((await routeCall(markRoute, 'GET', j.id)).status, 409);
  assert.equal((await claimsOf(j.id)).length, 2);
  assert.equal((await job(j.id)).status, 'completed');
});

test('U-R2 two workers (two Prisma clients) admitting intersecting sets "0" and "1,0" concurrently: exactly one admitted', async () => {
  const second = F.newClient();
  try {
    const a = await makeJob(F.db, { name: 'a', gpu_ids: '0', status: 'queued' });
    const b = await makeJob(F.db, { name: 'b', gpu_ids: '1,0', status: 'queued' });
    await second.job.count(); // both clients connected before the race starts
    // Both admissions start at once. Prisma's SQLite transactions take the write lock at BEGIN
    // (prisma-lock-probe), so the second waits for the first and then sees its claims.
    const [ra, rb] = await Promise.all([admit(a), admit(b, second)]);
    const admitted = [ra, rb].filter(r => r.admitted);
    assert.equal(admitted.length, 1, JSON.stringify([ra.reason, rb.reason]));
    const loser = ra.admitted ? b : a;
    assert.equal((await job(loser.id)).status, 'queued');
    assert.match((await job(loser.id)).ownership_note, /GPU 0 in use by job/);
    assert.equal((await job((ra.admitted ?? rb.admitted).id)).status, 'starting');
    const claims = await F.db.runClaim.findMany();
    assert.equal(new Set(claims.map(c => c.run_id)).size, 1);
    assert.ok(claims.map(c => c.resource).includes('gpu:0'));
    for (const r of await F.db.job.findMany({ where: { status: { in: ['starting', 'running'] } } })) {
      assert.ok((await claimsOf(r.id)).length > 0, `${r.name}: never starting without claims`);
    }
  } finally {
    await second.$disconnect();
  }
});

test('U-R2 a claim the occupancy read did not see still cannot be duplicated: the primary-key conflict rolls the whole admission back', async () => {
  const a = await makeJob(F.db, { name: 'pk', gpu_ids: '4', status: 'queued' });
  const r = await admit(a, F.db, {
    afterRead: async tx => {
      await tx.runClaim.create({ data: { resource: 'gpu:4', job_id: 'other', run_id: 'other-run', kind: 'run' } });
    },
  });
  assert.equal(r.admitted, null);
  assert.match(r.reason, /P2002/);
  assert.equal((await job(a.id)).status, 'queued');
  assert.equal(await F.db.runClaim.count(), 0, 'rolled back together, including the injected row');
});

test('U-R2 normalization: "1,0" owns the same GPUs as "0,1"; disjoint sets run together; mps and -1 handled', async () => {
  const x = await makeJob(F.db, { name: 'x', gpu_ids: '1,0', status: 'queued' });
  const y = await makeJob(F.db, { name: 'y', gpu_ids: ' 0 , 1 ', status: 'queued' });
  const z = await makeJob(F.db, { name: 'z', gpu_ids: '2', status: 'queued' });
  const cpu = await makeJob(F.db, { name: 'cpu', gpu_ids: '-1', status: 'queued' });
  const mps = await makeJob(F.db, { name: 'mps', gpu_ids: 'mps', status: 'queued' });
  assert.ok((await admit(x)).admitted);
  assert.equal((await admit(y)).admitted, null);
  assert.ok((await admit(z)).admitted);
  assert.ok((await admit(cpu)).admitted);
  assert.deepEqual((await claimsOf(cpu.id)).map(c => c.resource), [`folder:${resources.canonicalFolder(F.trainingRoot, 'cpu')}`]);
  assert.ok((await admit(mps)).admitted);
});

test('opaque GPU tokens are exclusive across all GPU claims; empty gpu_ids are refused with a message', async () => {
  const uuid = await makeJob(F.db, { name: 'uuid', gpu_ids: 'GPU-1234abcd', status: 'queued' });
  const n = await makeJob(F.db, { name: 'n', gpu_ids: '3', status: 'queued' });
  assert.ok((await admit(uuid)).admitted);
  assert.match((await admit(n)).reason, /all GPUs in use/);
  const empty = await makeJob(F.db, { name: 'empty', gpu_ids: '', status: 'stopped' });
  const r = await routeCall(startRoute, 'GET', empty.id);
  assert.equal(r.status, 400);
  assert.match(r.body.error, /No GPU is selected/);
  const queuedEmpty = await makeJob(F.db, { name: 'qe', gpu_ids: '  ', status: 'queued' });
  assert.equal((await admit(queuedEmpty)).admitted, null);
  assert.equal((await job(queuedEmpty.id)).status, 'error');
  assert.equal((await jobsRoute.POST({ json: async () => ({ name: 'new', gpu_ids: '', job_config: {} }) })).status, 400);
});

test('U-R3 Stop -> stopping; early Python "stopped" does not free the GPU until the exit receipt; then released and the next job admitted', async () => {
  const j = await runningJob('r3');
  const next = await makeJob(F.db, { name: 'r3next', gpu_ids: '0', status: 'queued' });
  assert.equal((await routeCall(stopRoute, 'GET', j.id)).status, 200);
  let row = await job(j.id);
  assert.equal(row.status, 'stopping');
  assert.equal(row.after_exit, 'stopped');
  assert.equal(row.stop, true);
  await F.db.job.updateMany({ where: { id: j.id, run_id: j.run_id }, data: { status: 'stopped', info: 'Job stopped' } }); // Python
  await reconcile(fakeProcCtl({ runs: { 'r3-sup@1': alive(rootOf('r3')) } }));
  assert.equal((await claimsOf(j.id)).length, 2);
  assert.equal((await admit(next)).admitted, null, 'not admitted while the stopped run is alive');
  writeExitReceipt(await job(j.id));
  const rep = await reconcile(fakeProcCtl());
  assert.match(rep[j.run_id], /released: exit receipt -> stopped/);
  row = await job(j.id);
  assert.equal(row.status, 'stopped');
  assert.equal(row.launch_phase, 'exited');
  assert.match(row.ownership_note, /Exit verified/);
  assert.ok((await admit(next)).admitted);
});

test('an exit receipt must be bound to this run AND its supervisor', async () => {
  const j = await runningJob('rb');
  writeExitReceipt(j, { supervisor: 'someone-else@1' });
  await reconcile(fakeProcCtl({ runs: { 'rb-sup@1': { supervisor: 'alive', boundary: 'open', active: 0, live: [], members: {} } } }));
  assert.equal((await claimsOf(j.id)).length, 2);
  writeExitReceipt(j, { run_id: 'other-run' });
  await reconcile(fakeProcCtl({ runs: { 'rb-sup@1': alive(rootOf('rb')) } }));
  assert.equal((await claimsOf(j.id)).length, 2);
});

test('exit codes honor stop and queue requests, reported errors, and older receipts', () => {
  const outcome = (status, after_exit, code) => store.finalStatus({ status, after_exit }, code);
  assert.deepEqual(outcome('stopping', 'stopped', -9), { status: 'stopped', unexpected: false });
  assert.deepEqual(outcome('running', 'queued', 0), { status: 'queued', unexpected: false });
  assert.deepEqual(outcome('error', null, 0), { status: 'error', unexpected: false });
  assert.deepEqual(outcome('completed', null, undefined), { status: 'completed', unexpected: false });
  assert.deepEqual(outcome('running', null, undefined), { status: 'error', unexpected: true });
});

test('a force-stop click after the process failed preserves the failure', async () => {
  const j = await runningJob('already-exited');
  writeExitReceipt(j, { exit_code: 7 });
  const pc = fakeProcCtl();
  const result = await forceStop(F.db, pc, j.id);
  assert.equal(result.body.killed, false);
  assert.equal((await job(j.id)).status, 'error');
  assert.match((await job(j.id)).info, /code 7/);
  assert.equal(pc.calls.terminate.length, 0);
});

test('boundary evidence (decision rev2): only an open+empty job object with its supervisor gone, an exit receipt or a boot change release; a vanished job name never does', async () => {
  const j = await runningJob('desc');
  // root gone, an orphaned grandchild still in the job object
  await reconcile(fakeProcCtl({ runs: { 'desc-sup@1': { supervisor: 'alive', boundary: 'open', active: 1, live: ['3@3'], members: { [rootOf('desc')]: 'exited' } } } }));
  assert.equal((await claimsOf(j.id)).length, 2);
  assert.match((await job(j.id)).ownership_note, /1 remaining process/);
  assert.deepEqual(JSON.parse((await job(j.id)).run_members).sort(), [rootOf('desc'), '3@3'].sort());
  // supervisor gone, boundary unqueryable: fail closed
  await reconcile(fakeProcCtl({ runs: { 'desc-sup@1': { supervisor: 'exited', boundary: 'unknown', live: [], members: {}, error: 'access denied' } } }));
  assert.equal((await claimsOf(j.id)).length, 2);
  assert.match((await job(j.id)).ownership_note, /Mark as Stopped/);
  // supervisor gone, job name absent, EVERY recorded member exited: still not proof (Q4: the name disappears before
  // unrecorded members finish terminating)
  await reconcile(fakeProcCtl({ runs: { 'desc-sup@1': goneAbsent(rootOf('desc'), '3@3') } }));
  assert.equal((await claimsOf(j.id)).length, 2);
  assert.match((await job(j.id)).ownership_note, /Mark as Stopped/);
  // supervisor alive with an empty job: it will write the exit receipt itself; nothing released yet
  await reconcile(fakeProcCtl({ runs: { 'desc-sup@1': { supervisor: 'alive', boundary: 'open', active: 0, live: [], members: {} } } }));
  assert.equal((await claimsOf(j.id)).length, 2);
  await reconcile(fakeProcCtl({ down: true }));
  assert.equal((await claimsOf(j.id)).length, 2);
  assert.match((await job(j.id)).ownership_note, /Cannot check process state/);
  // proof: job object still open, 0 active processes, and the only possible spawner (its supervisor) is gone
  const rep = await reconcile(fakeProcCtl({ runs: { 'desc-sup@1': { supervisor: 'exited', boundary: 'open', active: 0, live: [], members: {} } } }));
  assert.match(rep[j.run_id], /released: empty job object/);
  assert.equal((await job(j.id)).status, 'error', 'no Python ending -> died');
  assert.match((await job(j.id)).info, /exited unexpectedly/);
});

test('a boot id change proves exit for any run phase; an unreadable boot id never does', async () => {
  const j = await runningJob('boot');
  assert.equal((await job(j.id)).boot_id, 'boot-test-1', 'admission records the worker boot id');
  const unknown = { runs: { 'boot-sup@1': { supervisor: 'unknown', boundary: 'unknown', live: [], members: {} } } };
  await reconcile(fakeProcCtl(unknown), { self: { ...self, boot_id: null } });
  assert.equal((await claimsOf(j.id)).length, 2);
  const rep = await reconcile(fakeProcCtl(unknown), { self: { ...self, boot_id: 'boot-test-2' } });
  assert.match(rep[j.run_id], /boot changed/);
  assert.equal((await claimsOf(j.id)).length, 0);
});

test('U-R4 force-stop failure or unsupported platform keeps claims and reports the error; success releases only on the exit receipt', async () => {
  const j = await runningJob('r4');
  const failing = fakeProcCtl({ terminate: () => ({ supported: true, verified: false, remaining: 1 }) });
  const r = await forceStop(F.db, failing, j.id, 50);
  assert.equal(r.status, 500);
  assert.match(r.body.error, /still running/);
  assert.equal((await claimsOf(j.id)).length, 2);
  assert.equal((await job(j.id)).status, 'stopping');
  const unsupported = fakeProcCtl({ terminate: () => ({ supported: false, verified: false, error: 'not supported on darwin' }) });
  assert.equal((await forceStop(F.db, unsupported, j.id, 50)).status, 501);
  assert.equal((await claimsOf(j.id)).length, 2);
  // verified but no receipt yet: claims kept, reconciler finishes later
  const noReceipt = await forceStop(F.db, fakeProcCtl(), j.id, 50);
  assert.equal(noReceipt.status, 200);
  assert.equal((await claimsOf(j.id)).length, 2);
  writeExitReceipt(await job(j.id));
  const ok = await forceStop(F.db, fakeProcCtl(), j.id, 50);
  assert.equal(ok.body.killed, true);
  assert.equal((await job(j.id)).status, 'stopped');
  assert.equal((await claimsOf(j.id)).length, 0);
  assert.equal(fakeProcCtl().calls.terminate.length, 0);
});

test('U-R5 restart in each launch phase (scripted process layer)', async () => {
  let gpu = 10;
  const mk = async (name, launcher, phase = 'spawn_requested') => {
    const j = await makeJob(F.db, { name, gpu_ids: String(gpu++), status: 'queued' });
    assert.ok((await store.admit(F.db, j.id, F.trainingRoot, launcher)).admitted);
    if (phase === 'spawn_requested') await F.db.job.update({ where: { id: j.id }, data: { launch_phase: 'spawn_requested', spawn_requested_at: new Date() } });
    return job(j.id);
  };
  // claimed by another worker that is now gone -> released (nothing was spawned)
  const a = await mk('p-claimed', { pid: 4242, identity: '4242@7' }, 'claimed');
  let rep = await reconcile(fakeProcCtl({ describe: { 4242: { state: 'gone' } } }));
  assert.match(rep[a.run_id], /released/);
  // claimed by another worker still alive -> kept; pid alive with another creation time -> gone -> released
  const b = await mk('p-claimed2', { pid: 4343, identity: '4343@7' }, 'claimed');
  await reconcile(fakeProcCtl({ describe: { 4343: { state: 'alive', identity: '4343@7' } } }));
  assert.equal((await claimsOf(b.id)).length, 2);
  await reconcile(fakeProcCtl({ describe: { 4343: { state: 'alive', identity: '4343@8' } } }));
  assert.equal((await claimsOf(b.id)).length, 0);
  // claimed by THIS worker: released only when no launch is registered for it
  const d = await mk('p-flight', self, 'claimed');
  rep = await reconcile(fakeProcCtl(), { inFlight: new Set([d.run_id]) });
  assert.equal(rep[d.run_id], 'launch in flight');
  rep = await reconcile(fakeProcCtl(), { inFlight: new Set() });
  assert.match(rep[d.run_id], /never spawned/);

  const dead = { pid: 5000, identity: '5000@1' };
  const deadDescribe = { 5000: { state: 'gone' } };
  const claimFile = j => path.join(j.run_dir, `.run.${j.run_id}.boundary.json`);
  const writeClaim = (j, record) => {
    fs.mkdirSync(j.run_dir, { recursive: true });
    fs.writeFileSync(claimFile(j), JSON.stringify({ run_id: j.run_id, ...record }));
  };
  // no supervisor ever claimed the run and its launcher is gone: fenced (no later supervisor can start a root), released
  const e = await mk('p-sr', dead);
  const pcE = fakeProcCtl({ describe: deadDescribe });
  rep = await reconcile(pcE);
  assert.match(rep[e.run_id], /released: fenced/);
  assert.equal(JSON.parse(fs.readFileSync(claimFile(e), 'utf8')).kind, 'fence');
  assert.equal(pcE.calls.fence.length, 1);
  // launcher alive, nothing claimed: kept, and no fence attempted (it could still be launching)
  const g = await mk('p-sr-alive', { pid: 5100, identity: '5100@1' });
  const aliveLauncher = { 5100: { state: 'alive', identity: '5100@1' } };
  const pcG = fakeProcCtl({ describe: aliveLauncher });
  await reconcile(pcG);
  assert.equal((await claimsOf(g.id)).length, 2);
  assert.equal(pcG.calls.fence, undefined);
  // relay exit recorded: nothing can claim any more except a supervisor already on its way, which the fence beats
  await F.db.job.update({ where: { id: g.id }, data: { relay_exited_at: new Date() } });
  await reconcile(fakeProcCtl({ describe: aliveLauncher }));
  assert.equal((await claimsOf(g.id)).length, 0);
  // the fence loses to a supervisor that claimed first: kept; a fence I/O failure: kept
  const k = await mk('p-sr-race', dead);
  const pcK = fakeProcCtl({ describe: deadDescribe, onProbe: async () => writeClaim(k, { kind: 'supervisor', supervisor: '9100@1' }) });
  assert.match((await reconcile(pcK))[k.run_id], /kept: supervisor claimed first/);
  const m = await mk('p-sr-fencefail', dead);
  assert.match((await reconcile(fakeProcCtl({ describe: deadDescribe, fenceError: 'EPERM: injected' })))[m.run_id], /kept: fence failed/);
  assert.equal((await claimsOf(m.id)).length, 2);
  // a supervisor claimed the boundary (no receipt yet)
  const n = await mk('p-sr-claimed', dead);
  writeClaim(n, { kind: 'supervisor', supervisor: '9200@1' });
  await reconcile(fakeProcCtl({ runs: { '9200@1': { supervisor: 'alive', boundary: 'open', active: 1, live: [], members: {} } } }));
  assert.equal((await claimsOf(n.id)).length, 2, 'supervisor alive');
  await reconcile(fakeProcCtl({ runs: { '9200@1': { supervisor: 'exited', boundary: 'absent', live: [], members: {} } } }));
  assert.equal((await claimsOf(n.id)).length, 2, 'supervisor died before its receipt, job name gone: NOT proof (Q4)');
  assert.match((await job(n.id)).ownership_note, /Mark as Stopped/);
  await reconcile(fakeProcCtl({ runs: { '9200@1': { supervisor: 'exited', boundary: 'open', active: 1, live: [], members: {} } } }));
  assert.equal((await claimsOf(n.id)).length, 2, 'members still running');
  assert.match((await reconcile(fakeProcCtl({ runs: { '9200@1': { supervisor: 'exited', boundary: 'open', active: 0, live: [], members: {} } } })))[n.run_id], /released: supervisor gone, job object empty/);
  // post-spawn DB failure: the attach never happened but the receipt is durable -> reconciler attaches from it
  const h = await mk('p-sr-receipt', dead);
  fs.mkdirSync(h.run_dir, { recursive: true });
  fs.writeFileSync(path.join(h.run_dir, `.run.${h.run_id}.receipt.json`), JSON.stringify({ run_id: h.run_id, state: 'spawned', pid: 8000, identity: '8000@1', supervisor: '8001@1', boundary: 'win-job', boot_id: 'boot-test-1' }));
  await reconcile(fakeProcCtl());
  let row = await job(h.id);
  assert.equal(row.launch_phase, 'attached');
  assert.equal(row.pid_identity, '8000@1');
  assert.equal(row.supervisor_identity, '8001@1');
  // receipts for another run id, or with a None identity, are ignored (never attached)
  for (const bad of [{ run_id: 'someone-else', state: 'spawned', identity: '1@1', supervisor: '2@2' }, { state: 'spawned', identity: '1@None', supervisor: '2@2' }]) {
    const i = await mk(`p-bad-${gpu}`, { pid: 5100, identity: '5100@1' });
    fs.mkdirSync(i.run_dir, { recursive: true });
    fs.writeFileSync(path.join(i.run_dir, `.run.${i.run_id}.receipt.json`), JSON.stringify({ run_id: i.run_id, ...bad }));
    await reconcile(fakeProcCtl({ describe: aliveLauncher }));
    row = await job(i.id);
    assert.equal(row.launch_phase, 'spawn_requested');
    assert.equal(row.pid_identity, null);
  }
});

test('phase race (Codex reconcile-phase-race-probe): "claimed" evidence cannot release a run whose launcher meanwhile committed spawn_requested', async () => {
  const j = await makeJob(F.db, { name: 'phase', gpu_ids: '30', status: 'queued' });
  const { admitted } = await store.admit(F.db, j.id, F.trainingRoot, { pid: 4444, identity: '4444@1' });
  const pc = fakeProcCtl({
    describe: { 4444: { state: 'gone' } },
    // while the reconciler is collecting evidence, the launcher advances the phase and spawns
    onProbe: async () => {
      await F.db.job.updateMany({ where: { id: j.id, run_id: admitted.run_id, launch_phase: 'claimed' }, data: { launch_phase: 'spawn_requested', spawn_requested_at: new Date() } });
    },
  });
  const rep = await reconcile(pc);
  assert.match(rep[admitted.run_id], /kept: phase is spawn_requested, evidence was for claimed/);
  assert.equal((await claimsOf(j.id)).length, 2);
  assert.equal((await job(j.id)).launch_phase, 'spawn_requested');
});

test('U-R6 a stale watcher/late write for run A cannot change run B; evidence for an old identity cannot release', async () => {
  const j = await runningJob('r6');
  const runA = j.run_id;
  writeExitReceipt(j);
  await reconcile(fakeProcCtl());
  await routeCall(startRoute, 'GET', j.id);
  const { admitted } = await admit(await job(j.id));
  const runB = admitted.run_id;
  assert.notEqual(runA, runB);
  assert.equal(await store.attachRun(F.db, j.id, runA, { identity: 'stale@1', supervisor: 's@1' }), false);
  const stale = await store.releaseRun(F.db, j.id, runA, { phase: 'attached' }, { kind: 'exited' });
  assert.equal(stale.released, false);
  const fenced = await F.db.job.updateMany({ where: { id: j.id, run_id: runA }, data: { status: 'error' } });
  assert.equal(fenced.count, 0, 'a fenced Python write from run A matches nothing');
  let row = await job(j.id);
  assert.equal(row.run_id, runB);
  assert.equal(row.status, 'starting');
  assert.equal((await claimsOf(j.id)).length, 2);
  await F.db.job.update({ where: { id: j.id }, data: { launch_phase: 'spawn_requested' } });
  await store.attachRun(F.db, j.id, runB, { identity: 'r6b@1', supervisor: 'r6b-sup@1' });
  const wrongIdentity = await store.releaseRun(F.db, j.id, runB, { phase: 'attached', identity: 'r6a@1' }, { kind: 'exited' });
  assert.equal(wrongIdentity.released, false);
  assert.equal((await claimsOf(j.id)).length, 2);
});

test('stop backstop: a run alive long after the stop request gets ONE force stop (through the run boundary) per retry window', async () => {
  const j = await runningJob('bs');
  await routeCall(stopRoute, 'GET', j.id);
  await F.db.job.update({ where: { id: j.id }, data: { stop_requested_at: new Date(Date.now() - 120_000) } });
  const pc = fakeProcCtl({ runs: { 'bs-sup@1': alive(rootOf('bs')) } });
  const attempts = new Map();
  await reconcile(pc, { backstopAttempts: attempts });
  await reconcile(pc, { backstopAttempts: attempts });
  assert.equal(pc.calls.terminate.length, 1);
  assert.deepEqual(pc.calls.terminate[0], { runId: j.run_id, supervisor: 'bs-sup@1' });
  assert.equal((await claimsOf(j.id)).length, 2, 'termination is not exit proof');
});

test('an update releases stale legacy jobs on the first worker tick, preserves outputs and starts the queued job', async () => {
  const old = new Date(Date.now() - 86_400_000);
  const done = await makeJob(F.db, { name: 'done', status: 'completed', pid: 101, step: 100, updated_at: old });
  const stale = await makeJob(F.db, { name: 'stale', status: 'running', pid: 102, updated_at: old });
  const noPid = await makeJob(F.db, { name: 'no-pid', status: 'running', pid: null, updated_at: old });
  const next = await makeJob(F.db, { name: 'next', status: 'queued' });
  const output = path.join(F.trainingRoot, done.name, 'checkpoint.safetensors');
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, 'checkpoint');
  await F.db.queue.create({ data: { gpu_ids: '0', is_running: true } });
  const pc = fakeProcCtl({
    describe: { 101: { state: 'gone' }, 102: { state: 'alive', started_ms: Date.now(), cmdline: 'unrelated' } },
    scans: { '.job_config.json': { complete: true, matches: [] } },
  });
  await reconcile(pc);
  assert.equal((await job(done.id)).status, 'completed');
  assert.equal((await job(done.id)).step, 100);
  assert.equal((await job(done.id)).pid, null);
  assert.equal((await job(stale.id)).status, 'stopped');
  assert.equal((await job(noPid.id)).status, 'stopped');
  assert.equal(fs.readFileSync(output, 'utf8'), 'checkpoint');
  const launched = [];
  await queue.processQueueWith({ db: F.db, trainingRoot: F.trainingRoot, launcher: self, launch: j => launched.push(j.id) });
  assert.deepEqual(launched, [next.id]);
  assert.equal((await routeCall(deleteRoute, 'GET', done.id)).status, 200);
  assert.equal(fs.existsSync(output), false);
  assert.equal(pc.calls.terminate.length, 0);
});

test('live legacy trainers stay reserved with or without a PID; failed observations have a manual recovery', async () => {
  const { checkStopped } = require(ts.uiPath('cron/ownership/recoverStopped.ts'));
  const old = new Date(Date.now() - 86_400_000);
  const a = await makeJob(F.db, { name: 'live', status: 'running', pid: 444, updated_at: old });
  const b = await makeJob(F.db, { name: 'legacy orphan', status: 'running', pid: null, updated_at: old });
  const c = await makeJob(F.db, { name: 'unknown', status: 'stopping', pid: 445, updated_at: old });
  const cmd = j => 'python run.py "' + path.join(F.trainingRoot, j.name, '.job_config.json') + '"';
  const pc = fakeProcCtl({
    describe: { 444: { state: 'alive', cmdline: cmd(a) }, 445: { state: 'denied' } },
    scans: { '.job_config.json': { complete: false, matches: [{ identity: '71@1', cmdline: cmd(b) }] } },
  });
  await reconcile(pc);
  for (const j of [a, b]) {
    assert.equal((await job(j.id)).status, 'running');
    assert.equal((await checkStopped(F.db, pc, j.id, j)).status, 409);
  }
  assert.match((await job(c.id)).ownership_note, /Mark as Stopped/);
  assert.equal((await checkStopped(F.db, pc, c.id, c)).status, 200);
  assert.equal((await store.markStopped(F.db, c.id, c)).status, 200);
  assert.equal((await job(c.id)).pid, null);
  assert.equal((await routeCall(deleteRoute, 'GET', c.id)).status, 200);
  assert.equal(pc.calls.terminate.length, 0);
});

test('manual recovery preserves live runs and rejects a stale confirmation after a relaunch', async () => {
  const { checkStopped } = require(ts.uiPath('cron/ownership/recoverStopped.ts'));
  const j = await runningJob('manual');
  const pc = fakeProcCtl({ runs: { 'manual-sup@1': alive(rootOf('manual')) } });
  assert.equal((await checkStopped(F.db, pc, j.id, j)).status, 409);
  pc.script.runs['manual-sup@1'] = { supervisor: 'exited', boundary: 'absent', live: [], members: { [rootOf('manual')]: 'exited' } };
  assert.equal((await checkStopped(F.db, pc, j.id, j)).status, 200);
  assert.equal((await store.markStopped(F.db, j.id, j)).status, 200);
  assert.equal((await claimsOf(j.id)).length, 0);
  assert.equal((await job(j.id)).run_id, null);
  await store.requestStart(F.db, j.id, F.trainingRoot);
  const newer = (await admit(j)).admitted;
  assert.ok(newer);
  assert.equal((await store.markStopped(F.db, j.id, j)).status, 409);
  assert.equal((await store.deleteJob(F.db, j.id, F.trainingRoot, { seen: j })).status, 409);
  assert.equal((await job(j.id)).run_id, newer.run_id);
  assert.equal((await claimsOf(j.id)).length, 2);
});

test('U-R7 migration compatibility: the ORIGINAL schema pushed to a database with rows, then this schema pushed without data loss', async () => {
  const { isolatedPrisma, pushSchema, tempDbFile, clientFor } = require('./harness/prisma.cjs');
  const original = fs.readFileSync(path.join(__dirname, 'fixtures', 'schema-f6c137b4.prisma'), 'utf8');
  const old = isolatedPrisma(original, 'original');
  const file = tempDbFile('migrate');
  pushSchema(old.schemaPath, file);
  const oldClient = clientFor(old.PrismaClient, file);
  await oldClient.job.create({ data: { name: 'legacy', gpu_ids: '0', job_config: '{}', status: 'running', pid: 4321 } });
  await oldClient.$disconnect();
  const cur = isolatedPrisma();
  pushSchema(cur.schemaPath, file); // no --accept-data-loss
  const c = clientFor(cur.PrismaClient, file);
  try {
    const row = await c.job.findFirst({ where: { name: 'legacy' } });
    assert.equal(row.pid, 4321);
    assert.equal(row.status, 'running');
    assert.equal(row.run_id, null);
    assert.equal(store.isLegacyOccupant(row), true);
    assert.equal(await c.runClaim.count(), 0);
  } finally {
    await c.$disconnect();
  }
});

test('U-R8 delete: claim-free -> deleting reserves the canonical folder; case alias create and admission refused; failure -> delete_failed', async () => {
  const d = await makeJob(F.db, { name: 'MyJob', status: 'stopped' });
  const folder = path.join(F.trainingRoot, 'MyJob');
  fs.mkdirSync(folder, { recursive: true });
  assert.equal((await store.beginDelete(F.db, d.id, F.trainingRoot)).status, 200);
  assert.equal((await job(d.id)).status, 'deleting');
  // Make these alternate spellings real aliases on filesystems that distinguish them.
  for (const spelling of ['myjob', 'MYJOB. ']) {
    const aliasPath = path.join(F.trainingRoot, spelling);
    if (!fs.existsSync(aliasPath)) fs.symlinkSync(folder, aliasPath, 'dir');
  }
  for (const name of ['MyJob', 'myjob', 'MYJOB. ']) {
    assert.equal((await jobsRoute.POST({ json: async () => ({ name, gpu_ids: '0', job_config: {} }) })).status, 409, name);
  }
  assert.equal((await routeCall(startRoute, 'GET', d.id)).status, 409);
  const alias = await F.db.job.create({ data: { name: 'myjob', gpu_ids: '5', job_config: '{}', status: 'queued' } });
  assert.match((await admit(alias)).reason, /job folder in use by job "MyJob" \(being deleted\)/);
  await F.db.job.delete({ where: { id: alias.id } });
  await store.finishDelete(F.db, d.id, 'EBUSY: injected');
  assert.equal((await job(d.id)).status, 'delete_failed');
  assert.equal(await F.db.runClaim.count({ where: { kind: 'delete' } }), 1, 'folder still reserved');
  assert.equal((await jobsRoute.POST({ json: async () => ({ name: 'myjob', gpu_ids: '0', job_config: {} }) })).status, 409);
  assert.equal((await routeCall(startRoute, 'GET', d.id)).status, 409);
  const r = await routeCall(deleteRoute, 'GET', d.id);
  assert.equal(r.status, 200);
  assert.equal(await job(d.id), null);
  assert.equal(fs.existsSync(folder), false);
  assert.equal(await F.db.runClaim.count(), 0);
  for (const spelling of ['myjob', 'MYJOB. ']) {
    const aliasPath = path.join(F.trainingRoot, spelling);
    if (fs.lstatSync(aliasPath, { throwIfNoEntry: false })?.isSymbolicLink()) fs.unlinkSync(aliasPath);
  }
  assert.equal((await jobsRoute.POST({ json: async () => ({ name: 'myjob', gpu_ids: '0', job_config: {} }) })).status, 200);
});

test('U-R8 controlled interleavings: Delete issued while an admission transaction is open; Delete then admission; Start racing Delete', async () => {
  const second = F.newClient();
  try {
    await second.job.count();
    const j = await makeJob(F.db, { name: 'race', gpu_ids: '6', status: 'queued' });
    let deletion;
    const r = await admit(j, F.db, { afterRead: async () => void (deletion = store.beginDelete(second, j.id, F.trainingRoot)) });
    const d = await deletion;
    assert.ok(r.admitted, 'admitted first');
    assert.equal(d.status, 409, JSON.stringify(d.body));
    assert.equal((await job(j.id)).status, 'starting');
    await store.releaseRun(F.db, j.id, r.admitted.run_id, { phase: 'claimed' }, { kind: 'not_spawned', info: 'test' });
    const k = await makeJob(F.db, { name: 'race1', gpu_ids: '6', status: 'queued' });
    assert.equal((await store.beginDelete(second, k.id, F.trainingRoot)).status, 200);
    assert.equal((await admit(k)).admitted, null);
    assert.equal(await F.db.runClaim.count({ where: { job_id: k.id, kind: 'run' } }), 0);
    const m = await makeJob(F.db, { name: 'race2', status: 'stopped' });
    const [s, del] = await Promise.all([store.requestStart(F.db, m.id, F.trainingRoot), store.beginDelete(second, m.id, F.trainingRoot)]);
    assert.equal((await job(m.id)).status, 'deleting', `start ${s.status}, delete ${del.status}`);
    assert.equal(del.status, 200);
  } finally {
    await second.$disconnect();
  }
});

test('U-R9 Edit while starting/running/stopping/deleting -> 409; edit and rename after verified exit -> 200; rename alias refused', async () => {
  for (const status of ['starting', 'running', 'stopping', 'deleting']) {
    const j = await makeJob(F.db, { name: `e-${status}`, gpu_ids: '20', status });
    const r = await jobsRoute.POST({ json: async () => ({ id: j.id, name: `e-${status}`, gpu_ids: '1', job_config: {} }) });
    assert.equal(r.status, 409, status);
    assert.equal((await job(j.id)).gpu_ids, '20');
  }
  const j = await runningJob('e-run', '7');
  writeExitReceipt(j);
  await reconcile(fakeProcCtl());
  const ok = await jobsRoute.POST({ json: async () => ({ id: j.id, name: 'e-renamed', gpu_ids: '1', job_config: { a: 1 } }) });
  assert.equal(ok.status, 200);
  assert.equal((await job(j.id)).name, 'e-renamed');
  const other = await makeJob(F.db, { name: 'Other', status: 'stopped' });
  const renamed = path.join(F.trainingRoot, 'e-renamed');
  fs.mkdirSync(renamed, { recursive: true });
  const aliasPath = path.join(F.trainingRoot, 'E-RENAMED');
  if (!fs.existsSync(aliasPath)) fs.symlinkSync(renamed, aliasPath, 'dir');
  assert.equal((await jobsRoute.POST({ json: async () => ({ id: other.id, name: 'E-RENAMED', gpu_ids: '1', job_config: {} }) })).status, 409);
  assert.equal((await jobsRoute.POST({ json: async () => ({ name: '../escape', gpu_ids: '1', job_config: {} }) })).status, 400);
});

test('queue pause returns only that queue\'s running jobs to the queue, after verified exit', async () => {
  const a = await runningJob('qa', '0');
  const b = await runningJob('qb', '1,2');
  await F.db.queue.create({ data: { gpu_ids: '0', is_running: false } });
  await F.db.queue.create({ data: { gpu_ids: '1,2', is_running: true } });
  const launched = [];
  await queue.processQueueWith({ db: F.db, trainingRoot: F.trainingRoot, launcher: self, launch: j => launched.push(j) });
  let ra = await job(a.id);
  assert.equal(ra.status, 'stopping');
  assert.equal(ra.after_exit, 'queued');
  assert.equal(ra.return_to_queue, true);
  assert.equal((await job(b.id)).status, 'running', 'the other queue is untouched');
  await F.db.job.updateMany({ where: { id: a.id, run_id: a.run_id }, data: { status: 'queued' } }); // Python's return_to_queue path
  await reconcile(fakeProcCtl({ runs: { 'qa-sup@1': alive(rootOf('qa')) } }));
  assert.equal((await claimsOf(a.id)).length, 2);
  writeExitReceipt(ra);
  await reconcile(fakeProcCtl({ runs: { 'qb-sup@1': alive(rootOf('qb')) } }));
  ra = await job(a.id);
  assert.equal(ra.status, 'queued');
  assert.equal((await claimsOf(a.id)).length, 0);
  assert.equal(launched.length, 0);
});

test('admission through processQueue launches exactly the admitted run; a busy queue head is skipped', async () => {
  await F.db.queue.create({ data: { gpu_ids: '0', is_running: true } });
  const first = await makeJob(F.db, { name: 'pq1', gpu_ids: '0', status: 'queued', queue_position: 1 });
  await makeJob(F.db, { name: 'pq2', gpu_ids: '0', status: 'queued', queue_position: 2 });
  const launched = [];
  const ctx = { db: F.db, trainingRoot: F.trainingRoot, launcher: self, launch: j => launched.push(j) };
  await queue.processQueueWith(ctx);
  await queue.processQueueWith(ctx);
  assert.equal(launched.length, 1);
  assert.equal(launched[0].id, first.id);
  assert.equal(launched[0].status, 'starting');
  assert.equal(launched[0].launch_phase, 'claimed');
});

test('folder aliases through a real junction (Codex folder-alias-probe): same physical folder -> same claim key, admission and create refused', { skip: process.platform !== 'win32' }, async () => {
  const realA = path.join(F.trainingRoot, 'alias-a');
  fs.mkdirSync(realA, { recursive: true });
  fs.symlinkSync(realA, path.join(F.trainingRoot, 'alias-b'), 'junction');
  assert.equal(resources.canonicalFolder(F.trainingRoot, 'alias-a'), resources.canonicalFolder(F.trainingRoot, 'alias-b'));
  const a = await makeJob(F.db, { name: 'alias-a', gpu_ids: '40', status: 'queued' });
  const b = await F.db.job.create({ data: { name: 'alias-b', gpu_ids: '41', job_config: '{}', status: 'queued' } });
  assert.ok((await admit(a)).admitted);
  assert.match((await admit(b)).reason, /job folder in use by job "alias-a"/);
  await F.db.job.delete({ where: { id: b.id } });
  assert.equal((await jobsRoute.POST({ json: async () => ({ name: 'alias-b', gpu_ids: '42', job_config: {} }) })).status, 409);
  // the training root itself reached through a junction (this host's normal C:/AI style root): one key either way
  const rootLink = `${F.trainingRoot}-link`;
  fs.symlinkSync(F.trainingRoot, rootLink, 'junction');
  try {
    assert.equal(resources.canonicalFolder(rootLink, 'not-yet-created'), resources.canonicalFolder(F.trainingRoot, 'not-yet-created'));
    assert.equal(resources.canonicalFolder(rootLink, 'alias-a'), resources.canonicalFolder(F.trainingRoot, 'alias-a'));
  } finally {
    fs.rmSync(rootLink, { recursive: false, force: true });
  }
});

test('log cursor carries the log file generation (Codex log-generation-probe): a new run\'s larger log after rotation is a reset, not a continuation', async () => {
  const logRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/log/route.ts'));
  const j = await makeJob(F.db, { name: 'logjob', status: 'running' });
  const dir = path.join(F.trainingRoot, 'logjob');
  fs.mkdirSync(path.join(dir, 'logs'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'log.txt'), 'A'.repeat(1000));
  const get = async params => (await logRoute.GET({ nextUrl: { searchParams: new URLSearchParams(params) } }, { params: Promise.resolve({ jobID: j.id }) })).json();
  const first = await get({});
  assert.equal(first.offset, 1000);
  assert.ok(first.generation);
  const same = await get({ offset: '1000', generation: first.generation });
  assert.equal(same.reset, false);
  assert.equal(same.log, '');
  // the next launch rotates log.txt and the new run writes more than the old offset
  fs.renameSync(path.join(dir, 'log.txt'), path.join(dir, 'logs', '0_log.txt'));
  fs.writeFileSync(path.join(dir, 'log.txt'), 'B'.repeat(2000));
  const second = await get({ offset: String(first.offset), generation: first.generation });
  assert.equal(second.reset, true);
  assert.equal(second.log.length, 2000);
  assert.notEqual(second.generation, first.generation);
});

test('queue snapshot races (Codex queue-snapshot-probe): a pause committed after the tick read blocks admission; an enqueue+start after the empty read keeps the queue running', async () => {
  const q = await F.db.queue.create({ data: { gpu_ids: '-1', is_running: true } });
  await makeJob(F.db, { name: 'pause-race', status: 'queued', gpu_ids: '-1' });
  let paused = false;
  const pauseJob = new Proxy(F.db.job, {
    get(t, k) {
      if (k !== 'findMany') return t[k];
      return async args => {
        if (!paused) {
          paused = true;
          await F.db.queue.update({ where: { id: q.id }, data: { is_running: false } });
        }
        return t.findMany(args);
      };
    },
  });
  const launches = [];
  await queue.processQueueWith({ db: new Proxy(F.db, { get: (t, k) => (k === 'job' ? pauseJob : t[k]) }), trainingRoot: F.trainingRoot, launcher: self, launch: j => launches.push(j) });
  assert.equal(paused, true);
  assert.equal(launches.length, 0, 'not admitted into a paused queue');
  assert.equal(await F.db.runClaim.count(), 0);
  await reset(F.db);
  const empty = await F.db.queue.create({ data: { gpu_ids: '-1', is_running: true } });
  let enqueued = false;
  const enqueueJob = new Proxy(F.db.job, {
    get(t, k) {
      if (k !== 'findFirst') return t[k];
      return async args => {
        const snapshot = await t.findFirst(args);
        if (!snapshot && !enqueued) {
          enqueued = true;
          await makeJob(F.db, { name: 'newly-queued', status: 'queued', gpu_ids: '-1' });
          await F.db.queue.update({ where: { id: empty.id }, data: { is_running: true } });
        }
        return snapshot;
      };
    },
  });
  await queue.processQueueWith({ db: new Proxy(F.db, { get: (t, k) => (k === 'job' ? enqueueJob : t[k]) }), trainingRoot: F.trainingRoot, launcher: self, launch: () => assert.fail('nothing to launch this tick') });
  assert.equal(enqueued, true);
  assert.equal((await F.db.queue.findUnique({ where: { id: empty.id } })).is_running, true);
  // and a genuinely idle queue is still switched off
  await F.db.job.deleteMany();
  await queue.processQueueWith({ db: F.db, trainingRoot: F.trainingRoot, launcher: self, launch: () => {} });
  assert.equal((await F.db.queue.findUnique({ where: { id: empty.id } })).is_running, false);
});

test('fault injection: an unreadable folder identity (EACCES from realpath) is refused, never replaced by a lexical key', async () => {
  const real = fs.realpathSync.native;
  const j = await makeJob(F.db, { name: 'acl', gpu_ids: '50', status: 'queued' });
  fs.realpathSync.native = p => {
    if (String(p).includes('acl')) {
      const e = new Error(`EACCES: permission denied, realpath '${p}'`);
      e.code = 'EACCES';
      throw e;
    }
    return real(p);
  };
  try {
    fs.mkdirSync(path.join(F.trainingRoot, 'acl'), { recursive: true });
    assert.throws(() => resources.canonicalFolder(F.trainingRoot, 'acl'), resources.FolderIdentityError);
    const r = await admit(j);
    assert.equal(r.admitted, null);
    assert.match((await job(j.id)).ownership_note, /Cannot determine the real location/);
    assert.equal(await F.db.runClaim.count(), 0);
    const created = await jobsRoute.POST({ json: async () => ({ name: 'acl2', gpu_ids: '51', job_config: {} }) });
    assert.equal(created.status, 500);
  } finally {
    fs.realpathSync.native = real;
  }
});

test('folder ancestry through a real junction (Codex folder-ancestor-probe): nested physical folders conflict both ways; adjacent prefixes do not', { skip: process.platform !== 'win32' }, async () => {
  const nested = path.join(F.trainingRoot, 'parent', 'nested');
  fs.mkdirSync(nested, { recursive: true });
  fs.symlinkSync(nested, path.join(F.trainingRoot, 'inner'), 'junction'); // job "inner" lives inside job "parent"
  const inner = await makeJob(F.db, { name: 'inner', gpu_ids: '60', status: 'queued' });
  const parent = await makeJob(F.db, { name: 'parent', gpu_ids: '61', status: 'stopped' });
  assert.ok((await admit(inner)).admitted);
  const del = await store.beginDelete(F.db, parent.id, F.trainingRoot);
  assert.equal(del.status, 409, 'deleting the outer folder would remove the running inner job');
  assert.equal((await job(parent.id)).status, 'stopped');
  // reverse direction: the outer job runs, the inner one cannot be admitted
  await reset(F.db);
  const outer = await makeJob(F.db, { name: 'parent', gpu_ids: '62', status: 'queued' });
  const inner2 = await makeJob(F.db, { name: 'inner', gpu_ids: '63', status: 'queued' });
  assert.ok((await admit(outer)).admitted);
  assert.match((await admit(inner2)).reason, /job folder \(nested in or containing it\) in use by job "parent"/);
  // negative control: "ab" is not inside "a"
  assert.equal(resources.foldersOverlap(resources.canonicalFolder(F.trainingRoot, 'a'), resources.canonicalFolder(F.trainingRoot, 'ab')), false);
  const a = await makeJob(F.db, { name: 'a', gpu_ids: '64', status: 'queued' });
  const ab = await makeJob(F.db, { name: 'ab', gpu_ids: '65', status: 'queued' });
  assert.ok((await admit(a)).admitted);
  assert.ok((await admit(ab)).admitted);
});

test('queue idle-stop sees a run another worker admitted after this tick read the queue (Codex queue-active-snapshot-probe)', async () => {
  const q = await F.db.queue.create({ data: { gpu_ids: '-1', is_running: true } });
  const j = await makeJob(F.db, { name: 'other-worker-admits', status: 'queued', gpu_ids: '-1' });
  let admitted = false;
  const proxyJob = new Proxy(F.db.job, {
    get(t, k) {
      if (k !== 'findMany') return t[k];
      return async args => {
        const snapshot = await t.findMany(args);
        if (!admitted) {
          const out = await store.admit(F.db, j.id, F.trainingRoot, { pid: 111, identity: '111@1000', boot_id: 'boot-test-1' });
          admitted = !!out.admitted;
        }
        return snapshot;
      };
    },
  });
  await queue.processQueueWith({ db: new Proxy(F.db, { get: (t, k) => (k === 'job' ? proxyJob : t[k]) }), trainingRoot: F.trainingRoot, launcher: self, launch: () => assert.fail('no second launch') });
  assert.equal(admitted, true);
  assert.equal((await job(j.id)).launch_phase, 'claimed');
  assert.equal((await F.db.queue.findUnique({ where: { id: q.id } })).is_running, true, 'the queue keeps running for the new run');
});

test('the operation-lock directory name is reserved in every spelling Windows maps to it', async () => {
  for (const name of ['.aitk_locks', '.AITK_LOCKS', '.aitk_locks.', '.aitk_locks  ']) {
    assert.match(resources.jobFolderError(F.trainingRoot, name), /reserved/, JSON.stringify(name));
    assert.equal((await jobsRoute.POST({ json: async () => ({ name, gpu_ids: '0', job_config: {} }) })).status, 400, JSON.stringify(name));
  }
  const legacy = await F.db.job.create({ data: { name: '.aitk_locks', gpu_ids: '0', job_config: '{}', status: 'stopped' } });
  const lockFolder = path.join(F.trainingRoot, '.aitk_locks');
  fs.mkdirSync(lockFolder, { recursive: true });
  const existingLock = path.join(lockFolder, 'existing.lock');
  fs.writeFileSync(existingLock, 'preserve');
  const deleted = await routeCall(deleteRoute, 'POST', legacy.id, legacy);
  assert.equal(deleted.status, 200);
  assert.match(deleted.body.keptOutput, /no valid output folder/);
  assert.equal(await job(legacy.id), null);
  assert.equal(fs.readFileSync(existingLock, 'utf8'), 'preserve');
});

test('resuming after a rename and a training-root change retains the original checkpoints', async () => {
  const name = 'original-name';
  const folder = path.join(F.trainingRoot, name);
  fs.mkdirSync(folder);
  const checkpoint = path.join(folder, name + '_000000010.safetensors');
  fs.writeFileSync(checkpoint, 'existing checkpoint');
  const j = await makeJob(F.db, {
    name: 'renamed-job', status: 'stopped', gpu_ids: '-1', run_id: 'previous-run', run_dir: folder, launch_phase: 'exited',
  });
  const newRoot = path.join(F.trainingRoot, 'other-root');
  assert.equal((await store.requestStart(F.db, j.id, newRoot)).status, 200);
  const { admitted } = await store.admit(F.db, j.id, newRoot, self);
  assert.equal(admitted.run_dir, folder);
  const { bindLaunchConfig } = require(ts.uiPath('cron/ownership/launchConfig.ts'));
  const config = bindLaunchConfig({ config: { process: [{ type: 'diffusion_trainer' }] } }, admitted.name, admitted.run_dir);
  assert.equal(config.config.process[0].name, name);
  assert.equal(fs.readFileSync(checkpoint, 'utf8'), 'existing checkpoint');
});

test('pausing a queue during startup returns the unspawned run to the queue', async () => {
  const j = await makeJob(F.db, { name: 'pause-before-spawn', status: 'queued' });
  const { admitted } = await admit(j);
  assert.ok(admitted);
  await F.db.queue.upsert({ where: { gpu_ids: j.gpu_ids }, create: { gpu_ids: j.gpu_ids }, update: { is_running: false } });
  await store.requestQueueReturn(F.db, admitted);
  assert.equal(await store.markSpawnRequested(F.db, j.id, admitted.run_id), false);
  await reconcile(fakeProcCtl());
  const paused = await job(j.id);
  assert.equal(paused.status, 'queued');
  assert.equal(paused.info, 'Job queued');
  assert.equal((await claimsOf(j.id)).length, 0);
});

test('moving or copying the run to the configured root preserves the checkpoint name and chooses the verified copy', async () => {
  const movedRoot = path.join(F.trainingRoot, 'new-location');
  const name = 'writer-prefix';
  const old = path.join(F.trainingRoot, name);
  const moved = path.join(movedRoot, name);
  const marker = '.job_config.moved-generation.json';
  const weights = name + '_000000800.safetensors';
  fs.mkdirSync(moved, { recursive: true });
  fs.writeFileSync(path.join(moved, marker), '{}');
  fs.writeFileSync(path.join(moved, weights), 'moved');
  const j = await makeJob(F.db, { name: 'renamed-in-the-ui', status: 'stopped', step: 800, run_id: 'moved-generation', run_dir: old, launch_phase: 'exited' });
  // Even a leftover directory at the former location must not win over the matching launch generation.
  fs.mkdirSync(old);
  fs.writeFileSync(path.join(old, 'unrelated'), 'keep');
  assert.equal(resources.jobOutputFolder(movedRoot, j), moved);
  fs.writeFileSync(path.join(old, marker), '{}'); // the user kept a copy at both locations
  assert.equal(resources.jobOutputFolder(movedRoot, j), moved);
  assert.equal((await store.requestStart(F.db, j.id, movedRoot)).status, 200);
  const { admitted } = await store.admit(F.db, j.id, movedRoot, self);
  assert.equal(admitted.run_dir, moved);
  assert.equal(admitted.previousRunId, j.run_id, 'launch can still verify the file for the previous generation');
  fs.unlinkSync(path.join(moved, weights));
  assert.doesNotThrow(() => resources.checkResumeFolder(moved, admitted, admitted.previousRunId));
  fs.writeFileSync(path.join(moved, weights), 'moved');
  const { bindLaunchConfig } = require(ts.uiPath('cron/ownership/launchConfig.ts'));
  const bound = bindLaunchConfig({ config: { process: [{ type: 'sd_trainer' }] } }, admitted.name, admitted.run_dir);
  assert.equal(path.join(bound.config.process[0].training_folder, bound.config.process[0].name, weights), path.join(moved, weights));
});

test('a missing resume directory reports the path and yields the queue without starting from zero', async () => {
  const name = 'unmounted-run';
  const old = path.join(F.trainingRoot, 'disconnected', name);
  const current = path.join(F.trainingRoot, name);
  fs.mkdirSync(current);
  fs.writeFileSync(path.join(current, '.job_config.unrelated.json'), '{}');
  const j = await makeJob(F.db, { name, status: 'stopped', step: 800, run_dir: old, run_id: 'previous-generation', launch_phase: 'exited' });
  const start = await store.requestStart(F.db, j.id, F.trainingRoot);
  assert.equal(start.status, 409);
  assert.ok(start.body.error.includes(old));
  await F.db.job.update({ where: { id: j.id }, data: { status: 'queued' } });
  assert.equal((await admit(j)).admitted, null);
  assert.equal((await job(j.id)).status, 'error');
  assert.equal((await job(j.id)).step, 800);
  assert.equal(fs.existsSync(path.dirname(old)), false);
  fs.mkdirSync(old, { recursive: true });
  assert.equal((await store.requestStart(F.db, j.id, F.trainingRoot)).status, 409, 'empty remounted folder is still missing its run');
  await F.db.job.update({ where: { id: j.id }, data: { step: 0 } });
  assert.equal((await store.requestStart(F.db, j.id, F.trainingRoot)).status, 200, 'a job interrupted before it made progress can retry');
});
