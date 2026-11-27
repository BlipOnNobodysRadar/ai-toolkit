const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { setupDb, makeJob, ts } = require('./harness/fixture.cjs');
const { recoverTrainingProgress } = require(ts.uiPath('cron/recoverTrainingProgress.ts'));

test('older interrupted jobs recover the last reported training step for their own launch', async t => {
  const { db, trainingRoot } = await setupDb('historical-progress');
  t.after(() => db.$disconnect());
  const originalTime = new Date('2026-11-08T00:00:00Z');
  async function fixture(name, status, step, options = {}) {
    const folder = path.join(trainingRoot, name);
    fs.mkdirSync(folder);
    const job = await makeJob(db, { name, status, step, updated_at: originalTime, ...options });
    return { ...job, log: path.join(folder, 'log.txt') };
  }
  const interrupted = await fixture('My [run] 日本語', 'stopped', 20);
  const oldFolderName = 'name before editing';
  const renamed = await fixture(oldFolderName, 'error', 3, { run_id: 'old', launch_phase: 'exited' });
  await db.job.update({ where: { id: renamed.id }, data: { name: 'renamed job', run_dir: path.dirname(renamed.log) } });
  const untouched = await fixture('other job', 'error', 7);
  const exact = await fixture('recent run', 'error', 21);
  const active = await fixture('live run', 'running', 7, { pid: 321 });
  await fixture('missing log', 'error', 0);
  const bar = (name, step) => `\r${name}:  25%|##5       | ${step}/100 [00:10<00:30,  2.50it/s]`;
  fs.writeFileSync(interrupted.log, `Job ID: "${interrupted.id}"\n${bar(interrupted.name, 22)}${bar(interrupted.name, 25)}${bar('Sampling', 90)}\nJob stopped\n`);
  fs.writeFileSync(renamed.log, `Job ID: "${renamed.id}"\n${bar(oldFolderName, 26)}\n`);
  fs.writeFileSync(untouched.log, `Job ID: "${interrupted.id}"\n${bar(untouched.name, 25)}\n`);
  fs.writeFileSync(exact.log, `Job ID: "${exact.id}"\n${bar(exact.name, 20)}\n`);
  fs.writeFileSync(active.log, `Job ID: "${active.id}"\n${bar(active.name, 25)}\n`);
  await recoverTrainingProgress(db, trainingRoot);
  const read = job => db.job.findUnique({ where: { id: job.id } });
  assert.equal((await read(interrupted)).step, 25);
  assert.deepEqual((await read(interrupted)).updated_at, originalTime);
  assert.equal((await read(renamed)).step, 26);
  assert.equal((await read(untouched)).step, 7);
  assert.equal((await read(exact)).step, 21);
  assert.equal((await read(active)).step, 7);
});

test('loss API invalidates a cursor whose logged step was rewritten by resumed training', async t => {
  const { db, trainingRoot } = await setupDb('loss-rewind');
  t.after(() => db.$disconnect());
  const job = await makeJob(db, { name: 'resumed', run_id: 'current' });
  const jobDir = path.join(trainingRoot, job.name);
  fs.mkdirSync(jobDir);
  const sqlite = require('sqlite3');
  const metrics = new sqlite.Database(path.join(jobDir, 'loss_log.db'));
  t.after(() => new Promise(resolve => metrics.close(resolve)));
  const exec = sql => new Promise((resolve, reject) => metrics.exec(sql, err => err ? reject(err) : resolve()));
  await exec(`CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL);
    CREATE TABLE metric_keys (key TEXT PRIMARY KEY);
    CREATE TABLE metrics (step INTEGER, key TEXT, value_real REAL);
    INSERT INTO steps VALUES (3, 117), (10, 120);
    INSERT INTO metric_keys VALUES ('loss');
    INSERT INTO metrics VALUES (3, 'loss', 1), (10, 'loss', 9);`);
  const { GET } = require(ts.uiPath('src/app/api/jobs/[jobID]/loss/route.ts'));
  const read = async query => (await GET(new Request('http://localhost/loss' + query), { params: Promise.resolve({ jobID: job.id }) })).json();
  assert.deepEqual((await read('')).cursor, { step: 10, wall_time: 120 });
  assert.deepEqual((await read('?since_step=10&since_time=120')).series.loss.steps, [10]);
  await exec('UPDATE steps SET wall_time = 155 WHERE step = 10; UPDATE metrics SET value_real = 2 WHERE step = 10;');
  const reset = await read('?since_step=10&since_time=120');
  assert.equal(reset.reset, true);
  assert.deepEqual(reset.series.loss.values, [1, 2]);
  await exec('DELETE FROM steps WHERE step = 10; DELETE FROM metrics WHERE step = 10;');
  assert.deepEqual((await read('?since_step=10&since_time=155')).series.loss.steps, [3]);
});
