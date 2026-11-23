const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { setupDb, makeJob, ts } = require('./harness/fixture.cjs');

test('marking a stale job stopped preserves its output, then Delete removes only that job', async t => {
  const { db, trainingRoot } = await setupDb('stale-job');
  t.after(async () => {
    await db.$disconnect();
    fs.rmSync(trainingRoot, { recursive: true, force: true });
  });

  const markStopped = require(ts.uiPath('src/app/api/jobs/[jobID]/mark_stopped/route.ts'));
  const deleteJob = require(ts.uiPath('src/app/api/jobs/[jobID]/delete/route.ts'));
  const stale = await makeJob(db, { name: 'old-job', status: 'running', pid: 2147483647, step: 150 });
  const next = await makeJob(db, { name: 'next-job', status: 'queued' });
  const checkpoint = job => path.join(trainingRoot, job.name, 'model.safetensors');
  for (const job of [stale, next]) {
    fs.mkdirSync(path.join(trainingRoot, job.name));
    fs.writeFileSync(checkpoint(job), job.name);
  }

  const params = Promise.resolve({ jobID: stale.id });
  const where = { id: stale.id };
  const signal = t.mock.method(process, 'kill', () => { throw new Error('unexpected process signal'); });
  assert.equal((await markStopped.GET({}, { params })).status, 200);
  assert.equal(signal.mock.callCount(), 0);
  const stopped = await db.job.findUniqueOrThrow({ where });
  assert.equal(stopped.status, 'stopped');
  assert.equal(stopped.pid, null);
  assert.equal(stopped.step, 150);
  assert.equal(fs.readFileSync(checkpoint(stale), 'utf8'), stale.name);

  assert.equal((await deleteJob.GET({}, { params })).status, 200);
  assert.equal(await db.job.findUnique({ where }), null);
  assert.equal(fs.existsSync(path.dirname(checkpoint(stale))), false);
  assert.equal((await db.job.findUniqueOrThrow({ where: { id: next.id } })).status, 'queued');
  assert.equal(fs.readFileSync(checkpoint(next), 'utf8'), next.name);
});
