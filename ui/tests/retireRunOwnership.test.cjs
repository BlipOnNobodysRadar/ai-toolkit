const assert = require('node:assert/strict');
const test = require('node:test');
const { retireRunOwnership } = require('../prisma/retireRunOwnership.cjs');
const { clientFor, isolatedPrisma, pushSchema, tempDbFile } = require('./harness/prisma.cjs');

test('normal db push preserves jobs, queues and settings when retiring populated ownership data', async t => {
  const { PrismaClient, schemaPath } = isolatedPrisma();
  const file = tempDbFile('retire-ownership');
  const db = clientFor(PrismaClient, file);
  t.after(() => db.$disconnect());

  // The update command also runs on fresh databases before Prisma creates its tables.
  await retireRunOwnership(db);
  pushSchema(schemaPath, file);
  const queue = await db.queue.create({ data: { gpu_ids: '0', is_running: true } });
  const setting = await db.settings.create({ data: { key: 'TRAINING_FOLDER', value: 'custom-output' } });
  const statuses = ['running', 'completed', 'queued', 'stopped', 'starting', 'stopping', 'delete_failed', 'deleting'];
  const jobs = [];
  for (const [index, status] of statuses.entries()) {
    jobs.push(await db.job.create({ data: {
      name: status, gpu_ids: '0', job_config: JSON.stringify({ name: status }), status,
      stop: status === 'stopping', pid: status === 'running' ? 4321 : null,
      queue_position: index, step: index * 10, info: 'existing details',
    } }));
  }

  await db.$executeRaw`ALTER TABLE "Job" ADD COLUMN "run_id" TEXT`;
  await db.$executeRaw`ALTER TABLE "Job" ADD COLUMN "boot_id" TEXT`;
  await db.$executeRaw`CREATE INDEX "Job_run_id_idx" ON "Job" ("run_id")`;
  await db.$executeRaw`UPDATE "Job" SET "run_id" = 'old-run' WHERE "status" != 'running'`;
  await db.$executeRaw`UPDATE "Job" SET "boot_id" = 'old-boot' WHERE "status" = 'running'`;
  await db.$executeRaw`CREATE TABLE "RunClaim" ("resource" TEXT PRIMARY KEY, "run_id" TEXT)`;
  await db.$executeRaw`INSERT INTO "RunClaim" VALUES ('gpu:0', 'old-run')`;

  // Retrying a partial update, and then starting again on an up-to-date installation.
  await retireRunOwnership(db);
  await retireRunOwnership(db);
  pushSchema(schemaPath, file);
  await retireRunOwnership(db);
  pushSchema(schemaPath, file);

  const newStatus = { starting: 'running', stopping: 'running', deleting: 'error', delete_failed: 'error' };
  assert.deepEqual(await db.job.findMany({ orderBy: { queue_position: 'asc' } }),
    jobs.map(job => ({ ...job, status: newStatus[job.status] ?? job.status })));
  assert.deepEqual(await db.queue.findMany(), [queue]);
  assert.deepEqual(await db.settings.findMany(), [setting]);
  assert.deepEqual(await db.$queryRaw`SELECT name FROM sqlite_master WHERE name IN ('RunClaim', 'Job_run_id_idx')`, []);
  assert.equal((await db.$queryRaw`PRAGMA table_info("Job")`).some(c => ['run_id', 'boot_id'].includes(c.name)), false);
});
