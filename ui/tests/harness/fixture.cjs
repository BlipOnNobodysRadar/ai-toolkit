// Per-file fixture: an isolated Prisma client on a fresh temporary SQLite database (schema pushed with the real
// Prisma CLI), a temporary training root registered in Settings, and jobs stored in that database.
const fs = require('fs');
const path = require('path');
const { SCRATCH } = require('./env.cjs');
const { isolatedPrisma, pushSchema, tempDbFile, clientFor } = require('./prisma.cjs');
const ts = require('./tsrequire.cjs');

const setupDb = async (label, { wal = true } = {}) => {
  const p = isolatedPrisma();
  const file = tempDbFile(label);
  pushSchema(p.schemaPath, file);
  ts.usePrismaClient(p.clientDir);
  process.env.AITK_TEST_DATABASE_URL = `file:${file.replace(/\\/g, '/')}`;
  const db = clientFor(p.PrismaClient, file);
  if (wal) await db.$queryRawUnsafe('PRAGMA journal_mode = WAL;');
  const trainingRoot = fs.mkdtempSync(path.join(SCRATCH, `${label}-root-`));
  await db.settings.create({ data: { key: 'TRAINING_FOLDER', value: trainingRoot } });
  // the routes' shared client (src/server/prisma.ts) picks this up from globalThis
  globalThis.prisma = db;
  return { db, file, trainingRoot, newClient: () => clientFor(p.PrismaClient, file), PrismaClient: p.PrismaClient };
};

const jobConfig = name => ({ job: 'extension', config: { name, process: [{ type: 'ui_trainer', train: { steps: 10 } }] } });

const makeJob = (db, data) =>
  db.job.create({ data: { name: data.name, gpu_ids: data.gpu_ids ?? '0', job_config: JSON.stringify(jobConfig(data.name)), ...data } });

module.exports = { setupDb, makeJob, ts };
