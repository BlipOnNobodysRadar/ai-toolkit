// Per-file fixture: an isolated Prisma client on a fresh temporary SQLite database (schema pushed with the real
// Prisma CLI), a temporary training root registered in Settings, and helpers for barriers and fake process layers.
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

const reset = async db => {
  await db.runClaim.deleteMany();
  await db.job.deleteMany();
  await db.queue.deleteMany();
};

const jobConfig = name => ({ job: 'extension', config: { name, process: [{ type: 'ui_trainer', train: { steps: 10 } }] } });

const makeJob = (db, data) =>
  db.job.create({ data: { name: data.name, gpu_ids: data.gpu_ids ?? '0', job_config: JSON.stringify(jobConfig(data.name)), ...data } });

// n arrivals release everyone; later arrivals pass straight through
const barrier = n => {
  let count = 0;
  let open;
  const opened = new Promise(r => (open = r));
  return async () => {
    count += 1;
    if (count === n) open();
    if (count <= n) await opened;
  };
};

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return { promise, resolve, reject };
};

// A scripted process layer for reconciler/force-stop unit tests (real OS behaviour is covered in launch.test.cjs).
const fakeProcCtl = (script = {}) => {
  const calls = { probe: [], terminate: [] };
  const api = {
    helperPath: 'fake',
    python: 'fake',
    calls,
    script,
    // runs are scripted by supervisor identity; scans by needle; describe by pid
    async probe(req) {
      calls.probe.push(JSON.parse(JSON.stringify(req)));
      if (script.onProbe) await script.onProbe(req);
      if (script.down) return { ok: false, error: 'helper down', describe: {}, runs: {}, scans: {} };
      const out = { ok: true, boot_ms: script.boot_ms ?? 0, describe: {}, runs: {}, scans: {} };
      for (const pid of req.describe || []) out.describe[pid] = (script.describe || {})[pid] ?? { state: 'unknown' };
      for (const [k, v] of Object.entries(req.runs || {})) {
        out.runs[k] = (script.runs || {})[v.supervisor] ?? { supervisor: 'unknown', boundary: 'unknown', live: [], members: {} };
      }
      for (const [k, v] of Object.entries(req.scans || {})) out.scans[k] = (script.scans || {})[v.needle] ?? { complete: false, matches: [] };
      return out;
    },
    async terminate(runId, supervisor) {
      calls.terminate.push({ runId, supervisor });
      return script.terminate ? script.terminate(runId, supervisor) : { supported: true, verified: true, remaining: 0 };
    },
    // fence: the real helper's exclusive hard-link claim, done on the real run folder (script.fenceError injects a failure)
    async fence(runId, runDir, by) {
      calls.fence = calls.fence || [];
      calls.fence.push({ runId, runDir, by });
      if (script.fenceError) return { fenced: false, error: script.fenceError };
      const file = path.join(runDir, `.run.${runId}.boundary.json`);
      fs.mkdirSync(runDir, { recursive: true });
      const tmp = `${file}.${process.pid}.tmp`;
      fs.writeFileSync(tmp, JSON.stringify({ run_id: runId, kind: 'fence', by }));
      try {
        fs.linkSync(tmp, file);
        return { fenced: true };
      } catch (e) {
        if (e.code !== 'EEXIST') throw e;
        return { fenced: false, boundary: JSON.parse(fs.readFileSync(file, 'utf8')) };
      } finally {
        fs.unlinkSync(tmp);
      }
    },
    async self() {
      return { identity: 'self@1', boot_id: 'boot-test-1' };
    },
  };
  return api;
};

const routeCall = async (mod, method, jobID, body) => {
  const req = body === undefined ? {} : { json: async () => body };
  const res = await mod[method](req, { params: Promise.resolve({ jobID }) });
  return { status: res.status, body: await res.json() };
};

module.exports = { setupDb, reset, makeJob, barrier, deferred, fakeProcCtl, routeCall, jobConfig, ts };
