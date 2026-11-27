// Actual log route + real temporary files. DB/settings and the precise filesystem interleavings are injected.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs'), path = require('path'), vm = require('vm');
const { UI_ROOT, SCRATCH, fromModules } = require('./harness/env.cjs');
const ts = fromModules('typescript');
const source = fs.readFileSync(path.join(UI_ROOT, 'src/app/api/jobs/[jobID]/log/route.ts'), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;

function fixture(open) {
  const root = fs.mkdtempSync(path.join(SCRATCH, 'log-stream-'));
  fs.mkdirSync(path.join(root, 'job'));
  const log = path.join(root, 'job', 'log.txt');
  const wrapped = { ...fs, promises: { ...fs.promises, open: (...args) => open(log, ...args) } };
  const mocks = {
    '../../../../../../cron/ownership/jobArtifacts': require(require('./harness/tsrequire.cjs').uiPath('cron/ownership/jobArtifacts.ts')),
    'next/server': { NextResponse: { json: body => ({ body }) } },
    '@/server/prisma': { default: { job: { findUnique: async () => ({ id: 'job', name: 'job', run_id: 'run' }) } } },
    '@/server/settings': { getTrainingFolder: async () => root },
    fs: { default: wrapped }, path: { default: path },
  };
  const box = { exports: {}, require: name => mocks[name] ?? require(name), Buffer, console };
  vm.runInNewContext(code, box);
  return { log, get: params => box.exports.GET({ nextUrl: { searchParams: new URLSearchParams(params) } }, { params: { jobID: 'job' } }) };
}

test('rotation immediately before opening the path returns the opened stream identity and a reset', async () => {
  let rotate = false;
  const f = fixture(async (log, ...args) => {
    if (rotate && args[0] === log) {
      rotate = false;
      fs.renameSync(log, `${log}.old`);
      fs.writeFileSync(log, 'B'.repeat(3000));
    }
    return fs.promises.open(...args);
  });
  fs.writeFileSync(f.log, 'A'.repeat(1000));
  const first = (await f.get({})).body;
  fs.appendFileSync(f.log, 'A'.repeat(1000));
  rotate = true;
  const next = (await f.get({ offset: String(first.offset), generation: first.generation })).body;
  assert.equal(next.reset, true);
  assert.notEqual(next.generation, first.generation);
  assert.equal(next.log, 'B'.repeat(3000));
  assert.equal(next.offset, 3000);
});

test('a short read advances only by consumed bytes, without zero padding or skipping the remainder', async () => {
  let short = true;
  const f = fixture(async (_log, ...args) => {
    const fh = await fs.promises.open(...args);
    if (short) {
      short = false;
      const read = fh.read.bind(fh);
      fh.read = (buffer, offset, length, position) => read(buffer, offset, Math.min(4, length), position);
    }
    return fh;
  });
  fs.writeFileSync(f.log, 'abcdefghij');
  const first = (await f.get({})).body;
  assert.equal(first.log, 'abcd');
  assert.equal(first.offset, 4);
  const next = (await f.get({ offset: String(first.offset), generation: first.generation })).body;
  assert.equal(next.reset, false);
  assert.equal(next.log, 'efghij');
  assert.equal(next.offset, 10);
});
