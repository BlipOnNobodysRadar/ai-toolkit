// Log cursor and multi-byte characters (docs/STATE_MACHINES.md, log stream): the actual log route on a real temporary
// log file, appended in pieces that split UTF-8 sequences between two reads (isolated Prisma + temporary SQLite).
//   node --test --test-concurrency=1 ui/tests/logUtf8.test.cjs
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { setupDb, makeJob, ts } = require('./harness/fixture.cjs');

let F, route, jobId, logFile;
before(async () => {
  F = await setupDb('logutf8');
  route = require(ts.uiPath('src/app/api/jobs/[jobID]/log/route.ts'));
  const j = await makeJob(F.db, { name: 'utf8job', status: 'running' });
  jobId = j.id;
  fs.mkdirSync(path.join(F.trainingRoot, 'utf8job'), { recursive: true });
  logFile = path.join(F.trainingRoot, 'utf8job', 'log.txt');
});
after(async () => F.db.$disconnect());

const get = async params => (await route.GET({ nextUrl: { searchParams: new URLSearchParams(params) } }, { params: Promise.resolve({ jobID: jobId }) })).json();

test('a character split across two appends is delivered whole, once, by the next read', async () => {
  const e = Buffer.from('é'); // c3 a9
  const rocket = Buffer.from('🚀'); // f0 9f 9a 80
  fs.writeFileSync(logFile, Buffer.concat([Buffer.from('A'), e.subarray(0, 1)]));
  const first = await get({});
  assert.equal(first.log, 'A');
  assert.equal(first.offset, 1, 'the cursor stops before the incomplete character');
  fs.appendFileSync(logFile, Buffer.concat([e.subarray(1), Buffer.from('B'), rocket.subarray(0, 1)]));
  const second = await get({ offset: String(first.offset), generation: first.generation });
  assert.equal(second.reset, false);
  assert.equal(second.log, 'éB');
  fs.appendFileSync(logFile, Buffer.concat([rocket.subarray(1), Buffer.from('\n')]));
  const third = await get({ offset: String(second.offset), generation: second.generation });
  assert.equal(third.log, '🚀\n');
  assert.equal(third.offset, fs.statSync(logFile).size);
  assert.doesNotMatch(first.log + second.log + third.log, /�/);
});

test('malformed bytes are not held back (they decode as U+FFFD as before)', async () => {
  const before = fs.statSync(logFile).size;
  fs.appendFileSync(logFile, Buffer.from([0x58, 0xff, 0x59, 0x80])); // X <invalid> Y <stray continuation>
  const r = await get({ offset: String(before), generation: (await get({})).generation });
  assert.equal(r.offset, before + 4);
  assert.equal(r.log, 'X�Y�');
});

test('already-invalid prefixes (overlong E0 80, surrogate ED A0, overlong F0 80, above U+10FFFF F4 90) are decoded now, not withheld; valid prefixes still wait', async () => {
  const cases = [
    { tail: [0xe0, 0x80], held: false, why: 'overlong 3-byte' },
    { tail: [0xed, 0xa0], held: false, why: 'UTF-16 surrogate' },
    { tail: [0xf0, 0x80], held: false, why: 'overlong 4-byte' },
    { tail: [0xf4, 0x90], held: false, why: 'above U+10FFFF' },
    { tail: [0xf0, 0x80, 0x80], held: false, why: 'overlong 4-byte, 3 bytes' },
    { tail: [0xe0, 0xa0], held: true, why: 'valid U+0800 prefix' },
    { tail: [0xed, 0x9f], held: true, why: 'valid U+D7C0 prefix' },
    { tail: [0xf0, 0x90], held: true, why: 'valid U+10000 prefix' },
    { tail: [0xf4, 0x8f], held: true, why: 'valid U+10FFFF prefix' },
    { tail: [0xf0, 0x9f, 0x9a], held: true, why: 'valid 3 of 4 bytes' },
    { tail: [0xe2], held: true, why: 'lone valid lead' },
  ];
  for (const c of cases) {
    const before = fs.statSync(logFile).size;
    const generation = (await get({})).generation;
    fs.appendFileSync(logFile, Buffer.from([0x5a, ...c.tail])); // 'Z' + tail
    const r = await get({ offset: String(before), generation });
    const expectOffset = c.held ? before + 1 : before + 1 + c.tail.length;
    assert.equal(r.offset, expectOffset, `${c.why}: offset`);
    if (c.held) assert.equal(r.log, 'Z', c.why);
    else assert.match(r.log, /^Z\uFFFD+$/, c.why);
    // start the next case on a clean boundary: complete or terminate the pending bytes
    fs.appendFileSync(logFile, Buffer.from(c.held ? [0x0a] : []));
    if (c.held) await get({ offset: String(r.offset), generation });
  }
});

test('a renamed job under a previous training root reads its own artifacts and downloads', async t => {
  const previous = fs.mkdtempSync(path.join(path.dirname(F.trainingRoot), 'historical-artifacts-'));
  t.after(() => fs.rmSync(previous, { recursive: true, force: true }));
  const folder = path.join(previous, 'original');
  const samples = path.join(folder, 'samples');
  fs.mkdirSync(samples, { recursive: true });
  const old = await makeJob(F.db, { name: 'renamed', run_dir: folder, run_id: 'saved-run', launch_phase: 'exited' });
  const params = { params: Promise.resolve({ jobID: old.id }) };
  const file = path.join(folder, 'model_10.safetensors');
  const sample = path.join(samples, 'step_10.WAV');
  fs.writeFileSync(file, 'checkpoint');
  fs.writeFileSync(sample, 'audio');
  fs.writeFileSync(path.join(samples, 'step_10.txt'), 'caption');
  fs.writeFileSync(path.join(folder, 'log.txt'), 'previous run');
  const req = { nextUrl: new URL('http://localhost'), headers: new Headers() };

  assert.equal((await (await route.GET(req, params)).json()).log, 'previous run');
  const files = require(ts.uiPath('src/app/api/jobs/[jobID]/files/route.ts'));
  const gallery = require(ts.uiPath('src/app/api/jobs/[jobID]/samples/route.ts'));
  assert.deepEqual((await (await files.GET(req, params)).json()).files.map(f => f.path), [file]);
  assert.deepEqual((await (await gallery.GET(req, params)).json()).samples, [sample]);

  const loras = require(ts.uiPath('src/app/api/loras/route.ts'));
  const loraJobs = (await (await loras.GET()).json()).jobs;
  assert.deepEqual(loraJobs.find(row => row.id === old.id).files.map(f => f.path), [file]);
  const engineFolder = path.join(previous, 'inference');
  fs.mkdirSync(engineFolder);
  const engine = await makeJob(F.db, { name: 'renamed-engine', job_type: 'inference', status: 'running',
    run_dir: engineFolder, run_id: 'engine-run', launch_phase: 'attached' });
  fs.writeFileSync(path.join(engineFolder, 'engine.json'), JSON.stringify({ job_id: engine.id, host: '127.0.0.1', port: 1234, pid: process.pid }));
  const { listEngines } = require(ts.uiPath('src/server/inferenceEngine.ts'));
  assert.equal((await listEngines(true)).find(item => item.jobId === engine.id).endpoint.port, 1234);

  const download = require(ts.uiPath('src/app/api/files/[...filePath]/route.ts'));
  const getFile = p => download.GET(req, { params: Promise.resolve({ filePath: [path.dirname(p), path.basename(p)] }) });
  assert.equal(await (await getFile(file)).text(), 'checkpoint');
  assert.equal((await getFile(path.join(folder, 'log.txt'))).status, 403);
  const neighbor = path.join(previous, 'other.safetensors');
  fs.writeFileSync(neighbor, 'unrelated');
  assert.equal((await getFile(neighbor)).status, 403);
  if (process.platform !== 'win32') {
    const link = path.join(samples, 'link.WAV');
    fs.symlinkSync(neighbor, link);
    assert.equal((await getFile(link)).status, 403);
  }
});
