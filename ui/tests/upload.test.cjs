// LoRA upload protocol (docs/STATE_MACHINES.md M19, U-R12): the actual route and server/uploadSession.ts on real
// temporary Windows files, with REAL second Node processes (fixtures/uploadChild.cjs) for concurrent finishes, a
// chunk holding the lock mid-write, and a finish that dies right after publishing. Original route failures are kept
// separately (upload_probe.cjs/json in the run folder: both racing finishes 200 with the loser's bytes, a 1-of-4-byte
// upload published zero-padded).
//   node --test --test-concurrency=1 ui/tests/upload.test.cjs
const { test, before } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const ts = require('./harness/tsrequire.cjs');
const { SCRATCH } = require('./harness/env.cjs');

let up, route, root, partDir;
before(() => {
  root = fs.mkdtempSync(path.join(SCRATCH, 'uploads-'));
  partDir = path.join(root, 'loras', '.uploads');
  ts.mock('@/server/settings', { getModelsPath: async () => root });
  up = require(ts.uiPath('src/server/uploadSession.ts'));
  route = require(ts.uiPath('src/app/api/loras/upload/route.ts'));
});
const loras = () => path.join(root, 'loras');
const call = async (action, params = {}, bytes = Buffer.alloc(0)) => {
  const res = await route.POST({ nextUrl: { searchParams: new URLSearchParams({ action, ...params }) }, arrayBuffer: async () => bytes });
  return { status: res.status, body: await res.json() };
};
const start = async (fileName, size, overwrite = false) => (await call('start', { fileName, size: String(size), ...(overwrite ? { overwrite: '1' } : {}) })).body.uploadId;
const chunk = (uploadId, offset, text) => call('chunk', { uploadId, offset: String(offset) }, Buffer.from(text));
const state = id => JSON.parse(fs.readFileSync(path.join(partDir, id, 'state.json'), 'utf8'));
const child = (mode, id, gate, offset, data) =>
  new Promise((resolve, reject) => {
    let dataFile = '';
    if (data !== undefined) {
      dataFile = path.join(root, `chunk-${Date.now()}-${Math.random()}.bin`);
      fs.writeFileSync(dataFile, data);
    }
    const c = spawn(process.execPath, [path.join(__dirname, 'fixtures', 'uploadChild.cjs'), partDir, loras(), mode, id, gate ?? '', String(offset ?? ''), dataFile], {
      env: process.env,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    let err = '';
    c.stdout.on('data', d => (out += d));
    c.stderr.on('data', d => (err += d));
    c.on('error', reject);
    c.on('exit', code => resolve({ code, out: out ? JSON.parse(out) : null, err }));
  });
const gateFile = () => path.join(fs.mkdtempSync(path.join(SCRATCH, 'upgate-')), 'go');
const waitFile = async f => {
  while (!fs.existsSync(f)) await new Promise(r => setTimeout(r, 10));
};

test('normal client protocol through the actual route; a retried finish returns the first result', async () => {
  const id = await start('normal_lora.safetensors', 8);
  assert.equal((await chunk(id, 0, 'ABCD')).status, 200);
  assert.equal((await chunk(id, 4, 'EFGH')).status, 200);
  const fin = await call('finish', { uploadId: id });
  assert.equal(fin.status, 200, JSON.stringify(fin.body));
  assert.equal(fs.readFileSync(path.join(loras(), 'normal_lora.safetensors'), 'utf8'), 'ABCDEFGH');
  assert.deepEqual((await call('finish', { uploadId: id })).body, fin.body, 'lost-reply retry');
  assert.equal(fs.existsSync(path.join(partDir, id, 'data.part')), false);
  assert.equal(state(id).state, 'published');
  assert.equal((await chunk(id, 0, 'ABCD')).status, 409, 'no chunk after publication');
  assert.equal((await call('start', { fileName: 'normal_lora.safetensors', size: '8' })).status, 409, 'start refuses an existing name');
});

test('two finishes to one name from two Node processes: exactly one publishes, with its own complete bytes; nothing overwritten', async () => {
  const a = await start('race_lora.safetensors', 4);
  const b = await start('race_lora.safetensors', 4);
  await chunk(a, 0, 'AAAA');
  await chunk(b, 0, 'BBBB');
  const [ra, rb] = await Promise.all([child('finish', a), child('finish', b)]);
  const statuses = [ra.out?.status, rb.out?.status].sort();
  assert.deepEqual(statuses, [200, 409], JSON.stringify([ra, rb]));
  const winner = ra.out.status === 200 ? 'AAAA' : 'BBBB';
  const loser = ra.out.status === 200 ? b : a;
  assert.equal(fs.readFileSync(path.join(loras(), 'race_lora.safetensors'), 'utf8'), winner);
  assert.equal(state(loser).state, 'receiving', 'the loser keeps its data');
  assert.equal(fs.readFileSync(path.join(partDir, loser, 'data.part'), 'utf8'), winner === 'AAAA' ? 'BBBB' : 'AAAA');
});

test('a finish from another process waits for a chunk that holds the lock mid-write, then publishes the complete file', async () => {
  const id = await start('held_lora.safetensors', 8);
  await chunk(id, 0, 'ABCD');
  const gate = gateFile();
  const writer = child('chunk-hold', id, gate, 4, Buffer.from('EFGH'));
  await waitFile(`${gate}.holding`);
  assert.equal((await up.finishUpload(partDir, loras(), id, { lockWaitMs: 0 })).status, 503, 'busy, not a hang');
  const finishing = up.finishUpload(partDir, loras(), id, { lockWaitMs: 30000 });
  await new Promise(r => setTimeout(r, 200));
  fs.writeFileSync(gate, '');
  assert.equal((await writer).out.status, 200);
  const fin = await finishing;
  assert.equal(fin.status, 200, JSON.stringify(fin.body));
  assert.equal(fs.readFileSync(path.join(loras(), 'held_lora.safetensors'), 'utf8'), 'ABCDEFGH');
});

test('a cancel from another process waits for the held chunk, then wins; later chunk/finish are refused', async () => {
  const id = await start('cancel_lora.safetensors', 8);
  const gate = gateFile();
  const writer = child('chunk-hold', id, gate, 0, Buffer.from('ABCD'));
  await waitFile(`${gate}.holding`);
  const cancelling = child('cancel', id);
  await new Promise(r => setTimeout(r, 200));
  fs.writeFileSync(gate, '');
  assert.equal((await writer).out.status, 200);
  assert.equal((await cancelling).out.status, 200);
  assert.equal(state(id).state, 'cancelled');
  assert.equal(fs.existsSync(path.join(partDir, id, 'data.part')), false);
  assert.equal((await chunk(id, 4, 'EFGH')).status, 410);
  assert.equal((await call('finish', { uploadId: id })).status, 410);
});

test('holes, size violations, conflicting retries and overlaps are refused; an identical retry is accepted', async () => {
  const id = await start('holes_lora.safetensors', 8);
  assert.equal((await chunk(id, 4, 'EFGH')).status, 200);
  const fin = await call('finish', { uploadId: id });
  assert.equal(fin.status, 409);
  assert.deepEqual(fin.body.missing, [[0, 4]]);
  assert.equal(fs.existsSync(path.join(loras(), 'holes_lora.safetensors')), false, 'nothing published with a hole');
  assert.equal((await chunk(id, 6, 'XYZW')).status, 400, 'beyond the declared size');
  assert.equal((await chunk(id, 4, 'EFGH')).body.duplicate, true);
  assert.equal((await chunk(id, 4, 'eFGH')).status, 409, 'same range, different bytes');
  assert.equal((await chunk(id, 2, 'cdEF')).status, 409, 'overlap');
  assert.equal((await chunk(id, 0, 'ABCD')).status, 200);
  assert.equal((await call('finish', { uploadId: id })).status, 200);
  assert.equal(fs.readFileSync(path.join(loras(), 'holes_lora.safetensors'), 'utf8'), 'ABCDEFGH');
});

test('a finish process dying right after the hard link (before state.json says published) is completed by the next finish', async () => {
  const id = await start('linkcrash_lora.safetensors', 4);
  await chunk(id, 0, 'LINK');
  const r = await child('finish-exit-after-link', id);
  assert.equal(r.code, 0, r.err);
  assert.equal(state(id).state, 'publishing');
  assert.equal(fs.readFileSync(path.join(loras(), 'linkcrash_lora.safetensors'), 'utf8'), 'LINK');
  const fin = await call('finish', { uploadId: id });
  assert.equal(fin.status, 200, JSON.stringify(fin.body));
  assert.equal(state(id).state, 'published');
  assert.equal(fs.existsSync(path.join(partDir, id, 'data.part')), false);
});

test('an overwrite finish dying right after the rename is completed; a replaced destination with the data gone is indeterminate, never overwritten', async () => {
  fs.writeFileSync(path.join(loras(), 'over_lora.safetensors'), 'OLD!');
  const id = await start('over_lora.safetensors', 4, true);
  await chunk(id, 0, 'NEW!');
  const r = await child('finish-exit-after-rename', id);
  assert.equal(r.code, 0, r.err);
  assert.equal(state(id).state, 'publishing');
  assert.equal((await call('finish', { uploadId: id })).status, 200);
  assert.equal(fs.readFileSync(path.join(loras(), 'over_lora.safetensors'), 'utf8'), 'NEW!');
  // the same interruption, then the destination is replaced by someone else before recovery
  const id2 = await start('over_lora.safetensors', 4, true);
  await chunk(id2, 0, 'MINE');
  await child('finish-exit-after-rename', id2);
  fs.unlinkSync(path.join(loras(), 'over_lora.safetensors'));
  fs.writeFileSync(path.join(loras(), 'over_lora.safetensors'), 'THEM');
  const fin = await call('finish', { uploadId: id2 });
  assert.equal(fin.status, 500);
  assert.equal(fin.body.indeterminate, true);
  assert.equal(state(id2).state, 'indeterminate');
  assert.equal(fs.readFileSync(path.join(loras(), 'over_lora.safetensors'), 'utf8'), 'THEM', 'not overwritten');
});

test('a file system without hard links: explicit unsupported result, data retained, no copy fallback', async () => {
  const id = await start('nolink_lora.safetensors', 4);
  await chunk(id, 0, 'DATA');
  const eperm = async () => {
    const e = new Error('EPERM: operation not permitted, link');
    e.code = 'EPERM';
    throw e;
  };
  const fin = await up.finishUpload(partDir, loras(), id, { link: eperm });
  assert.equal(fin.status, 500);
  assert.equal(fin.body.unsupported, true);
  assert.equal(state(id).state, 'publish_unsupported');
  assert.equal(fs.readFileSync(path.join(partDir, id, 'data.part'), 'utf8'), 'DATA');
  assert.equal(fs.existsSync(path.join(loras(), 'nolink_lora.safetensors')), false);
});

test('cancel with a failed data removal stays cancelled and reports the cleanup failure', async () => {
  const id = await start('cleanup_lora.safetensors', 4);
  await chunk(id, 0, 'DATA');
  const busy = async () => {
    const e = new Error('EBUSY: resource busy');
    e.code = 'EBUSY';
    throw e;
  };
  const out = await up.cancelUpload(partDir, id, { unlink: busy });
  assert.equal(out.status, 200);
  assert.match(out.body.cleanup_error, /EBUSY/);
  assert.equal(state(id).state, 'cancelled');
  assert.equal((await call('finish', { uploadId: id })).status, 410);
  assert.equal((await chunk(id, 0, 'DATA')).status, 410);
});

test('a write interrupted mid-chunk is not acknowledged or recorded; the retry of the same range succeeds', async () => {
  const id = await start('partial_lora.safetensors', 4);
  const fail = { hooks: { duringWrite: () => Promise.reject(new Error('injected I/O failure mid-chunk')) } };
  await assert.rejects(up.writeChunk(partDir, id, 0, Buffer.from('ABCD'), fail), /injected/);
  assert.deepEqual(state(id).chunks, {}, 'nothing recorded for the failed write');
  assert.equal((await chunk(id, 0, 'ABCD')).status, 200);
  assert.equal((await call('finish', { uploadId: id })).status, 200);
});

test('expiry removes only old sessions that are not publishing and not busy, under each session lock', async () => {
  const old = await start('old_lora.safetensors', 4);
  const pub = await start('pub_lora.safetensors', 4);
  const busy = await start('busy_lora.safetensors', 4);
  for (const id of [old, pub, busy]) {
    const st = state(id);
    st.created_at = Date.now() - 2 * 24 * 3600 * 1000;
    if (id === pub) st.state = 'publishing';
    fs.writeFileSync(path.join(partDir, id, 'state.json'), JSON.stringify(st));
  }
  const gate = gateFile();
  const holder = child('chunk-hold', busy, gate, 0, Buffer.from('DATA'));
  await waitFile(`${gate}.holding`);
  await up.removeStaleSessions(partDir);
  assert.equal(fs.existsSync(path.join(partDir, old, 'state.json')), false, 'old session removed');
  assert.equal(fs.existsSync(path.join(partDir, pub, 'state.json')), true, 'publishing kept for recovery');
  assert.equal(fs.existsSync(path.join(partDir, busy, 'state.json')), true, 'busy session untouched');
  fs.writeFileSync(gate, '');
  assert.equal((await holder).out.status, 200);
  assert.equal((await chunk(old, 0, 'DATA')).status, 404);
});

test('cancel commit ordering (Codex upload-cancel-commit-probe): a failed "cancelled" write removes nothing; a failed note write after cleanup keeps it cancelled', async () => {
  const id = await start('cancelorder_lora.safetensors', 4);
  await chunk(id, 0, 'DATA');
  const eio = async () => {
    const e = new Error('EIO: injected state write failure');
    e.code = 'EIO';
    throw e;
  };
  await assert.rejects(up.cancelUpload(partDir, id, { saveState: eio }), /EIO/);
  assert.equal(state(id).state, 'receiving', 'nothing decided');
  assert.equal(fs.readFileSync(path.join(partDir, id, 'data.part'), 'utf8'), 'DATA', 'nothing removed');
  assert.equal((await chunk(id, 0, 'DATA')).body.duplicate, true, 'the acknowledged bytes still exist');
  // cancelled is written; cleanup fails; the cleanup note cannot be written either
  let writes = 0;
  const failSecond = async (file, st) => {
    writes += 1;
    if (writes > 1) return eio();
    fs.writeFileSync(file, JSON.stringify(st));
  };
  const busy = async () => {
    const e = new Error('EBUSY: injected');
    e.code = 'EBUSY';
    throw e;
  };
  const out = await up.cancelUpload(partDir, id, { saveState: failSecond, unlink: busy });
  assert.equal(out.status, 200);
  assert.match(out.body.cleanup_error, /EBUSY/);
  assert.equal(state(id).state, 'cancelled', 'terminal status persisted before cleanup');
  assert.equal((await chunk(id, 0, 'DATA')).status, 410);
  // a later Cancel retries the cleanup under the same lock
  const again = await up.cancelUpload(partDir, id);
  assert.equal(again.status, 200);
  assert.equal(again.body.cleanup_error, undefined);
  assert.equal(fs.existsSync(path.join(partDir, id, 'data.part')), false);
});

test('expiry keeps publish_unsupported and indeterminate sessions (their data is the promised evidence) until an explicit Cancel', async () => {
  const kept = [];
  for (const st8 of ['publish_unsupported', 'indeterminate']) {
    const id = await start(`${st8}_lora.safetensors`, 4);
    await chunk(id, 0, 'KEEP');
    const st = state(id);
    st.state = st8;
    st.created_at = Date.now() - 3 * 24 * 3600 * 1000;
    fs.writeFileSync(path.join(partDir, id, 'state.json'), JSON.stringify(st));
    kept.push(id);
  }
  await up.removeStaleSessions(partDir);
  for (const id of kept) assert.equal(fs.readFileSync(path.join(partDir, id, 'data.part'), 'utf8'), 'KEEP');
  assert.equal((await up.cancelUpload(partDir, kept[0])).status, 200);
  assert.equal(fs.existsSync(path.join(partDir, kept[0], 'data.part')), false);
});
