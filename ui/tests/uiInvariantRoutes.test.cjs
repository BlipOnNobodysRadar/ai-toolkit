// Regression tests for SEC-01, SEC-02 and UPL-01: actual transpiled middleware and mutation routes on real temp files.
//   node --test --test-concurrency=1 ui/tests/uiInvariantRoutes.test.cjs
const { test, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const ts = require('./harness/tsrequire.cjs');
const { SCRATCH } = require('./harness/env.cjs');

class NextResponse {
  constructor(body, init = {}) {
    this.body = body;
    this.status = init.status || 200;
    this.headers = init.headers;
  }
  static json(body, init = {}) {
    return new NextResponse(body, init);
  }
  static next() {
    return { next: true, status: 200 };
  }
}
class NextRequest {}
ts.mock('next/server', { NextResponse, NextRequest });

const settings = { datasets: '', training: '' };
ts.mock('@/server/settings', {
  getDatasetsRoot: async () => settings.datasets,
  getTrainingFolder: async () => settings.training,
});

ts.mock('@/server/prisma', { job: { findMany: async () => [] } });

const middleware = require(ts.uiPath('src/middleware.ts')).middleware;
const imgDelete = require(ts.uiPath('src/app/api/img/delete/route.ts'));
const imgCaption = require(ts.uiPath('src/app/api/img/caption/route.ts'));
const filesDelete = require(ts.uiPath('src/app/api/files/delete/route.ts'));
const upload = require(ts.uiPath('src/app/api/datasets/upload/route.ts'));

const scratchRoots = [];
const makeRoot = label => {
  const dir = fs.mkdtempSync(path.join(SCRATCH, `ui-invariant-${label}-`));
  scratchRoots.push(dir);
  return dir;
};
after(() => {
  for (const dir of scratchRoots.reverse()) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // best effort; temp fixture only
    }
  }
});

const responseJson = async res => {
  if (res.body && typeof res.body.json === 'function') return res.body.json();
  return res.body;
};
const post = (route, body) => route.POST({ json: async () => body });
const middlewareRequest = (method, pathname, token, readKey) => ({
  method,
  headers: new Headers(token ? { Authorization: `Bearer ${token}` } : {}),
  nextUrl: { pathname, searchParams: new URLSearchParams(readKey ? { k: readKey } : {}) },
});
const uploadRequest = (datasetName, entries) => ({
  formData: async () => {
    const files = entries.map(([name, value]) => ({
      name,
      arrayBuffer: async () => Buffer.from(value),
    }));
    return {
      getAll: key => (key === 'files' ? files : []),
      get: key => (key === 'datasetName' ? datasetName : null),
    };
  },
});

// Upstream 73c84b09 (merged 20 Nov 2026): with a password set, media reads no longer pass bare. They need the
// rolling read key (?k=), the bearer token, or AI_TOOLKIT_DISABLE_READ_PROTECTION=1. The read key never opens a
// mutating route.
test('middleware: public media reads need a live read key; mutating routes require the configured bearer token', async () => {
  const { getReadKeys } = require(ts.uiPath('src/server/readKey.ts'));
  const names = ['AI_TOOLKIT_AUTH', 'AI_TOOLKIT_READ_SECRET', 'AI_TOOLKIT_DISABLE_READ_PROTECTION'];
  const saved = Object.fromEntries(names.map(name => [name, process.env[name]]));
  const reads = [
    ['GET', '/api/img/a.png'],
    ['HEAD', '/api/files/a.png'],
    ['GET', '/api/audio/art/a.png'],
  ];
  const protectedRoutes = [
    ['POST', '/api/img/delete'],
    ['POST', '/api/files/delete'],
    ['POST', '/api/img/caption'],
    ['PATCH', '/api/files/delete'],
    ['DELETE', '/api/img/a.png'],
    ['GET', '/api/img-other/a.png'],
    ['GET', '/api/jobs'],
  ];
  try {
    process.env.AI_TOOLKIT_AUTH = 'fixture-only-token';
    process.env.AI_TOOLKIT_READ_SECRET = 'fixture-only-read-secret';
    delete process.env.AI_TOOLKIT_DISABLE_READ_PROTECTION;
    const [currentKey, previousKey] = await getReadKeys();
    assert.match(currentKey, /^[0-9a-f]{32}$/);

    for (const [method, url] of reads) {
      assert.equal((await middleware(middlewareRequest(method, url))).status, 401, `${method} ${url} bare`);
      assert.equal((await middleware(middlewareRequest(method, url, null, 'f'.repeat(32)))).status, 401, `${method} ${url} wrong key`);
      assert.equal((await middleware(middlewareRequest(method, url, null, currentKey))).next, true, `${method} ${url} key`);
      assert.equal((await middleware(middlewareRequest(method, url, null, previousKey))).next, true, `${method} ${url} previous key`);
      assert.equal((await middleware(middlewareRequest(method, url, 'fixture-only-token'))).next, true, `${method} ${url} token`);
    }
    for (const [method, url] of protectedRoutes) {
      assert.equal((await middleware(middlewareRequest(method, url))).status, 401, `${method} ${url} must be protected`);
      // a read key is for reads on the media prefixes only
      assert.equal((await middleware(middlewareRequest(method, url, null, currentKey))).status, 401, `${method} ${url} with a read key`);
    }
    assert.equal((await middleware(middlewareRequest('POST', '/api/img/delete', 'fixture-only-token'))).next, true);
    assert.equal((await middleware(middlewareRequest('GET', '/api/jobs', 'fixture-only-token'))).next, true);

    // no launch secret: no key validates, reads fall back to the token
    delete process.env.AI_TOOLKIT_READ_SECRET;
    assert.equal((await middleware(middlewareRequest('GET', '/api/img/a.png', null, currentKey))).status, 401);

    // the opt-out restores bare reads, and still nothing else
    process.env.AI_TOOLKIT_DISABLE_READ_PROTECTION = '1';
    for (const [method, url] of reads) {
      assert.equal((await middleware(middlewareRequest(method, url))).next, true, `${method} ${url} opt-out`);
    }
    for (const [method, url] of protectedRoutes) {
      assert.equal((await middleware(middlewareRequest(method, url))).status, 401, `${method} ${url} opt-out`);
    }

    delete process.env.AI_TOOLKIT_AUTH;
    delete process.env.AI_TOOLKIT_DISABLE_READ_PROTECTION;
    assert.equal((await middleware(middlewareRequest('POST', '/api/img/delete'))).next, true, 'local no-auth mode stays open');
  } finally {
    for (const name of names) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
  }
});

test('img delete rejects traversal and sibling-prefix paths, and validates both image and caption before deleting', async () => {
  const root = makeRoot('delete');
  const outside = makeRoot('delete-outside');
  settings.datasets = root;
  settings.training = makeRoot('training-delete');
  const victim = path.join(outside, 'victim.png');
  const victimCaption = path.join(outside, 'victim.txt');
  fs.writeFileSync(victim, 'outside image');
  fs.writeFileSync(victimCaption, 'outside caption');

  const traversal = path.join(root, '..', path.basename(outside), 'victim.png');
  let res = await post(imgDelete, { imgPath: traversal });
  assert.equal(res.status, 400);
  assert.equal(fs.existsSync(victim), true);
  assert.equal(fs.existsSync(victimCaption), true);

  const sibling = `${root}-sibling`;
  fs.mkdirSync(sibling);
  scratchRoots.push(sibling);
  fs.writeFileSync(path.join(sibling, 'victim.png'), 'sibling image');
  res = await post(imgDelete, { imgPath: path.join(sibling, 'victim.png') });
  assert.equal(res.status, 400);
  assert.equal(fs.existsSync(path.join(sibling, 'victim.png')), true);

  // Nested junction escaping the configured root is rejected for the image and (separately) caption targets.
  const linkedDir = path.join(root, 'linked');
  fs.symlinkSync(outside, linkedDir, 'junction');
  res = await post(imgDelete, { imgPath: path.join(linkedDir, 'victim.png') });
  assert.equal(res.status, 400);
  assert.equal(fs.existsSync(victim), true);

  // A nested directory link for the caption destination is also rejected before the image is removed.
  const safeImage = path.join(root, 'still-here.png');
  fs.writeFileSync(safeImage, 'safe');
  fs.symlinkSync(outside, path.join(root, 'caption-link'), 'junction');
  res = await post(imgCaption, { imgPath: path.join(root, 'caption-link', 'victim.png'), caption: 'x', ext: 'txt' });
  assert.equal(res.status, 400);
  assert.equal(fs.existsSync(victim), true);
});

test('a configured root that is itself a junction remains usable; caption writes reject arbitrary extensions', async () => {
  const actualRoot = makeRoot('root-junction-target');
  const configuredJunction = `${actualRoot}-junction`;
  fs.symlinkSync(actualRoot, configuredJunction, 'junction');
  scratchRoots.push(configuredJunction);
  settings.datasets = configuredJunction;
  settings.training = makeRoot('training-junction');

  const image = path.join(actualRoot, 'photo.png');
  fs.writeFileSync(image, 'image');
  const caption = path.join(actualRoot, 'photo.txt');
  fs.writeFileSync(caption, 'old');

  let res = await post(imgCaption, {
    imgPath: path.join(configuredJunction, 'photo.png'),
    caption: 'new caption',
    ext: '.TXT',
  });
  assert.equal(res.status, 200);
  assert.equal(fs.readFileSync(caption, 'utf8'), 'new caption');

  const outside = makeRoot('caption-outside');
  const arbitrary = path.join(outside, 'photo.bat');
  fs.writeFileSync(arbitrary, 'must not change');
  res = await post(imgCaption, {
    imgPath: path.join(configuredJunction, 'photo.png'),
    caption: 'overwrite?',
    ext: 'bat',
  });
  assert.equal(res.status, 400);
  assert.equal(fs.readFileSync(arbitrary, 'utf8'), 'must not change');
  assert.equal(fs.existsSync(path.join(actualRoot, 'photo.bat')), false);

  res = await post(imgDelete, { imgPath: path.join(configuredJunction, 'photo.png') });
  assert.equal(res.status, 200);
  assert.equal(fs.existsSync(image), false);
  assert.equal(fs.existsSync(caption), false);
});

test('files delete rejects physical escapes through nested junctions but accepts a linked configured root', async () => {
  const actualRoot = makeRoot('files-root-target');
  const configuredJunction = `${actualRoot}-junction`;
  fs.symlinkSync(actualRoot, configuredJunction, 'junction');
  scratchRoots.push(configuredJunction);
  const outside = makeRoot('files-outside');
  settings.datasets = configuredJunction;
  settings.training = makeRoot('files-training');

  const victim = path.join(outside, 'victim.png');
  fs.writeFileSync(victim, 'outside');
  const linkedDir = path.join(actualRoot, 'linked');
  fs.symlinkSync(outside, linkedDir, 'junction');
  let res = await post(filesDelete, { filePath: path.join(configuredJunction, 'linked', 'victim.png') });
  assert.equal(res.status, 403);
  assert.equal(fs.existsSync(victim), true);

  const local = path.join(actualRoot, 'local.png');
  fs.writeFileSync(local, 'inside');
  res = await post(filesDelete, { filePath: path.join(configuredJunction, 'local.png') });
  assert.equal(res.status, 200);
  assert.equal(fs.existsSync(local), false);
});

test('dataset upload refuses sanitization collisions and concurrent publication without overwriting', async () => {
  const root = makeRoot('upload');
  settings.datasets = root;
  settings.training = makeRoot('upload-training');

  let res = await upload.POST(uploadRequest('collision', [
    ['a b.png', 'first'],
    ['a_b.png', 'second'],
  ]));
  assert.equal(res.status, 409);
  assert.match((await responseJson(res)).error, /collide after sanitization/);
  assert.equal(fs.existsSync(path.join(root, 'collision')), false, 'nothing is published on a preflight collision');

  const datasetDir = path.join(root, 'existing');
  fs.mkdirSync(datasetDir);
  const existing = path.join(datasetDir, 'a_b.png');
  fs.writeFileSync(existing, 'keep me');
  res = await upload.POST(uploadRequest('existing', [['a b.png', 'replace attempt']]));
  assert.equal(res.status, 409);
  assert.equal(fs.readFileSync(existing, 'utf8'), 'keep me');

  const concurrent = await Promise.all([
    upload.POST(uploadRequest('race', [['same name.png', 'AAAA']])),
    upload.POST(uploadRequest('race', [['same_name.png', 'BBBB']])),
  ]);
  assert.deepEqual(concurrent.map(r => r.status).sort(), [200, 409], JSON.stringify(concurrent));
  const raceFiles = fs.readdirSync(path.join(root, 'race')).filter(name => !name.startsWith('.upload-'));
  assert.deepEqual(raceFiles, ['same_name.png']);
  assert.ok(['AAAA', 'BBBB'].includes(fs.readFileSync(path.join(root, 'race', 'same_name.png'), 'utf8')));
});
