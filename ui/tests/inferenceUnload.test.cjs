// Generate-page model unload: the page must expose the engine's documented POST /unload control,
// and the existing inference proxy must forward that request to the active engine.
//   node --test --test-concurrency=1 ui/tests/inferenceUnload.test.cjs
const { test, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const ts = require('./harness/tsrequire.cjs');

class TestResponse {
  constructor(body, init = {}) {
    this.body = body;
    this.status = init.status || 200;
  }
  static json(body, init = {}) {
    return new TestResponse(body, init);
  }
}

ts.mock('next/server', { NextRequest: class {}, NextResponse: TestResponse });
ts.mock('@/server/prisma', { __esModule: true, default: { job: { findMany: async () => [] } } });
ts.mock('@/server/settings', { getTrainingFolder: async () => process.cwd() });
const engine = { endpoint: { host: '127.0.0.1', port: 9876, token: 'fixture-token' }, jobId: 'J1', status: 'running' };
ts.mock('@/server/inferenceEngine', {
  listEngines: async () => [engine],
  resolveEngine: async () => engine,
});

const route = require(ts.uiPath('src/app/api/inference/[...path]/route.ts'));
const originalFetch = global.fetch;
after(() => {
  global.fetch = originalFetch;
});

test('Generate page has an explicit Unload model action wired to the engine /unload endpoint', () => {
  const source = fs.readFileSync(ts.uiPath('src/app/generate/page.tsx'), 'utf8');
  assert.match(source, /const unloadModel = async \(\) =>/);
  assert.match(source, /apiClient\.post\(proxy\('unload'\)\)/);
  assert.match(source, /Unload model/);
  assert.match(source, /health\?\.active\?\.arch/);
});

test('inference proxy forwards POST /api/inference/unload with the selected job to the engine', async () => {
  let request = null;
  global.fetch = async (url, init) => {
    request = { url: String(url), init };
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
  const incoming = {
    method: 'POST',
    nextUrl: { searchParams: new URLSearchParams('job=J1') },
    headers: new Headers(),
    arrayBuffer: async () => new ArrayBuffer(0),
  };
  const response = await route.POST(incoming, { params: { path: ['unload'] } });
  assert.equal(response.status, 200);
  assert.equal(request.url, 'http://127.0.0.1:9876/unload');
  assert.equal(request.init.method, 'POST');
  assert.equal(request.init.headers.get('x-engine-token'), 'fixture-token');
});
