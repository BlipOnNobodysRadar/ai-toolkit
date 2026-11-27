const { test } = require('node:test');
const assert = require('node:assert/strict');
const { setupDb, ts } = require('./harness/fixture.cjs');

test('settings keep saved credentials server-side and support preserving, replacing and clearing a token', async t => {
  const { db, trainingRoot } = await setupDb('settings-privacy');
  t.after(() => db.$disconnect());
  const route = require(ts.uiPath('src/app/api/settings/route.ts'));
  const token = () => db.settings.findUnique({ where: { key: 'HF_TOKEN' } });
  const write = body => route.POST({ json: async () => body });
  await db.settings.createMany({ data: [
    { key: 'HF_TOKEN', value: 'private-fixture-token' },
    { key: 'UNRELATED_PRIVATE_KEY', value: 'another-private-value' },
  ] });
  const response = await route.GET();
  assert.equal(response.headers.get('cache-control'), 'no-store');
  const data = await response.json();
  assert.deepEqual(Object.keys(data).sort(), ['DATASETS_FOLDER', 'HAS_HF_TOKEN', 'MODELS_PATH', 'TRAINING_FOLDER']);
  assert.equal(data.HAS_HF_TOKEN, true);
  assert.equal(data.TRAINING_FOLDER, trainingRoot);
  assert.equal((await write({ DATASETS_FOLDER: 'another-folder' })).status, 200);
  assert.equal((await token()).value, 'private-fixture-token');
  assert.equal((await write({ HF_TOKEN: 'replacement-fixture-token' })).status, 200);
  assert.equal((await token()).value, 'replacement-fixture-token');
  assert.equal((await write({ HF_TOKEN: '' })).status, 200);
  assert.equal((await (await route.GET()).json()).HAS_HF_TOKEN, false);
});

test('the browser attaches its toolkit credential only to requests to its own origin', async t => {
  const { apiClient } = require(ts.uiPath('src/utils/api.ts'));
  Object.defineProperty(globalThis, 'window', { configurable: true, value: { location: new URL('http://localhost:8675/jobs') } });
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: () => 'fixture-token' } });
  t.after(() => {
    delete globalThis.window;
    delete globalThis.localStorage;
  });
  const headers = new Map();
  const adapter = async config => {
    headers.set(config.url, config.headers.get('Authorization'));
    return { status: 200, statusText: 'OK', headers: {}, data: null, config };
  };
  await apiClient.get('/api/queue', { adapter });
  await apiClient.get('https://example.invalid/probe', { adapter });
  await apiClient.get('/api/settings', { baseURL: 'http://localhost:8080', adapter });
  assert.equal(headers.get('/api/queue'), 'Bearer fixture-token');
  assert.equal(headers.get('https://example.invalid/probe'), undefined);
  assert.equal(headers.get('/api/settings'), undefined);
});

test('file server binds local anonymous sessions to loopback and honors a configured network deployment', t => {
  const http = require('node:http');
  const entry = ts.uiPath('cron/fileServer.ts');
  // Exercise the worker startup without opening any socket or reading a developer's .env.
  ts.mock('cluster', { isPrimary: false });
  ts.mock('@next/env', { loadEnvConfig: () => ({}), resetEnv: () => {} });
  let bound;
  t.mock.method(http.Server.prototype, 'listen', function(port, host) { bound = host; return this; });
  const originalHost = process.env.AI_TOOLKIT_HOST;
  const originalAuth = process.env.AI_TOOLKIT_AUTH;
  const disconnect = process.rawListeners('disconnect');
  t.after(() => {
    for (const [key, value] of [['AI_TOOLKIT_HOST', originalHost], ['AI_TOOLKIT_AUTH', originalAuth]]) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    for (const listener of process.rawListeners('disconnect')) {
      if (!disconnect.includes(listener)) process.removeListener('disconnect', listener);
    }
  });
  delete process.env.AI_TOOLKIT_HOST;
  delete process.env.AI_TOOLKIT_AUTH;
  require(entry);
  assert.equal(bound, '127.0.0.1');
  process.env.AI_TOOLKIT_AUTH = 'test-token';
  delete require.cache[entry];
  require(entry);
  assert.equal(bound, '0.0.0.0');
  process.env.AI_TOOLKIT_HOST = '::1';
  delete require.cache[entry];
  require(entry);
  assert.equal(bound, '::1');
});
