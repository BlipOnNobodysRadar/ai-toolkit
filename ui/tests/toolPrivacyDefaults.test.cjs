const { test } = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { SCRATCH } = require('./harness/env.cjs');
const ts = require('./harness/tsrequire.cjs');

test('the Next configuration disables telemetry and incoming media URL logging by default', () => {
  delete process.env.NEXT_TELEMETRY_DISABLED;
  const config = require(ts.uiPath('next.config.ts')).default('phase-production-server');
  assert.equal(process.env.NEXT_TELEMETRY_DISABLED, '1');
  assert.equal(config.logging.incomingRequests, false);
});

test('the install and its dependency hooks inherit the Prisma telemetry preference', t => {
  const dir = fs.mkdtempSync(path.join(SCRATCH, 'install-privacy-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const probe = path.join(dir, 'npm.cjs');
  fs.writeFileSync(probe, 'process.stdout.write(JSON.stringify(process.env.CHECKPOINT_DISABLE))');
  const env = { ...process.env, npm_execpath: probe };
  delete env.CHECKPOINT_DISABLE;
  const cli = ts.uiPath('scripts/run-tool.cjs');
  const check = env => JSON.parse(execFileSync(process.execPath, [cli, 'npm', 'install'], { env, encoding: 'utf8' }));
  assert.equal(check(env), '1');
  assert.equal(check({ ...env, CHECKPOINT_DISABLE: '' }), '');
});

test('configuration viewers and editors use the local Monaco distribution', () => {
  let base;
  ts.mock('@monaco-editor/react', { loader: { config: options => { base = options.paths.vs; } } });
  require(ts.uiPath('src/components/JobConfigViewer.tsx'));
  assert.equal(base, '/monaco/vs');
  delete require.cache[require.resolve(ts.uiPath('src/components/LocalMonacoEditor.ts'))];
  base = undefined;
  require(ts.uiPath('src/components/AdvancedConfigEditor.tsx'));
  assert.equal(base, '/monaco/vs');
});
