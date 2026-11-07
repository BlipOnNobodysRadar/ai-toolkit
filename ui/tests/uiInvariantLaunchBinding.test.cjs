// Regression tests for UI-01: the actual launch config binding helper and launchRun pre-spawn path. No Python, GPU,
// Prisma database or child process is used; the launch result is forced to the supported-platform refusal after the
// real config has been bound and written, so the test can inspect exactly what the trainer would read.
//   node --test --test-concurrency=1 ui/tests/uiInvariantLaunchBinding.test.cjs
const { test, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const ts = require('./harness/tsrequire.cjs');
const { SCRATCH } = require('./harness/env.cjs');

const released = [];
ts.mock('./store', {
  attachRun: async () => {},
  markRelayExited: async () => {},
  markSpawnRequested: async () => false,
  noteOwnership: async () => {},
  recordRelay: async () => {},
  releaseRun: async (_db, jobId, runId, phase, info) => {
    released.push({ jobId, runId, phase, info });
  },
});

const launch = require(ts.uiPath('cron/ownership/launch.ts'));
const { bindLaunchConfig, LaunchConfigError } = require(ts.uiPath('cron/ownership/launchConfig.ts'));

const roots = [];
const makeRoot = label => {
  const dir = fs.mkdtempSync(path.join(SCRATCH, `ui-invariant-launch-${label}-`));
  roots.push(dir);
  return dir;
};
after(() => {
  for (const dir of roots.reverse()) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // temp fixture only
    }
  }
});

const jobFor = (runDir, config, name = path.basename(runDir), jobType = 'train') => ({
  id: 'job-fixture',
  name,
  run_id: 'run-fixture',
  run_dir: runDir,
  job_config: JSON.stringify(config),
  job_type: jobType,
  gpu_ids: '0',
});
const ctx = () => ({
  db: {},
  toolkitRoot: makeRoot('toolkit'),
  helperPath: 'unused-helper.py',
  helperPython: 'unused-python',
  jobPython: 'unused-pythonw',
  runPy: __filename,
  sqliteDbPath: 'fixture.sqlite',
  env: {},
  inFlight: new Set(),
  platform: 'darwin', // forces a pre-spawn refusal AFTER the real binding/config write, never a child process
});

test('launch binds a stale/mismatched training process to the exact claimed folder, with no outside write', async () => {
  const claimRoot = makeRoot('claim');
  const oldOutside = makeRoot('old-outside');
  fs.writeFileSync(path.join(oldOutside, 'do-not-touch.txt'), 'outside');
  const runDir = path.join(claimRoot, 'UI Job Name');
  const config = {
    job: 'extension',
    config: {
      name: 'saved_config_name',
      training_folder: oldOutside,
      process: [
        {
          type: 'ui_trainer',
          name: 'other_process_name',
          training_folder: oldOutside,
          sqlite_db_path: 'stale.sqlite',
          train: { steps: 1 },
        },
      ],
    },
  };
  const job = jobFor(runDir, config);
  const beforeOutside = fs.readdirSync(oldOutside);
  const result = await launch.launchRun(ctx(), job);
  assert.equal(result.result, 'not_spawned', JSON.stringify(result));
  assert.match(result.info, /not supported on darwin/);
  assert.equal(fs.readdirSync(oldOutside).join('|'), beforeOutside.join('|'), 'old Training Folder untouched');

  const files = launch.runFiles(runDir, job.run_id);
  assert.equal(fs.existsSync(files.config), true);
  const launched = JSON.parse(fs.readFileSync(files.config, 'utf8'));
  assert.equal(launched.config.name, 'UI Job Name');
  assert.equal(launched.config.process[0].name, 'UI Job Name');
  assert.equal(launched.config.process[0].training_folder, claimRoot);
  assert.equal(path.resolve(launched.config.process[0].training_folder, launched.config.process[0].name), runDir);
  assert.equal(launched.config.process[0].sqlite_db_path, 'fixture.sqlite');
  assert.equal(fs.existsSync(path.join(oldOutside, '.job_config.run-fixture.json')), false);
});

test('a linked configured root is preserved while the process save_root canonicalizes to the claim', async () => {
  const actualRoot = makeRoot('actual-root');
  const linkedRoot = `${actualRoot}-junction`;
  fs.symlinkSync(actualRoot, linkedRoot, 'junction');
  roots.push(linkedRoot);
  const runDir = path.join(linkedRoot, 'linked-job');
  const config = {
    job: 'extension',
    config: {
      name: 'wrong',
      process: [{ type: 'diffusion_trainer', training_folder: actualRoot, name: 'wrong' }],
    },
  };
  const job = jobFor(runDir, config, 'linked-job');
  const result = await launch.launchRun(ctx(), job);
  assert.equal(result.result, 'not_spawned', JSON.stringify(result));
  const launched = JSON.parse(fs.readFileSync(launch.runFiles(runDir, job.run_id).config, 'utf8'));
  const process = launched.config.process[0];
  assert.equal(process.training_folder, linkedRoot);
  assert.equal(process.name, 'linked-job');
  assert.equal(fs.realpathSync(path.resolve(process.training_folder, process.name)), fs.realpathSync(runDir));
});

test('inference engine job_folder is bound to the claim', async () => {
  const claimRoot = makeRoot('inference-claim');
  const oldOutside = makeRoot('inference-old');
  const runDir = path.join(claimRoot, 'engine-job');
  const config = {
    job: 'extension',
    config: {
      name: 'engine-name',
      process: [
        {
          type: 'InferenceEngine',
          engine: { job_folder: oldOutside, output_folder: path.join(oldOutside, 'outputs') },
        },
      ],
    },
  };
  const job = jobFor(runDir, config, 'engine-job', 'inference');
  const result = await launch.launchRun(ctx(), job);
  assert.equal(result.result, 'not_spawned', JSON.stringify(result));
  const launched = JSON.parse(fs.readFileSync(launch.runFiles(runDir, job.run_id).config, 'utf8'));
  assert.equal(launched.config.process[0].engine.job_folder, runDir);
  assert.equal(launched.config.name, 'engine-job');
  assert.equal(launched.config.process[0].name, 'engine-job');
});

test('ambiguous multi-process configs are refused before config/folder writes or spawn', async () => {
  const claimRoot = makeRoot('multi-claim');
  const runDir = path.join(claimRoot, 'multi');
  const config = {
    job: 'extension',
    config: {
      name: 'multi',
      process: [
        { type: 'ui_trainer', training_folder: claimRoot, name: 'one' },
        { type: 'ui_trainer', training_folder: claimRoot, name: 'two' },
      ],
    },
  };
  const job = jobFor(runDir, config, 'multi');
  released.length = 0;
  const result = await launch.launchRun(ctx(), job);
  assert.equal(result.result, 'not_spawned', JSON.stringify(result));
  assert.match(result.info, /exactly one output folder/);
  assert.equal(fs.existsSync(runDir), false, 'no run folder/config is created for an unsupported layout');
  assert.equal(released.length, 1);
  assert.equal(released[0].jobId, job.id);
  assert.throws(
    () => bindLaunchConfig(config, 'multi', runDir),
    err => err instanceof LaunchConfigError && /exactly one output folder/.test(err.message),
  );
});

test('unknown extension writers are refused before creating a run folder', async () => {
  const root = makeRoot('unknown');
  const runDir = path.join(root, 'job');
  const config = { job: 'extension', config: { name: 'job', process: [{ type: 'unregistered_writer' }] } };
  const result = await launch.launchRun(ctx(), jobFor(runDir, config));
  assert.equal(result.result, 'not_spawned');
  assert.match(result.info, /no supported UI output-folder binding/);
  assert.equal(fs.existsSync(runDir), false);
});

test('Windows folder casing does not change the admitted writer identity', { skip: process.platform !== 'win32' }, () => {
  const root = makeRoot('case');
  const folder = path.join(root, 'job');
  const config = { job: 'extension', config: { name: 'JOB', process: [{ type: 'ui_trainer' }] } };
  const bound = bindLaunchConfig(config, 'JOB', folder);
  assert.equal(path.relative(folder, path.resolve(bound.config.process[0].training_folder, bound.config.process[0].name)), '');
  assert.equal(bound.config.name, 'JOB', 'canonical folder spelling must not rename checkpoint prefixes');
  assert.equal(bound.config.process[0].name, 'JOB');
});
