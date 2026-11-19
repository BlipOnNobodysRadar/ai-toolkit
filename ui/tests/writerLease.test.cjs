// Cross-language exact-root writer lease regression. This test crosses the real TypeScript writerLease/store code with
// a real Python CPU subprocess and an isolated Prisma SQLite database. It deliberately uses the intended Python public
// API (lease_path/acquire/release), not the old in-root file-lock protocol. Missing or broken APIs fail the suite.
//   node --test --test-concurrency=1 ui/tests/writerLease.test.cjs
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { setupDb, makeJob, ts } = require('./harness/fixture.cjs');
const { SCRATCH, PYTHON } = require('./harness/env.cjs');

const PY_MODULE = path.resolve(__dirname, '..', '..', 'toolkit', 'save_root_lease.py');

const pythonEnv = () => ({ ...process.env, CUDA_VISIBLE_DEVICES: '-1' });

const PY_PROBE = String.raw`
import importlib.util, json, sys
module_path = sys.argv[1]
try:
    spec = importlib.util.spec_from_file_location("aitk_save_root_lease_probe", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create import spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
except BaseException as e:
    print(json.dumps({"ok": False, "missing": [], "error": "import failed: %s: %s" % (type(e).__name__, e)}))
    raise SystemExit(0)
missing = [name for name in ("lease_path", "acquire") if not callable(getattr(mod, name, None))]
has_release = any(callable(getattr(mod, name, None)) for name in ("release", "release_owner"))
if not has_release:
    missing.append("release|release_owner")
print(json.dumps({"ok": not missing, "missing": missing, "module": module_path}))
`;

// Load the pure-stdlib module directly: importing toolkit/__init__.py can pull unrelated dependencies. The holder
// reports readiness as one JSON line (no polling/sleep), blocks on stdin EOF, then releases. The one-shot mode uses
// the same public API and re-raises only to classify busy versus a real error.
const PY_CHILD = String.raw`
import importlib.util, json, os, sys

mode, module_path, root, owner, holder = sys.argv[1:6]

def emit(obj, code=0):
    sys.stdout.write(json.dumps(obj, sort_keys=True) + "\n")
    sys.stdout.flush()
    if code is not None:
        raise SystemExit(code)

try:
    spec = importlib.util.spec_from_file_location("aitk_save_root_lease_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create import spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
except BaseException as e:
    emit({"event": "error", "stage": "import", "type": type(e).__name__, "error": str(e)}, 1)

lease_path_fn = getattr(mod, "lease_path", None)
if not callable(lease_path_fn):
    emit({"event": "error", "stage": "contract", "error": "lease_path is missing"}, 1)
try:
    lease_path = lease_path_fn(root)
except BaseException as e:
    emit({"event": "error", "stage": "lease_path", "type": type(e).__name__, "error": str(e)}, 1)

def release(lease):
    try:
        mod.release(lease, holder)
        return "release"
    except BaseException as e:
        emit({"event": "error", "stage": "release", "type": type(e).__name__, "error": str(e)}, 1)

def busy_error(e):
    return isinstance(e, mod.SaveRootBusy)

if mode == "hold":
    try:
        lease = mod.acquire(root, owner, holder)
    except BaseException as e:
        emit({"event": "ready", "status": "busy" if busy_error(e) else "error",
              "type": type(e).__name__, "error": str(e), "lease_path": lease_path},
             0 if busy_error(e) else 1)
    emit({"event": "ready", "status": "held", "lease_path": lease_path, "pid": os.getpid()}, None)
    sys.stdin.buffer.read()  # explicit readiness gate: parent closes stdin for graceful release
    how = release(lease)
    emit({"event": "released", "release": how, "pid": os.getpid()})
elif mode == "once":
    try:
        lease = mod.acquire(root, owner, holder)
    except BaseException as e:
        emit({"event": "acquire", "status": "busy" if busy_error(e) else "error",
              "type": type(e).__name__, "error": str(e), "lease_path": lease_path},
             0 if busy_error(e) else 1)
    how = release(lease)
    emit({"event": "acquire", "status": "acquired", "lease_path": lease_path,
          "pid": os.getpid(), "release": how})
else:
    emit({"event": "error", "error": "unknown mode"}, 1)
`;

let F;
let store;
let writerLeasePath;
let canonicalFolder;
let PY_API = { ok: false, missing: [], error: 'Python contract probe did not run' };
const liveChildren = new Set();

function startPython(script, args) {
  const child = spawn(PYTHON, ['-I', '-S', '-B', '-u', '-c', script, ...args], {
    cwd: SCRATCH,
    env: pythonEnv(),
    windowsHide: true,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  liveChildren.add(child);
  let stdout = '';
  let stderr = '';
  let buffer = '';
  let exitInfo = null;
  const lines = [];
  const waiters = [];
  const exitError = () =>
    new Error(`Python exited before expected JSON (code=${exitInfo?.code}, signal=${exitInfo?.signal})\n` +
      `stdout:\n${stdout}\nstderr:\n${stderr}`);

  const deliver = line => {
    const waiter = waiters.shift();
    if (!waiter) {
      lines.push(line);
      return;
    }
    clearTimeout(waiter.timer);
    try {
      waiter.resolve(JSON.parse(line));
    } catch (e) {
      waiter.reject(new Error(`Invalid JSON from Python: ${line}\nstdout:\n${stdout}\nstderr:\n${stderr}`));
    }
  };

  child.stdout.on('data', d => {
    stdout += d;
    buffer += d;
    let i;
    while ((i = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, i).trim();
      buffer = buffer.slice(i + 1);
      if (line) deliver(line);
    }
  });
  child.stderr.on('data', d => (stderr += d));
  child.on('error', error => {
    exitInfo = { code: null, signal: null, error };
    liveChildren.delete(child);
    for (const waiter of waiters.splice(0)) {
      clearTimeout(waiter.timer);
      waiter.reject(error);
    }
  });
  child.on('exit', (code, signal) => {
    exitInfo = { code, signal };
    liveChildren.delete(child);
    for (const waiter of waiters.splice(0)) {
      clearTimeout(waiter.timer);
      waiter.reject(exitError());
    }
  });

  const next = (timeoutMs = 30000) =>
    new Promise((resolve, reject) => {
      if (lines.length) {
        const line = lines.shift();
        try {
          resolve(JSON.parse(line));
        } catch (e) {
          reject(new Error(`Invalid JSON from Python: ${line}\nstdout:\n${stdout}\nstderr:\n${stderr}`));
        }
        return;
      }
      if (exitInfo) {
        reject(exitError());
        return;
      }
      const waiter = { resolve, reject, timer: null };
      waiter.timer = setTimeout(() => {
        const i = waiters.indexOf(waiter);
        if (i >= 0) waiters.splice(i, 1);
        reject(new Error(`Timed out after ${timeoutMs}ms waiting for Python JSON\nstdout:\n${stdout}\nstderr:\n${stderr}`));
      }, timeoutMs);
      waiters.push(waiter);
    });

  const waitExit = (timeoutMs = 10000) => {
    if (exitInfo) return Promise.resolve(exitInfo);
    return new Promise((resolve, reject) => {
      let settled = false;
      const onExit = (code, signal) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve({ code, signal });
      };
      child.once('exit', onExit);
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        child.removeListener('exit', onExit);
        reject(new Error(`Timed out after ${timeoutMs}ms waiting for Python exit\nstdout:\n${stdout}\nstderr:\n${stderr}`));
      }, timeoutMs);
    });
  };

  return { child, next, waitExit, stdout: () => stdout, stderr: () => stderr };
}

async function stopGracefully(handle, timeoutMs = 5000) {
  if (!handle) return null;
  try {
    handle.child.stdin.end();
  } catch {
    // already closed
  }
  try {
    return await handle.waitExit(timeoutMs);
  } catch {
    try {
      handle.child.kill();
    } catch {
      // already gone
    }
    try {
      return await handle.waitExit(2000);
    } catch {
      return null;
    }
  }
}

async function probePythonContract() {
  const handle = startPython(PY_PROBE, [PY_MODULE]);
  try {
    handle.child.stdin.end();
  } catch {
    // spawn failure is reported by the exit/JSON path
  }
  try {
    const line = await handle.next(30000);
    const exit = await handle.waitExit(10000);
    if (exit.code !== 0) {
      return { ok: false, missing: [], error: `Python probe exited ${exit.code}: ${handle.stderr()}` };
    }
    return line;
  } catch (e) {
    await stopGracefully(handle);
    return { ok: false, missing: [], error: String(e?.message ?? e) };
  }
}

const folderBytes = (dir, bytes) => {
  const file = path.join(dir, 'checkpoints', 'sentinel.ckpt');
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, bytes);
  fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify({ sentinel: 'config', keep: true }));
  return file;
};

const sortedClaims = () => F.db.runClaim.findMany({ orderBy: { resource: 'asc' } });
const deleteLockPath = (trainingRoot, jobId) =>
  path.join(trainingRoot, store.LOCK_DIR, `delete.${jobId.replace(/[^\w-]/g, '_')}.lock`);

before(async () => {
  F = await setupDb('writerlease');
  store = require(ts.uiPath('cron/ownership/store.ts'));
  writerLeasePath = require(ts.uiPath('cron/ownership/writerLease.ts')).writerLeasePath;
  canonicalFolder = require(ts.uiPath('cron/ownership/resources.ts')).canonicalFolder;
  PY_API = await probePythonContract();
  assert.equal(PY_API.ok, true, JSON.stringify(PY_API));
}, { timeout: 180000 });

after(async () => {
  for (const child of [...liveChildren]) {
    try {
      child.stdin.end();
    } catch {
      // ignore
    }
    try {
      child.kill();
    } catch {
      // ignore
    }
  }
  liveChildren.clear();
  if (F?.db) await F.db.$disconnect();
});

test('Python save_root_lease implements the intended public API (direct importlib, pure stdlib)', { timeout: 30000 }, t => {
  t.diagnostic(JSON.stringify(PY_API));
  assert.equal(PY_API.ok, true, JSON.stringify(PY_API));
});

test('a real Python holder returns 409 from actual store.deleteJob, preserves stopped row/claims/checkpoint bytes, and matches TS path through a junction root', { timeout: 180000 }, async t => {

  const suffix = `${process.pid}-${Date.now()}`;
  const targetRoot = fs.mkdtempSync(path.join(SCRATCH, 'writerlease-target-'));
  const aliasRoot = path.join(F.trainingRoot, `root-alias-${suffix}`);
  const jobName = `leasehold-${suffix}`;
  const outputDir = path.join(targetRoot, jobName);
  const aliasOutput = path.join(aliasRoot, jobName);

  fs.symlinkSync(targetRoot, aliasRoot, process.platform === 'win32' ? 'junction' : 'dir');
  const sentinel = Buffer.from([0, 1, 2, 3, 0, 255, 254, 7, 42, 99]);
  const checkpoint = folderBytes(outputDir, sentinel);
  const job = await makeJob(F.db, { name: jobName, status: 'stopped' });
  // A claim outside the deleted folder on purpose: a lease refusal must not mutate any claim rows, and the later
  // successful deletion must not delete unrelated claims either.
  const unrelatedResource = `gpu:writerlease-${suffix}`;
  await F.db.runClaim.create({
    data: { resource: unrelatedResource, job_id: `other-${suffix}`, run_id: `other-run-${suffix}`, kind: 'run' },
  });
  const rowBefore = await F.db.job.findUnique({ where: { id: job.id } });
  const claimsBefore = await sortedClaims();
  const lockPath = writerLeasePath(aliasOutput);
  const owner = `owner:${jobName}`;
  const holder = `holder:${process.pid}`;

  const py = startPython(PY_CHILD, ['hold', PY_MODULE, aliasOutput, owner, holder]);
  t.after(async () => {
    await stopGracefully(py);
    try {
      fs.rmSync(aliasRoot, { recursive: true, force: true });
    } catch {
      // fixture cleanup
    }
    try {
      fs.rmSync(targetRoot, { recursive: true, force: true });
    } catch {
      // fixture cleanup
    }
  });

  const ready = await py.next(30000);
  assert.equal(ready.event, 'ready', JSON.stringify(ready));
  assert.equal(ready.status, 'held', `holder did not acquire: ${JSON.stringify(ready)}; stderr=${py.stderr()}`);
  assert.notEqual(ready.pid, process.pid);
  assert.equal(ready.lease_path, lockPath, 'Python lease_path must equal TS writerLeasePath for the junction spelling');

  const refused = await store.deleteJob(F.db, job.id, aliasRoot);
  assert.equal(refused.status, 409, JSON.stringify(refused.body));
  assert.match(refused.body.error, /owns this output folder|training run|another deletion/i);

  const rowHeld = await F.db.job.findUnique({ where: { id: job.id } });
  assert.equal(rowHeld.status, rowBefore.status);
  assert.equal(rowHeld.info, rowBefore.info);
  assert.deepEqual(await sortedClaims(), claimsBefore, 'the lease refusal must preserve claims byte-for-byte');
  assert.equal(fs.existsSync(aliasOutput), true);
  assert.equal(fs.existsSync(outputDir), true);
  assert.deepEqual(fs.readFileSync(checkpoint), sentinel, 'sentinel checkpoint bytes changed while Python held the lease');
  assert.equal(fs.existsSync(deleteLockPath(aliasRoot, job.id)), true);

  try {
    py.child.stdin.end();
  } catch {
    // handled by exit below
  }
  const released = await py.next(15000);
  assert.equal(released.event, 'released', JSON.stringify(released));
  const exit = await py.waitExit(15000);
  assert.equal(exit.code, 0, `holder exit failed: ${py.stderr()}`);

  const retry = await store.deleteJob(F.db, job.id, aliasRoot);
  assert.equal(retry.status, 200, JSON.stringify(retry.body));
  assert.equal(await F.db.job.findUnique({ where: { id: job.id } }), null);
  assert.equal(fs.existsSync(aliasOutput), false, 'the aliased job folder must be removed');
  assert.equal(fs.existsSync(outputDir), false, 'the canonical target job folder must be removed');
  assert.equal(fs.existsSync(lockPath), true, 'the writer lease file outside the deleted root must survive');
  assert.equal(fs.existsSync(deleteLockPath(aliasRoot, job.id)), true, 'the delete lock outside the deleted root must survive');
  const surviving = await F.db.runClaim.findUnique({ where: { resource: unrelatedResource } });
  assert.ok(surviving, 'unrelated claim was removed by deletion');
  assert.equal(surviving.run_id, `other-run-${suffix}`);
});

test('actual store.deleteJob afterBegin holds the TS writer lock so a one-shot Python acquire fails busy, then later acquire is clean', { timeout: 180000 }, async t => {

  const suffix = `${process.pid}-${Date.now()}`;
  const jobName = `afterbegin-${suffix}`;
  const dir = path.join(F.trainingRoot, jobName);
  const sentinel = Buffer.from([9, 8, 7, 0, 6, 5, 4, 3]);
  const checkpoint = folderBytes(dir, sentinel);
  const configPath = path.join(dir, 'config.json');
  const configBytes = fs.readFileSync(configPath);
  const outsideResource = path.join(F.trainingRoot, `outside-resource-${suffix}.bin`);
  const outsideBytes = Buffer.from([11, 12, 13, 0, 14, 15, 16]);
  fs.writeFileSync(outsideResource, outsideBytes);
  const job = await makeJob(F.db, { name: jobName, status: 'stopped' });
  const lockPath = writerLeasePath(dir);
  let oneShot = null;
  let inodeAtHook = null;
  let lockExistedAtHook = false;

  const out = await store.deleteJob(F.db, job.id, F.trainingRoot, {
    afterBegin: async () => {
      inodeAtHook = fs.statSync(dir).ino;
      lockExistedAtHook = fs.existsSync(lockPath);
      const py = startPython(PY_CHILD, ['once', PY_MODULE, dir, `owner:${jobName}`, `holder:${process.pid}`]);
      try {
        py.child.stdin.end();
        oneShot = await py.next(30000);
        const exit = await py.waitExit(15000);
        assert.equal(exit.code, 0, `one-shot Python failed: ${py.stderr()}\nstdout=${py.stdout()}`);
      } finally {
        await stopGracefully(py);
      }
      assert.equal(oneShot.status, 'busy', `expected SQLite busy from Python: ${JSON.stringify(oneShot)}`);
      assert.equal(oneShot.type, 'SaveRootBusy');
      assert.equal(fs.statSync(dir).ino, inodeAtHook, 'a refused acquire remade the save root');
      assert.deepEqual(fs.readFileSync(configPath), configBytes, 'a refused acquire wrote/remade config');
      assert.deepEqual(fs.readFileSync(checkpoint), sentinel);
    },
  });

  t.after(() => {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // fixture cleanup; deletion usually already removed it
    }
    try {
      fs.rmSync(outsideResource, { force: true });
    } catch {
      // fixture cleanup
    }
  });

  assert.equal(out.status, 200, JSON.stringify(out.body));
  assert.equal(await F.db.job.findUnique({ where: { id: job.id } }), null);
  assert.equal(fs.existsSync(dir), false);
  assert.equal(lockExistedAtHook, true, 'TS writer lock must exist while afterBegin runs');
  assert.deepEqual(fs.readFileSync(outsideResource), outsideBytes, 'resource file outside the deleted root changed');
  assert.equal(fs.existsSync(lockPath), true, 'the writer lease file outside the deleted root must survive');
  assert.equal(fs.existsSync(deleteLockPath(F.trainingRoot, job.id)), true, 'the delete lock outside the deleted root must survive');

  const after = startPython(PY_CHILD, ['once', PY_MODULE, dir, `owner:${jobName}`, `holder:${process.pid}`]);
  try {
    after.child.stdin.end();
    const acquired = await after.next(30000);
    const exit = await after.waitExit(15000);
    assert.equal(exit.code, 0, `post-delete Python failed: ${after.stderr()}\nstdout=${after.stdout()}`);
    assert.equal(acquired.status, 'acquired', `stale writer lease remained after deletion: ${JSON.stringify(acquired)}`);
    assert.equal(acquired.lease_path, writerLeasePath(dir), 'post-delete lease path differs from TS');
  } finally {
    await stopGracefully(after);
  }
  assert.deepEqual(fs.readFileSync(outsideResource), outsideBytes, 'post-delete acquire changed the outside resource file');
});

test('beginDelete expectedFolder guard refuses a changed folder before reservation and leaves both roots intact', { timeout: 60000 }, async t => {
  const suffix = `${process.pid}-${Date.now()}`;
  const oldName = `guard-old-${suffix}`;
  const currentName = `guard-current-${suffix}`;
  const oldDir = path.join(F.trainingRoot, oldName);
  const currentDir = path.join(F.trainingRoot, currentName);
  const oldBytes = Buffer.from([1, 1, 2, 3, 5, 8, 13]);
  const currentBytes = Buffer.from([21, 34, 55, 89]);
  fs.mkdirSync(oldDir, { recursive: true });
  fs.writeFileSync(path.join(oldDir, 'sentinel.ckpt'), oldBytes);
  fs.mkdirSync(currentDir, { recursive: true });
  fs.writeFileSync(path.join(currentDir, 'sentinel.ckpt'), currentBytes);
  t.after(() => {
    fs.rmSync(oldDir, { recursive: true, force: true });
    fs.rmSync(currentDir, { recursive: true, force: true });
  });
  const job = await makeJob(F.db, { name: currentName, status: 'stopped' });
  const rowBefore = await F.db.job.findUnique({ where: { id: job.id } });
  const expectedFolder = canonicalFolder(F.trainingRoot, oldName);

  const out = await store.beginDelete(F.db, job.id, F.trainingRoot, { expectedFolder });
  assert.equal(out.status, 409, JSON.stringify(out.body));
  assert.match(out.body.error, /folder changed before deletion was reserved/i);

  const rowAfter = await F.db.job.findUnique({ where: { id: job.id } });
  assert.equal(rowAfter.status, rowBefore.status);
  assert.equal(rowAfter.info, rowBefore.info);
  assert.equal(await F.db.runClaim.count({ where: { job_id: job.id } }), 0);
  assert.deepEqual(fs.readFileSync(path.join(oldDir, 'sentinel.ckpt')), oldBytes);
  assert.deepEqual(fs.readFileSync(path.join(currentDir, 'sentinel.ckpt')), currentBytes);
  assert.equal(fs.existsSync(oldDir), true);
  assert.equal(fs.existsSync(currentDir), true);

  fs.rmSync(oldDir, { recursive: true, force: true });
  fs.rmSync(currentDir, { recursive: true, force: true });
});
