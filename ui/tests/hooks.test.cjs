// Browser request ownership (docs/STATE_MACHINES.md M18, C07) for the ACTUAL ui/src/hooks modules, run under the
// SIMULATED React host in harness/reactHost.cjs (not React, not a browser) with a scripted apiClient whose responses
// the tests resolve in a chosen order.
//   node --test --test-concurrency=1 ui/tests/hooks.test.cjs
// AITK_HOOKS_DIR=<dir with the original hook files> runs the same cases against another copy of the hooks (used to
// record the original f6c137b4 behaviour; see docs).
const { test, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const ts = require('./harness/tsrequire.cjs');
const { react, renderHook, flush, scriptedApi } = require('./harness/reactHost.cjs');

let api = scriptedApi();
ts.mock('react', react);
ts.mock('@/utils/api', {
  get apiClient() {
    return api.apiClient;
  },
});
const HOOKS = process.env.AITK_HOOKS_DIR || ts.uiPath('src', 'hooks');
const hook = name => require(path.join(HOOKS, `${name}.tsx`)).default;
const useJobLog = hook('useJobLog');
const useJobLossLog = hook('useJobLossLog');
const useJob = hook('useJob');
const useJobByRef = hook('useJobByRef');
const useJobsList = hook('useJobsList');
const useSampleImages = hook('useSampleImages');
const useFilesList = hook('useFilesList');
const usePollLoop = require(ts.uiPath('src', 'hooks', 'usePollLoop.tsx')).default;

// every mounted host is unmounted after its test (stops poll-loop timers)
const hosts = [];
const mount = (fn, props) => {
  const h = renderHook(fn, props);
  hosts.push(h);
  return h;
};
beforeEach(() => {
  api = scriptedApi();
});
afterEach(() => {
  for (const h of hosts.splice(0)) h.unmount();
});
const gets = prefix => api.calls.filter(c => c.method === 'get' && c.url.startsWith(prefix));

test('useJobLog: an old job\'s response that resolves after the switch writes nothing into the new job\'s terminal or cursor', async () => {
  const h = mount(({ id }) => useJobLog(id, null), { id: 'A' });
  await flush();
  const [a] = gets('/api/jobs/A/log');
  h.rerender({ id: 'B' });
  await flush();
  const [b] = gets('/api/jobs/B/log');
  assert.ok(a && b, 'one read per job');
  a.resolve({ log: 'job A text\n', offset: 1000, reset: true, generation: 'ga' });
  await flush();
  assert.equal(h.result.current.log, '', 'A is not shown under B');
  b.resolve({ log: 'job B text\n', offset: 50, reset: true, generation: 'gb' });
  await flush();
  assert.match(h.result.current.log, /job B text/);
  assert.doesNotMatch(h.result.current.log, /job A/);
  // the next read continues B's cursor, not A's
  h.result.current.refresh();
  await flush();
  const next = gets('/api/jobs/B/log')[1];
  assert.equal(next.config.params.offset, 50);
  assert.equal(next.config.params.generation, 'gb');
});

test('useJobLog: the old job\'s finally does not release the new job\'s in-flight slot (no duplicate cursor reads)', async () => {
  const h = mount(({ id }) => useJobLog(id, null), { id: 'A' });
  await flush();
  const [a] = gets('/api/jobs/A/log');
  h.rerender({ id: 'B' });
  await flush();
  a.reject(new Error('A failed late'));
  await flush();
  h.result.current.refresh(); // B's first read is still pending: must not start a second one from the same offset
  await flush();
  assert.equal(gets('/api/jobs/B/log').length, 1);
  assert.notEqual(h.result.current.status, 'error', 'A\'s failure is not B\'s status');
});

test('useJobLossLog: a GET issued before deleteRange (resolving during or after the DELETE) never puts deleted points back', async () => {
  const h = mount(({ id }) => useJobLossLog(id, 1_000_000), { id: 'J' });
  await flush();
  const pts = steps => ({ keys: ['loss'], run_id: 'r1', series: { loss: { steps, values: steps.map(s => s / 10) } } });
  gets('/api/jobs/J/loss')[0].resolve(pts([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
  await flush();
  assert.deepEqual(h.result.current.series.loss.steps, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  h.result.current.refreshLoss(); // incremental read in flight, carries pre-delete data
  await flush();
  const stale = gets('/api/jobs/J/loss')[1];
  assert.equal(stale.config.params.since_step, 10);
  const del = h.result.current.deleteRange(5, 10);
  await flush();
  const d = api.calls.find(c => c.method === 'delete');
  stale.resolve(pts([10, 11]));
  await flush();
  d.resolve({ ok: true });
  await flush();
  assert.deepEqual(h.result.current.series.loss.steps, [1, 2, 3, 4], 'local removal applied; stale 10 discarded');
  const reload = gets('/api/jobs/J/loss')[2];
  assert.ok(reload, 'full reload after the DELETE');
  assert.equal(reload.config.params.since_step, undefined);
  reload.resolve(pts([1, 2, 3, 4, 11]));
  await del;
  await flush();
  assert.deepEqual(h.result.current.series.loss.steps, [1, 2, 3, 4, 11]);
});

test('useJobLossLog: a DELETE that resolves after the hook moved to another job changes nothing there', async () => {
  const h = mount(({ id }) => useJobLossLog(id, null), { id: 'A' });
  await flush();
  gets('/api/jobs/A/loss')[0].resolve({ keys: ['loss'], run_id: 'ra', series: { loss: { steps: [1, 2, 3], values: [1, 2, 3] } } });
  await flush();
  const del = h.result.current.deleteRange(1, 2);
  await flush();
  h.rerender({ id: 'B' });
  await flush();
  gets('/api/jobs/B/loss')[0].resolve({ keys: ['loss'], run_id: 'rb', series: { loss: { steps: [1, 2, 3], values: [7, 8, 9] } } });
  await flush();
  api.calls.find(c => c.method === 'delete').resolve({ ok: true });
  await del;
  await flush();
  assert.deepEqual(h.result.current.series.loss.steps, [1, 2, 3]);
  assert.equal(gets('/api/jobs/A/loss').length, 1, 'no reload for the old job');
});

test('useJobLossLog: a new run writing the same log makes the next read a full reload (history kept)', async () => {
  const h = mount(({ id }) => useJobLossLog(id, 1_000_000), { id: 'J' });
  await flush();
  gets('/api/jobs/J/loss')[0].resolve({ keys: ['loss'], run_id: 'r1', series: { loss: { steps: [1, 2, 3], values: [1, 1, 1] } } });
  await flush();
  h.result.current.refreshLoss();
  await flush();
  gets('/api/jobs/J/loss')[1].resolve({ keys: ['loss'], run_id: 'r2', series: { loss: { steps: [3], values: [5] } } });
  await new Promise(r => setTimeout(r, 5));
  await flush();
  const full = gets('/api/jobs/J/loss')[2];
  assert.ok(full);
  assert.equal(full.config.params.since_step, undefined);
  full.resolve({ keys: ['loss'], run_id: 'r2', series: { loss: { steps: [1, 2], values: [9, 9] } } });
  await flush();
  assert.deepEqual(h.result.current.series.loss.values, [9, 9]);
});

test('useJob: two manual refreshes resolving out of order cannot regress the status; an older poll is dropped', async () => {
  const h = mount(({ id }) => useJob(id, null), { id: 'J' });
  await flush();
  const poll = gets('/api/jobs?id=J')[0];
  h.result.current.refreshJob();
  h.result.current.refreshJob();
  await flush();
  const [, r1, r2] = gets('/api/jobs?id=J');
  r2.resolve({ id: 'J', status: 'stopping' });
  await flush();
  r1.resolve({ id: 'J', status: 'running' });
  poll.resolve({ id: 'J', status: 'queued' });
  await flush();
  assert.equal(h.result.current.job.status, 'stopping');
});

test('useJob: same-job mutation vs an older GET: a refresh after the action supersedes a poll issued before it, whatever the order', async () => {
  const h = mount(({ id }) => useJob(id, null), { id: 'J' });
  await flush();
  const before = gets('/api/jobs?id=J')[0]; // issued before the Stop click
  h.result.current.refreshJob(); // after the Stop request returned
  await flush();
  const after = gets('/api/jobs?id=J')[1];
  before.resolve({ id: 'J', status: 'running' }); // older data arrives first...
  await flush();
  assert.equal(h.result.current.job, null, 'a GET issued before the action is not applied after it');
  after.resolve({ id: 'J', status: 'stopping' });
  await flush();
  assert.equal(h.result.current.job.status, 'stopping');
});

test('useJob / useJobByRef / useSampleImages / useFilesList: a previous identity\'s response is dropped', async () => {
  const cases = [
    [p => useJob(p.id, null), '/api/jobs?id=', d => d.job?.id, x => ({ id: x })],
    [p => useJobByRef(p.id, null), '/api/jobs?job_ref=', d => d.job?.id, x => ({ id: x })],
    [p => useSampleImages(p.id, null), '/api/jobs/', d => d.sampleImages[0], x => ({ samples: [x] })],
    [p => useFilesList(p.id, null), '/api/jobs/', d => d.files[0]?.path, x => ({ files: [{ path: x, size: 1 }] })],
  ];
  for (const [fn, prefix, read, body] of cases) {
    api = scriptedApi();
    const h = mount(fn, { id: 'A' });
    await flush();
    const a = gets(`${prefix}A`)[0];
    h.rerender({ id: 'B' });
    await flush();
    const b = gets(`${prefix}B`)[0];
    b.resolve(body('B'));
    await flush();
    a.resolve(body('A'));
    await flush();
    assert.equal(read(h.result.current), 'B', prefix);
  }
});

test('useJobsList: manual refreshes resolving out of order keep the newest list', async () => {
  const h = mount(() => useJobsList({ reloadInterval: null }), {});
  await flush();
  h.result.current.refreshJobs();
  await flush();
  const [first, second] = gets('/api/jobs');
  second.resolve({ jobs: [{ id: 'new', status: 'stopping' }] });
  await flush();
  first.resolve({ jobs: [{ id: 'new', status: 'running' }] });
  await flush();
  assert.equal(h.result.current.jobs[0].status, 'stopping');
});

test('usePollLoop: a loop instance stopped by a deps change reports isCurrent() === false to its in-flight call', async () => {
  const seen = [];
  const h = mount(({ k }) => usePollLoop(isCurrent => seen.push({ k, isCurrent }), null, [k]), { k: 1 });
  h.rerender({ k: 2 });
  assert.equal(seen.length, 2);
  assert.equal(seen[0].isCurrent(), false);
  assert.equal(seen[1].isCurrent(), true);
  h.unmount();
  assert.equal(seen[1].isCurrent(), false);
});

test('useJobLossLog: a run-change retry queued for job A neither requests A nor touches B after the switch (Codex loss-retry-identity-probe)', async () => {
  const timers = [];
  const realSet = global.setTimeout;
  const realClear = global.clearTimeout;
  global.setTimeout = (fn, ms) => {
    const t = { fn, ms, cancelled: false };
    timers.push(t);
    return t;
  };
  global.clearTimeout = t => {
    if (t) t.cancelled = true;
  };
  const pts = (run, value) => ({ keys: ['loss'], run_id: run, series: { loss: { steps: [1], values: [value] } } });
  try {
    const h = mount(({ id }) => useJobLossLog(id, 1_000_000), { id: 'A' });
    await flush();
    api.calls[0].resolve(pts('A-old', 11));
    await flush();
    h.result.current.refreshLoss();
    await flush();
    api.calls[1].resolve(pts('A-new', 22)); // run changed: a full reload of A is scheduled
    await flush();
    const retry = timers.find(t => t.ms === 0 && !t.cancelled);
    assert.ok(retry, 'generation retry scheduled');
    h.rerender({ id: 'B' });
    await flush();
    assert.equal(retry.cancelled, true, 'the switch cancels the queued retry');
    api.calls[2].resolve(pts('B', 99));
    await flush();
    retry.fn(); // even if it fires anyway
    await flush();
    assert.deepEqual(api.calls.slice(3).map(c => c.url), [], 'no late request for A');
    assert.deepEqual(h.result.current.series.loss.values, [99]);
  } finally {
    global.setTimeout = realSet;
    global.clearTimeout = realClear;
  }
});

test('loss graph replaces old points when the current launch prunes metrics after its first poll', async () => {
  const h = mount(({ id }) => useJobLossLog(id, 1_000_000), { id: 'J' });
  await flush();
  const response = (steps, wall_time, reset) => ({
    keys: ['loss'], run_id: 'resumed', reset, cursor: { step: steps.at(-1), wall_time },
    series: { loss: { steps, values: steps } },
  });
  gets('/api/jobs/J/loss')[0].resolve(response([3, 8, 10], 120, false));
  await flush();
  h.result.current.refreshLoss();
  await flush();
  const poll = gets('/api/jobs/J/loss')[1];
  assert.deepEqual(poll.config.params, { since_step: 10, since_time: 120 });
  poll.resolve(response([3, 5], 125, true));
  await flush();
  assert.deepEqual(h.result.current.series.loss.steps, [3, 5]);
  h.result.current.refreshLoss();
  await flush();
  assert.deepEqual(gets('/api/jobs/J/loss')[2].config.params, { since_step: 5, since_time: 125 });
});
