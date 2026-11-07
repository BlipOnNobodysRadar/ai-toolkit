// Job action gating (docs/STATE_MACHINES.md M35 deletion recovery): the actual ui/src/utils/jobs.ts helper, and the
// actual JobActionBar component evaluated under the SIMULATED React host (harness/reactHost.cjs; its UI-library
// imports are stubbed, JSX becomes plain objects, nothing is rendered in a browser). Codex delete-ui-retry-probe:
// before this repair every action was false for a persisted 'deleting' row left by a crashed deleter.
//   node --test --test-concurrency=1 ui/tests/jobActions.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const ts = require('./harness/tsrequire.cjs');
const { react, renderHook, flush } = require('./harness/reactHost.cjs');

const requests = [];
const confirms = [];
const stub = name => Object.assign(() => null, { displayName: name });
ts.mock('react', react);
ts.mock('react/jsx-runtime', {
  jsx: (type, props) => ({ type, props }),
  jsxs: (type, props) => ({ type, props }),
  Fragment: 'Fragment',
});
ts.mock('@/utils/api', { apiClient: { get: url => (requests.push(url), Promise.resolve({ data: { ok: true } })) } });
ts.mock('@/components/ConfirmModal', { openConfirm: c => confirms.push(c) });
ts.mock('@/components/CaptionDatasetModal', { openCaptionDatasetModal: () => {} });
ts.mock('@/utils/queue', { startQueue: async () => {} });
ts.mock('next/link', { __esModule: true, default: stub('Link') });
ts.mock('next/navigation', { redirect: () => {} });
ts.mock('lucide-react', new Proxy({}, { get: (_, k) => stub(String(k)) }));
ts.mock('react-icons/lu', new Proxy({}, { get: (_, k) => stub(String(k)) }));
ts.mock('@headlessui/react', new Proxy({}, { get: (_, k) => stub(String(k)) }));

const { getAvaliableJobActions } = require(ts.uiPath('src/utils/jobs.ts'));
const JobActionBar = require(ts.uiPath('src/components/JobActionBar.tsx')).default;

const row = over => ({
  id: 'J1',
  name: 'job-one',
  job_type: 'train',
  gpu_ids: '0',
  status: 'stopped',
  run_id: null,
  pid: null,
  launch_phase: null,
  stop: false,
  step: 0,
  job_config: JSON.stringify({ config: { process: [{ train: { steps: 10 } }] } }),
  ...over,
});

const find = (node, pred) => {
  if (!node || typeof node !== 'object') return null;
  if (Array.isArray(node)) {
    for (const n of node) {
      const f = find(n, pred);
      if (f) return f;
    }
    return null;
  }
  if (node.props && pred(node.props)) return node;
  return find(node.props?.children, pred);
};

test('helper: an unfinished deletion (deleting/delete_failed, no run ownership) offers Retry Delete; owned or active rows do not', () => {
  for (const status of ['deleting', 'delete_failed']) {
    const a = getAvaliableJobActions(row({ status }));
    assert.equal(a.canDelete, true, status);
    assert.equal(a.isRetryDelete, true, status);
    assert.equal(a.canStart, false);
    assert.equal(a.canEdit, false);
  }
  const plain = getAvaliableJobActions(row({ status: 'stopped' }));
  assert.equal(plain.canDelete, true);
  assert.equal(plain.isRetryDelete, false);
  for (const r of [
    row({ status: 'running', run_id: 'r', launch_phase: 'attached' }),
    row({ status: 'completed', run_id: 'r', launch_phase: 'attached' }), // reported done, process not yet verified gone
    row({ status: 'deleting', run_id: 'r', launch_phase: 'attached' }), // never guessed: ownership wins
    row({ status: 'completed', pid: 123 }), // pre-update row with a recorded pid
  ]) {
    assert.equal(getAvaliableJobActions(r).canDelete, false, JSON.stringify([r.status, r.launch_phase, r.pid]));
  }
});

test('JobActionBar: the Retry Delete button is enabled and labelled for a persisted deleting row; confirming sends the Delete request', async () => {
  const h = renderHook(props => JobActionBar(props), { job: row({ status: 'deleting' }), onRefresh: () => {} });
  const button = find(h.result.current, p => p['aria-label'] === 'Retry Delete');
  assert.ok(button, 'Retry Delete button present');
  assert.equal(button.props.disabled, false);
  assert.match(button.props.title, /Retry Delete/);
  button.props.onClick();
  const c = confirms.pop();
  assert.equal(c.title, 'Retry Delete');
  assert.equal(c.confirmText, 'Retry Delete');
  assert.match(c.message, /did not finish/);
  await c.onConfirm();
  await flush();
  assert.deepEqual(requests.slice(-1), ['/api/jobs/J1/delete']);
  h.unmount();
});

test('JobActionBar: an ordinary stopped row shows a plain Delete; a running row has it disabled', () => {
  const plain = renderHook(props => JobActionBar(props), { job: row({ status: 'stopped' }) });
  const del = find(plain.result.current, p => p['aria-label'] === 'Delete');
  assert.equal(del.props.disabled, false);
  plain.unmount();
  const running = renderHook(props => JobActionBar(props), { job: row({ status: 'running', run_id: 'r', launch_phase: 'attached' }) });
  const off = find(running.result.current, p => p['aria-label'] === 'Delete');
  assert.equal(off.props.disabled, true);
  running.unmount();
});
