const { test } = require('node:test');
const assert = require('node:assert/strict');
const ts = require('./harness/tsrequire.cjs');
const { react, renderHook, flush, scriptedApi } = require('./harness/reactHost.cjs');

const elem = (type, props) => ({ type, props });
ts.mock('react', react);
ts.mock('react/jsx-runtime', { jsx: elem, jsxs: elem, Fragment: 'Fragment' });
ts.mock('next/navigation', { useRouter: () => ({ push() {} }), useSearchParams: () => new URLSearchParams('id=old') });
const api = scriptedApi();
ts.mock('@/utils/api', api);
ts.mock('@/hooks/useGPUInfo', () => ({ gpuList: [], isGPUInfoLoaded: false }));
ts.mock('@/hooks/useDatasetList', () => ({ datasets: [], status: 'error' }));
ts.mock('@/hooks/useSettings', () => ({ settings: {}, isSettingsLoaded: false }));
ts.mock('@/components/layout', { MainContent: 'main', TopBar: 'header' });
ts.mock('@/components/formInputs', { CreatableSelectInput: 'gpu-select', SelectInput: 'select' });
ts.mock('@headlessui/react', { Button: 'button' });
ts.mock('react-icons/fa', { FaChevronLeft: 'back' });
ts.mock('./SimpleJob', 'simple-editor');
ts.mock('./options', { jobTypeOptions: [] });
ts.mock('@/components/AdvancedConfigEditor', 'advanced-editor');
ts.mock('@/components/ErrorBoundary', 'error-boundary');
const Form = require(ts.uiPath('src/app/jobs/new/page.tsx')).default;
const { type: Editor } = Form();
const find = (node, match) => {
  if (Array.isArray(node)) return node.map(n => find(n, match)).find(Boolean) ?? null;
  if (!node || typeof node !== 'object') return null;
  return match(node) ? node : find(node.props?.children, match);
};

test('edit and clone load the persisted config before exposing Save, including after a failed request', async () => {
  for (const [runId, cloneId, name] of [['old', null, 'original'], [null, 'old', 'original_copy']]) {
    const h = renderHook(props => Editor(props), { runId, cloneId });
    assert.equal(find(h.result.current, n => n.type === 'simple-editor'), null);
    api.calls.at(-1).reject(new Error('unavailable'));
    await flush();
    const retry = find(h.result.current, n => n.type === 'button');
    assert.equal(retry.props.children, 'Retry');
    assert.equal(find(h.result.current, n => n.type === 'simple-editor'), null);
    retry.props.onClick();
    await flush();
    api.calls.at(-1).resolve({ gpu_ids: '1,0', job_config: JSON.stringify({ config: {
      name: 'original', process: [{ type: 'custom_trainer', training_folder: '/previous', train: { steps: 500 } }],
    } }) });
    await flush();
    const editor = find(h.result.current, n => n.type === 'simple-editor');
    assert.equal(editor.props.gpuIDs, '1,0');
    assert.equal(editor.props.jobConfig.config.name, name);
    await editor.props.handleSubmit({ preventDefault() {} });
    const request = api.calls.at(-1);
    assert.equal(request.method, 'post');
    assert.equal(request.config.id, runId);
    assert.equal(request.config.job_config.config.process[0].training_folder, '/previous');
    assert.equal(request.config.job_config.config.process[0].train.steps, 500);
    h.unmount();
  }
});
