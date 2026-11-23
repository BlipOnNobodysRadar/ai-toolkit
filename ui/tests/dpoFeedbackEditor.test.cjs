// Focused editor-state tests for the interleaved MOSS ranked-sample feedback mode.
// Run with: node --test --test-concurrency=1 ui/tests/dpoFeedbackEditor.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const ts = require('./harness/tsrequire.cjs');
const { react, renderHook } = require('./harness/reactHost.cjs');

// Keep module loading hermetic and free of browser/Next-only side effects.
const stub = name => Object.assign(() => null, { displayName: name });
ts.mock('react', react);
ts.mock('react/jsx-runtime', {
  jsx: (type, props) => ({ type, props }),
  jsxs: (type, props) => ({ type, props }),
  Fragment: 'Fragment',
});
ts.mock('react-icons/io5', new Proxy({}, { get: () => stub('Icon') }));
ts.mock('lucide-react', new Proxy({}, { get: () => stub('Icon') }));
ts.mock('@/components/formInputs', new Proxy({}, { get: (_, key) => stub(String(key)) }));
ts.mock('@/components/Card', { __esModule: true, default: stub('Card') });
ts.mock('@/components/DocModal', { openDoc: () => {} });
ts.mock('@/components/UpsamplePromptsModal', {
  openUpsamplePromptsModal: () => {},
  toAspectRatio: () => '1:1',
});
ts.mock('@/components/PromptBoxEditorModal', { openPromptBoxEditor: () => {} });
ts.mock('@/components/AddSingleImageModal', {
  __esModule: true,
  default: stub('AddSingleImageModal'),
  openAddImageModal: () => {},
});
ts.mock('@/components/SampleControlImage', { __esModule: true, default: stub('SampleControlImage') });
const MOSS_UI_ARCH = {
  name: 'moss_image',
  group: 'image',
  defaults: {},
  disableSections: [],
  additionalSections: [],
};
ts.mock('@/extensions/modelArchs', {
  useModelArchs: () => ({ archs: [MOSS_UI_ARCH], groupedModelOptions: [] }),
});

const { setNestedValue } = require(ts.uiPath('src/utils/hooks.tsx'));
const SimpleJob = require(ts.uiPath('src/app/jobs/new/SimpleJob.tsx')).default;
const { defaultJobConfig } = require(ts.uiPath('src/app/jobs/new/jobConfig.ts'));
const {
  DPO_BETA_DEFAULT,
  DPO_FEEDBACK_EVERY_DEFAULT,
  DPO_FEEDBACK_PROMPT_FALLBACK,
  DPO_FEEDBACK_START_AFTER_DEFAULT,
  applyDpoFeedbackMode,
  applyDpoMode,
  clearDpoConfig,
  clearDpoFeedbackConfig,
  handleModelArchChange,
  isDpoFeedbackEnabled,
  isValidDpoFeedbackEvery,
  isValidDpoFeedbackStartAfter,
} = require(ts.uiPath('src/app/jobs/new/utils.ts'));

const clone = value => JSON.parse(JSON.stringify(value));
const freshJob = () => clone(defaultJobConfig);

// This is the same path updater the real editor uses through useNestedState.
const stateHarness = initial => {
  let state = initial;
  const calls = [];
  const setJobConfig = (value, key) => {
    calls.push([key, value]);
    state = key ? setNestedValue(state, value, key) : value;
  };
  return { get state() { return state; }, calls, setJobConfig };
};

const mossArchs = [
  { name: 'moss_image', defaults: {}, additionalSections: [] },
  { name: 'flex1', defaults: {}, additionalSections: [] },
];

test('ranked sample feedback is absent/null-off and enabling is an ordinary-loss mode switch', () => {
  assert.equal(isDpoFeedbackEnabled(undefined), false);
  assert.equal(isDpoFeedbackEnabled(null), false);
  assert.equal(isDpoFeedbackEnabled({ every: 10, start_after: 0 }), true);

  const job = freshJob();
  const process = job.config.process[0];
  process.train.loss_type = 'mse';
  process.train.disable_sampling = true;
  process.sample.samples = [];
  delete process.train.dpo_beta;
  delete process.train.dpo_feedback;
  process.datasets[0].folder_path = 'C:/data/my-target';
  process.datasets[0].rejected_path = 'C:/somewhere/offline-rejected';

  const h = stateHarness(job);
  assert.equal(applyDpoFeedbackMode(h.state, h.setJobConfig), true);
  const out = h.state.config.process[0];

  assert.deepEqual(out.train.dpo_feedback, {
    every: DPO_FEEDBACK_EVERY_DEFAULT,
    start_after: DPO_FEEDBACK_START_AFTER_DEFAULT,
  });
  assert.equal(out.train.dpo_beta, DPO_BETA_DEFAULT);
  assert.equal(out.train.loss_type, 'mse', 'enabling feedback must leave the ordinary loss selectable');
  assert.equal(out.train.disable_sampling, false, 'feedback needs real samples to rank');
  assert.deepEqual(out.sample.samples, [{ prompt: DPO_FEEDBACK_PROMPT_FALLBACK }]);
  assert.equal(out.datasets[0].folder_path, 'C:/data/my-target', 'ordinary target folder is untouched');
  assert.equal(out.datasets[0].rejected_path, 'C:/somewhere/offline-rejected', 'offline config is not invented or removed');
  assert.equal(out.network.type, 'lora');
  assert.equal(out.network.transformer_only, true);
  assert.equal(out.train.train_unet, true);
  assert.equal(out.train.train_text_encoder, false);
});

test('re-enabling preserves valid feedback cadence, beta and normal sample settings', () => {
  const job = freshJob();
  const process = job.config.process[0];
  process.train.loss_type = 'mae';
  process.train.dpo_beta = 42;
  process.train.dpo_feedback = { every: 20, start_after: 5 };
  process.train.disable_sampling = true;
  process.sample.sample_every = 77;
  process.sample.width = 768;
  process.sample.height = 512;
  process.sample.samples = [{ prompt: 'my own prompt', seed: 7 }];

  const h = stateHarness(job);
  assert.equal(applyDpoFeedbackMode(h.state, h.setJobConfig), true);
  const out = h.state.config.process[0];

  assert.deepEqual(out.train.dpo_feedback, { every: 20, start_after: 5 });
  assert.equal(out.train.dpo_beta, 42);
  assert.equal(out.train.loss_type, 'mae');
  assert.equal(out.sample.sample_every, 77);
  assert.equal(out.sample.width, 768);
  assert.equal(out.sample.height, 512);
  assert.deepEqual(out.sample.samples, [{ prompt: 'my own prompt', seed: 7 }]);
});

test('turning feedback off removes only the online fields, and preserves beta for offline DPO', () => {
  const standard = freshJob();
  standard.config.process[0].train.loss_type = 'mse';
  standard.config.process[0].train.dpo_feedback = { every: 10, start_after: 0 };
  standard.config.process[0].train.dpo_beta = 100;
  let h = stateHarness(standard);
  clearDpoFeedbackConfig(h.state, h.setJobConfig);
  assert.equal('dpo_feedback' in h.state.config.process[0].train, false);
  assert.equal('dpo_beta' in h.state.config.process[0].train, false);

  const offline = freshJob();
  offline.config.process[0].train.loss_type = 'dpo';
  offline.config.process[0].train.dpo_feedback = { every: 10, start_after: 0 };
  offline.config.process[0].train.dpo_beta = 100;
  h = stateHarness(offline);
  clearDpoFeedbackConfig(h.state, h.setJobConfig);
  assert.equal('dpo_feedback' in h.state.config.process[0].train, false);
  assert.equal(h.state.config.process[0].train.dpo_beta, 100);
});

test('leaving offline DPO clears rejected paths but keeps beta while online feedback is active', () => {
  const job = freshJob();
  const process = job.config.process[0];
  process.train.dpo_feedback = { every: 10, start_after: 0 };
  process.train.dpo_beta = 100;
  process.train.dpo_reference_lora_path = 'C:/models/original.safetensors';
  process.train.dpo_reference_lora_strength = 0.8;
  process.datasets[0].rejected_path = 'C:/rejected';

  const h = stateHarness(job);
  clearDpoConfig(h.state, h.setJobConfig);
  assert.equal(h.state.config.process[0].train.dpo_beta, 100);
  assert.equal('rejected_path' in h.state.config.process[0].datasets[0], false);
  assert.equal('dpo_reference_lora_path' in h.state.config.process[0].train, false);
  assert.equal('dpo_reference_lora_strength' in h.state.config.process[0].train, false);

  h.setJobConfig(null, 'config.process[0].train.dpo_feedback');
  clearDpoConfig(h.state, h.setJobConfig);
  assert.equal('dpo_beta' in h.state.config.process[0].train, false);
});

test('changing to an ineligible model clears stale online and offline preference options', () => {
  const job = freshJob();
  const process = job.config.process[0];
  process.model.arch = 'moss_image';
  process.train.loss_type = 'mse';
  process.train.dpo_feedback = { every: 15, start_after: 2 };
  process.train.dpo_beta = 100;
  process.datasets[0].rejected_path = 'C:/rejected';

  const h = stateHarness(job);
  handleModelArchChange(mossArchs, 'moss_image', 'flex1', h.state, h.setJobConfig);
  const out = h.state.config.process[0];

  assert.equal(out.model.arch, 'flex1');
  assert.equal('dpo_feedback' in out.train, false);
  assert.equal('dpo_beta' in out.train, false);
  assert.equal('rejected_path' in out.datasets[0], false);
});

test('offline DPO remains selectable, clears online feedback, and rejects decimal cadence values', () => {
  assert.equal(isValidDpoFeedbackEvery(10), true);
  assert.equal(isValidDpoFeedbackEvery(10.5), false);
  assert.equal(isValidDpoFeedbackStartAfter(0), true);
  assert.equal(isValidDpoFeedbackStartAfter(-1), false);

  const online = freshJob();
  online.config.process[0].model.arch = 'moss_image';
  online.config.process[0].train.loss_type = 'dpo';
  delete online.config.process[0].train.dpo_feedback;
  const blocked = stateHarness(online);
  assert.equal(applyDpoFeedbackMode(blocked.state, blocked.setJobConfig), false);
  assert.equal('dpo_feedback' in blocked.state.config.process[0].train, false);

  const job = freshJob();
  job.config.process[0].model.arch = 'moss_image';
  job.config.process[0].train.dpo_feedback = { every: 10, start_after: 0 };
  const h = stateHarness(job);
  applyDpoMode(h.state, h.setJobConfig);
  assert.equal(h.state.config.process[0].train.loss_type, 'dpo');
  assert.equal('dpo_feedback' in h.state.config.process[0].train, false);
  assert.equal(h.state.config.process[0].train.dpo_beta, DPO_BETA_DEFAULT);
});
const findNode = (node, pred) => {
  if (!node || typeof node !== 'object') return null;
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findNode(child, pred);
      if (found) return found;
    }
    return null;
  }
  if (pred(node)) return node;
  return findNode(node.props?.children, pred);
};

const findText = (node, text) => {
  if (typeof node === 'string') return node.includes(text) ? node : null;
  if (!node || typeof node !== 'object') return null;
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findText(child, text);
      if (found) return found;
    }
    return null;
  }
  return findText(node.props?.children, text);
};

const uiProps = (jobConfig, setJobConfig = () => {}) => ({
  jobConfig,
  setJobConfig,
  status: 'idle',
  handleSubmit: () => {},
  runId: null,
  gpuIDs: null,
  setGpuIDs: () => {},
  gpuList: [],
  datasetOptions: [],
});

test('SimpleJob exposes the ranked-feedback switch and delayed-voting labels', () => {
  const job = freshJob();
  job.config.process[0].model.arch = 'moss_image';
  job.config.process[0].train.loss_type = 'mse';
  const r = renderHook(props => SimpleJob(props), uiProps(job));
  const tree = r.result.current;
  const toggle = findNode(tree, node => node.props?.label === 'Periodic ranked sample feedback (online DPO)');
  assert.ok(toggle, 'ranked sample feedback switch is rendered');
  assert.equal(toggle.props.disabled, false);
  assert.ok(findText(tree, 'Voting is'), 'the delayed-voting explanation is rendered');
  assert.ok(findText(tree, 'does not change Loss Type'), 'standard loss remains explained');
  r.unmount();
});

test('SimpleJob renders every/start_after/beta controls only while feedback is active', () => {
  const job = freshJob();
  job.config.process[0].model.arch = 'moss_image';
  job.config.process[0].train.loss_type = 'mse';
  job.config.process[0].train.dpo_feedback = { every: 12, start_after: 3 };
  job.config.process[0].train.dpo_beta = 100;
  const r = renderHook(props => SimpleJob(props), uiProps(job));
  const tree = r.result.current;
  assert.ok(findNode(tree, node => node.props?.label === 'Feedback Every (successful optimizer updates)'));
  assert.ok(findNode(tree, node => node.props?.label === 'Start After (optimizer updates)'));
  assert.ok(findNode(tree, node => node.props?.label === 'Feedback DPO Beta'));
  assert.ok(findText(tree, 'Regular gallery samples still follow Sample Every'));
  r.unmount();
});

test('SimpleJob disables online feedback while offline DPO is selected and its switch applies the normal-loss round trip', () => {
  const offline = freshJob();
  offline.config.process[0].model.arch = 'moss_image';
  offline.config.process[0].train.loss_type = 'dpo';
  let r = renderHook(props => SimpleJob(props), uiProps(offline));
  const disabledToggle = findNode(
    r.result.current,
    node => node.props?.label === 'Periodic ranked sample feedback (online DPO)',
  );
  assert.equal(disabledToggle.props.disabled, true);
  assert.ok(findText(r.result.current, 'Switch Loss Type away from offline DPO'));
  r.unmount();

  const job = freshJob();
  job.config.process[0].model.arch = 'moss_image';
  job.config.process[0].train.loss_type = 'mse';
  job.config.process[0].train.disable_sampling = true;
  job.config.process[0].sample.samples = [];
  const h = stateHarness(job);
  r = renderHook(props => SimpleJob(props), uiProps(h.state, h.setJobConfig));
  const toggle = findNode(
    r.result.current,
    node => node.props?.label === 'Periodic ranked sample feedback (online DPO)',
  );
  toggle.props.onChange(true);
  assert.deepEqual(h.state.config.process[0].train.dpo_feedback, { every: 10, start_after: 0 });
  assert.equal(h.state.config.process[0].train.loss_type, 'mse');
  assert.equal(h.state.config.process[0].train.disable_sampling, false);
  assert.deepEqual(h.state.config.process[0].sample.samples, [{ prompt: DPO_FEEDBACK_PROMPT_FALLBACK }]);
  r.unmount();
});

test('only the Krea Raw T2I preset adds eligibility alongside the existing MOSS presets', () => {
  const { isImageDpoEligible, isMossImageDpoEligible } = require(ts.uiPath('src/app/jobs/new/utils.ts'));
  for (const arch of ['moss_image', 'moss_image:base']) {
    assert.equal(isImageDpoEligible(arch), true);
    assert.equal(isMossImageDpoEligible(arch), true);
  }
  assert.equal(isMossImageDpoEligible('krea2'), false);
  assert.equal(isImageDpoEligible('krea2'), true);
  for (const arch of ['krea2:turbo', 'krea2:o_edit', 'krea2:o_edit_turbo', 'moss_image:edit', undefined]) {
    assert.equal(isImageDpoEligible(arch), false);
  }
});
