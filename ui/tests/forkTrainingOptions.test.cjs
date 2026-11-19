// Fork-specific job editor parity. Exercises the actual TSX field declarations through the installed
// TypeScript transpiler and a small React server render; no browser, app server, GPU, or model is launched.
//   node --test --test-concurrency=1 ui/tests/forkTrainingOptions.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const ts = require('./harness/tsrequire.cjs');
const env = require('./harness/env.cjs');
const React = env.fromModules('react');
const { renderToStaticMarkup } = require(path.join(env.NODE_MODULES, 'react-dom', 'server.js'));

const Field = ({ label, value, checked, children, ...rest }) =>
  React.createElement(
    'div',
    {
      'data-field': label || '',
      'data-value': value === undefined || value === null ? '' : String(value),
      'data-checked': checked === undefined ? '' : String(checked),
    },
    label ? React.createElement('span', null, label) : null,
    typeof children !== 'function' ? children : null,
  );

const SelectField = ({ label, value, options }) =>
  React.createElement(
    'div',
    { 'data-field': label || '', 'data-value': value === undefined || value === null ? '' : String(value) },
    React.createElement('span', null, label),
    React.createElement('span', { 'data-options': (options || []).map(o => o.value).join(',') }),
  );

const formInputs = {
  TextInput: Field,
  TextAreaInput: Field,
  NumberInput: Field,
  SelectInput: SelectField,
  CreatableSelectInput: SelectField,
  Checkbox: ({ label, checked }) => Field({ label, checked, children: null }),
  FormGroup: ({ label, children }) => React.createElement('section', { 'data-group': label || '' }, children),
};
ts.mock('@/components/formInputs', formInputs);
ts.mock('@/components/Card', {
  __esModule: true,
  default: ({ title, children }) => React.createElement('section', { 'data-card': title || '' }, children),
});

const model = require(ts.uiPath('src', 'app', 'jobs', 'new', 'ForkTrainingOptionModel.ts'));
const component = require(ts.uiPath('src', 'app', 'jobs', 'new', 'ForkTrainingOptions.tsx'));
const { setNestedValue } = require(ts.uiPath('src', 'utils', 'hooks.tsx'));

const makeConfig = () => ({
  job: 'extension',
  config: {
    name: 'fork-parity',
    process: [
      {
        type: 'diffusion_trainer',
        training_folder: 'output',
        device: 'cuda',
        trigger_word: null,
        network: {
          type: 'lora',
          linear: 32,
          linear_alpha: 8,
          conv: 16,
          conv_alpha: 16,
          respect_alpha: true,
          lora_plus_ratio: 4,
          lora_plus_text_encoder_ratio: 6,
          compute_dtype: 'bf16',
          network_kwargs: {
            use_scalar: true,
            scalar_init_value: 0.1,
            full_named_parameters: true,
            full_locon: true,
            use_model_lora_ignores: false,
            untouched_network_kwarg: 42,
          },
        },
        save: {
          dtype: 'bf16',
          save_every: 250,
          max_step_saves_to_keep: 4,
          save_format: 'diffusers',
          push_to_hub: false,
        },
        datasets: [
          {
            folder_path: 'data',
            random_crop_percent: 0.2,
            bucket_tolerance: 64,
            isolate_modalities: true,
            untouched_dataset_key: 'keep',
          },
        ],
        train: {
          batch_size: 1,
          steps: 100,
          gradient_accumulation: 1,
          optimizer: 'easy:WarpAINO',
          optimizer_params: { weight_decay: 0.0001, compile_step: true, beta1_warmup: 100, untouched_opt_key: 1 },
          lr: 0.0001,
          lr_scheduler: 'easy:RAWR',
          lr_scheduler_params: { d: 0.45, gamma: 0.9, untouched_sched_key: 2 },
          allow_tf32: true,
          reseed_per_step: false,
          mask_resize_mode: 'area',
          multires_noise_iterations: 6,
          multires_noise_discount: 0.2,
          ip_noise_gamma: 0.01,
          ip_noise_gamma_random_strength: true,
          noise_augmentation_seed: 7,
          timestep_type: 'logit_normal',
          logit_normal_mean: 1.09,
          logit_normal_std: 0.95,
          logit_normal_uniform: 0.01,
          logit_normal_cutoff: 0.95,
          logit_normal_clean_schedule: '60:7:30:15:0',
          training_extras: {
            loss: 'log_cosh',
            delta: 0.01,
            anchor: { scale: 0.01, levels: 3, snr_weight: true },
            hf: { scale: 0.02, patch: 4, exponent: 1 },
            topology: { scale: 0.01, loss_type: 'ce' },
            dynamic: { mode: 'dwa', user_weight: 1 },
            edm2: { enabled: true, optimizer: 'easy:SCORN', optimizer_args: { betas: [0.9, 0.99] } },
            wavelet: { enabled: true, minimum_fraction: 0.3 },
            token_mining: { enabled: true, alpha: 1 },
            future_untouched_extras_key: { keep: true },
          },
          validation_config: {
            validation_items: [],
            datasets: [{ folder_path: 'validation', resolution: 512 }],
            max_batches: 7,
            seed: 123,
            validate_every_n_steps: 25,
            validation_sigmas: [0.25, 0.75],
          },
        },
        logging: { log_every: 1, use_ui_logger: true },
        model: {
          name_or_path: 'model',
          arch: 'moss_image',
          quantize: false,
          qtype: 'qfloat8',
          quantize_te: false,
          qtype_te: 'qfloat8',
          low_vram: false,
          activation_offloading: true,
          model_kwargs: {
            batched_cfg: true,
            sdpa_cudnn: false,
            fused_pointwise: true,
            max_text_length: 1024,
            untouched_model_kwarg: 'keep',
          },
        },
        sample: {
          sampler: 'flowmatch',
          sample_every: 250,
          sample_start_step: 0,
          samples: [],
          neg: '',
          seed: 1,
          walk_seed: true,
          guidance_scale: 1,
          sample_steps: 8,
          num_frames: 1,
          fps: 24,
        },
      },
    ],
  },
  meta: { name: '[name]', version: '1.0' },
});

const render = (config, extra = {}) =>
  renderToStaticMarkup(
    React.createElement(component.default, {
      jobConfig: config,
      setJobConfig: () => {},
      datasetOptions: [{ value: 'validation', label: 'validation' }],
      modelArch: { name: 'moss_image', label: 'MOSS', group: 'image' },
      embedded: true,
      ...extra,
    }),
  );

test('inventory lists every fork control path and excludes DPO', () => {
  const flat = [];
  const walk = value => {
    if (typeof value === 'function') return;
    if (typeof value === 'string') flat.push(value);
    else if (value && typeof value === 'object') Object.values(value).forEach(walk);
  };
  walk(model.FORK_CONTROL_PATHS);
  assert.ok(flat.includes('config.process[0].network.respect_alpha'));
  assert.ok(flat.includes('config.process[0].network.lora_plus_ratio'));
  assert.ok(flat.includes('config.process[0].network.compute_dtype'));
  assert.ok(flat.includes('config.process[0].network.network_kwargs.use_scalar'));
  assert.ok(flat.includes('config.process[0].train.allow_tf32'));
  assert.ok(flat.includes('config.process[0].train.timestep_type'));
  assert.ok(flat.includes('config.process[0].train.logit_normal_clean_schedule'));
  assert.ok(flat.includes('config.process[0].train.training_extras'));
  assert.ok(flat.includes('config.process[0].model.activation_offloading'));
  assert.ok(flat.includes('config.process[0].model.model_kwargs.batched_cfg'));
  assert.ok(flat.includes('config.process[0].train.validation_config.datasets'));
  assert.ok(!flat.some(p => /dpo|rejected_path/i.test(p)), "DPO is out of this worker's ownership");
});

test('random-crop percent conversion accepts legacy fraction/percent and clears to the backend default', () => {
  assert.equal(model.randomCropPercentFromConfig(0.2), 20);
  assert.equal(model.randomCropPercentFromConfig(20), 20);
  assert.equal(model.randomCropPercentFromConfig(undefined), null);
  assert.equal(model.randomCropPercentToConfig(20), 0.2);
  assert.equal(model.randomCropPercentToConfig(95), 0.9);
  assert.equal(model.randomCropPercentToConfig(0), undefined);
  assert.equal(model.randomCropPercentToConfig(null), undefined);
});

test('EDM2 argument JSON parser accepts only objects and leaves invalid input untouched', () => {
  assert.deepEqual(model.parseJsonObject('{"betas":[0.9,0.99]}'), { ok: true, value: { betas: [0.9, 0.99] } });
  assert.deepEqual(model.parseJsonObject(''), { ok: true, value: undefined });
  assert.equal(model.parseJsonObject('[1,2]').ok, false);
  assert.equal(model.parseJsonObject('{bad').ok, false);
});

test('nested setters preserve untouched old YAML keys and values', () => {
  const config = makeConfig();
  const next = setNestedValue(config, 0.25, model.FORK_CONTROL_PATHS.dataset.randomCropPercent(0));
  assert.equal(next.config.process[0].datasets[0].random_crop_percent, 0.25);
  assert.equal(next.config.process[0].datasets[0].untouched_dataset_key, 'keep');
  assert.equal(next.config.process[0].model.model_kwargs.untouched_model_kwarg, 'keep');
  assert.equal(next.config.process[0].network.network_kwargs.untouched_network_kwarg, 42);
  assert.equal(next.config.process[0].train.optimizer_params.untouched_opt_key, 1);
  assert.equal(next.config.process[0].train.lr_scheduler_params.untouched_sched_key, 2);
  assert.deepEqual(next.config.process[0].train.training_extras.future_untouched_extras_key, { keep: true });
  assert.equal(config.config.process[0].datasets[0].random_crop_percent, 0.2, 'the original object is not mutated');
});

test('actual rendered fields cover the fork parity matrix and show existing YAML values', () => {
  const html = render(makeConfig());
  for (const label of [
    'Optimizer',
    'LR Scheduler',
    'RAWR d',
    'Allow TF32',
    'Reseed per step',
    'Mask resize mode',
    'Multires noise iterations',
    'Multires noise discount',
    'IP noise gamma',
    'Noise augmentation seed',
    'Logit-normal mean',
    'Logit-normal std',
    'Logit-normal uniform mix',
    'Logit-normal cutoff',
    'Clean-share schedule',
    'Respect alpha/rank',
    'LoRA+ ratio',
    'LoRA+ text-encoder ratio',
    'LoRA compute dtype',
    'Scalarized LoRA factors',
    'Scalar init value',
    'Full named parameters',
    'Full LoCon',
    'Use model-declared LoRA ignores',
    'Activation offloading',
    'Max text length',
    'Batched CFG previews',
    'cuDNN SDPA first',
    'Fused pointwise',
    'Dataset 1: random crop %',
    'Dataset 1: bucket tolerance',
    'Enable training_extras',
    'Base loss',
    'Anchor scale',
    'HF scale',
    'Frequency scale',
    'Topology scale',
    'Mode',
    'Enable EDM2',
    'EDM2 optimizer',
    'EDM2 optimizer_args (JSON object)',
    'Enable wavelet',
    'Enable token mining',
    'Validation max batches',
    'Validation seed',
    'Validation dataset 1',
  ]) {
    assert.match(html, new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')), `missing rendered field: ${label}`);
  }
  assert.match(html, /data-field="Optimizer" data-value="easy:WarpAINO"/);
  assert.match(html, /data-field="LR Scheduler" data-value="easy:RAWR"/);
  assert.match(html, /data-field="Dataset 1: random crop %" data-value="20"/);
  assert.match(html, /data-field="Mask resize mode" data-value="area"/);
  assert.match(html, /data-field="Logit-normal mean" data-value="1.09"/);
  assert.match(html, /data-field="Respect alpha\/rank"[^>]*data-checked="true"/);
  assert.match(html, /data-field="Batched CFG previews"[^>]*data-checked="true"/);
  assert.match(html, /data-field="Validation max batches" data-value="7"/);
});

test('model-specific fields are gated by arch and LTX2 isolation is exposed', () => {
  const config = makeConfig();
  config.config.process[0].model.arch = 'ltx2.3';
  config.config.process[0].datasets[0].isolate_modalities = true;
  const html = render(config, { modelArch: { name: 'ltx2.3', label: 'LTX', group: 'video' } });
  assert.match(html, /Isolate audio\/video modalities/);
  assert.doesNotMatch(html, /Batched CFG previews/);
  assert.doesNotMatch(html, /Reference min short edge/);
});

test('optional coordinator switches avoid duplicate optimizer/timestep widgets', () => {
  const html = render(makeConfig(), { includeOptimizer: false, includeTimestepType: false });
  assert.doesNotMatch(html, /data-field="Optimizer"/);
  assert.doesNotMatch(html, /data-field="Timestep Type"/);
  assert.match(html, /data-field="LR Scheduler"/);
  assert.match(html, /Logit-normal mean/);
});
