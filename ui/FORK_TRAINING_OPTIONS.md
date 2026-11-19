# Fork Training Options (simple job editor)

`ui/src/app/jobs/new/ForkTrainingOptions.tsx` is the dedicated normal-form editor for committed fork-only
training controls. It is intentionally not a generic YAML/JSON editor and it does not own DPO fields.

The component writes only the exact nested key the user edits. Unknown top-level YAML and unknown nested
`training_extras`/optimizer/scheduler keys stay in place. Back-end defaults are shown as placeholders; they are
not inserted into a job until the user changes that field (except the two open EDM2 argument dictionaries,
which are explicitly applied as JSON objects).

## Coverage

### Optimization

- `train.optimizer`: standard names plus fork `easy:` names; arbitrary dotted names remain enterable.
- `train.lr_scheduler`: standard names plus `easy:RAWR`; arbitrary dotted names remain enterable.
- `train.optimizer_params.compile_step` when the optimizer uses the fork resolver.
- `train.optimizer_params.beta1_warmup` for SimplifiedAdEMAMixExM (unset means backend auto-fills the run length).
- `train.lr_scheduler_params.{d,gamma,min_lr,warmup_steps,first_cycle_max_steps}` for `easy:RAWR`.

### Sampling, noise, reproducibility

- `train.allow_tf32`
- `train.reseed_per_step`
- `train.mask_resize_mode`
- `train.timestep_type` (including fork `logit_normal` / `shift_once`)
- `train.logit_normal_{mean,std,uniform,cutoff}`
- `train.logit_normal_clean_schedule`
- `train.multires_noise_iterations`
- `train.multires_noise_discount`
- `train.ip_noise_gamma`
- `train.ip_noise_gamma_random_strength`
- `train.noise_augmentation_seed`

### Network

- `network.respect_alpha`
- `network.lora_plus_ratio`
- `network.lora_plus_text_encoder_ratio`
- `network.compute_dtype`
- `network.network_kwargs.use_scalar`
- `network.network_kwargs.scalar_init_value`
- `network.network_kwargs.full_named_parameters`
- `network.network_kwargs.full_locon`
- `network.network_kwargs.use_model_lora_ignores`

### Model

- `model.activation_offloading`
- MOSS: `model.model_kwargs.{batched_cfg,sdpa_cudnn,fused_pointwise,max_text_length}`
- MiniMax-H3 Image: `model.model_kwargs.{max_text_length,reference_min_short_edge,reference_max_short_edge}`
- Krea2: `model.model_kwargs.{text_encoder_weight_path,vae_config_path}`

### Datasets and validation

- `datasets[].random_crop_percent` (edited in percent; stored as a 0..0.9 fraction accepted by the back end)
- `datasets[].bucket_tolerance`
- `datasets[].isolate_modalities` (LTX-2 only)
- `train.validation_config.datasets`
- `train.validation_config.max_batches`
- `train.validation_config.seed`

### Dynamics / `training_extras`

- Base bridge: `loss`, `delta`, `delta_schedule`, `scale`, `loss_float64`, `x0_direct`
- Auxiliary: `anchor`, `hf`, `frequency`, `topology`
- `dynamic` DWA/GradNorm settings
- `edm2` including its independent optimizer/scheduler and `optimizer_args` / `scheduler_args`
- `wavelet` and `token_mining`

The accepted top-level keys are exactly those enforced by `training_extras/runtime.py`. No invention of
undocumented keys is performed.

## Coordinator integration

The component takes:

```tsx
type ForkTrainingOptionsProps = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  datasetOptions?: SelectOption[];
  modelArch?: ModelArch | null;
  embedded?: boolean;
  includeOptimizer?: boolean;      // default true
  includeTimestepType?: boolean;   // default true
};
```

Recommended duplicate-free integration when the existing SimpleJob widgets are retained/replaced:

- Replace SimpleJob's fixed optimizer `SelectInput` with `ForkOptimizerSelect` and pass
  `includeOptimizer={false}` to `ForkTrainingOptions`.
- Replace SimpleJob's timestep-type `options` array with the exported `FORK_TIMESTEP_TYPE_OPTIONS`
  (or use `ForkTimestepTypeSelect`) and pass `includeTimestepType={false}`.
- Insert `ForkTrainingOptions` anywhere in the simple form; it renders its own `Card` unless `embedded` is set.

The run-notes file with exact current line numbers and field mapping is
`C:/Users/someb/Documents/Codex/2026-11-18/ad/work/runs/overnight-integration/parity/wiring.md`.

## Evidence

```powershell
node --test --test-concurrency=1 ui/tests/forkTrainingOptions.test.cjs
cd ui; .\node_modules\.bin\tsc --noEmit -p tsconfig.json
```
