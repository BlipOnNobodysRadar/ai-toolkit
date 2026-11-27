'use client';

import React, { useEffect, useMemo, useState } from 'react';
import Card from '@/components/Card';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  NumberInput,
  SelectInput,
  TextAreaInput,
  TextInput,
} from '@/components/formInputs';
import type { JobConfig, SelectOption } from '@/types';
import type { ModelArch } from './options';
import {
  DELTA_SCHEDULE_OPTIONS,
  DYNAMIC_MODE_OPTIONS,
  FORK_CONTROL_PATHS,
  FORK_OPTIMIZER_OPTIONS,
  FORK_SCHEDULER_OPTIONS,
  FORK_TIMESTEP_TYPE_OPTIONS,
  MASK_RESIZE_MODE_OPTIONS,
  NETWORK_COMPUTE_DTYPE_OPTIONS,
  TOPOLOGY_LOSS_OPTIONS,
  TRAINING_EXTRAS_LOSS_OPTIONS,
  datasetPath,
  fieldPath,
  getDatasets,
  getExtrasSection,
  getModel,
  getModelKwargs,
  getNetwork,
  getProcess,
  getTrain,
  getTrainingExtras,
  getValidationConfig,
  getValidationDatasets,
  isH3ImageArch,
  isKrea2Arch,
  isLoraLikeNetwork,
  isLtx2Arch,
  isMossArch,
  numberOr,
  numberOrNull,
  parseJsonObject,
  randomCropPercentFromConfig,
  randomCropPercentToConfig,
  trainingExtrasPath,
  validationDatasetPath,
} from './ForkTrainingOptionModel';

export type SetJobConfig = (value: any, key: string) => void;

export type ForkTrainingOptionsProps = {
  jobConfig: JobConfig;
  setJobConfig: SetJobConfig;
  datasetOptions?: SelectOption[];
  modelArch?: ModelArch | null;
  /** Render the fields without their own Card (for inserting inside an existing Card). */
  embedded?: boolean;
  /** Coordinator can turn this off when it replaces SimpleJob's optimizer widget with ForkOptimizerSelect. */
  includeOptimizer?: boolean;
  /** Coordinator can turn this off when it reuses FORK_TIMESTEP_TYPE_OPTIONS in SimpleJob's existing select. */
  includeTimestepType?: boolean;
};

type JsonArgsFieldProps = {
  label: string;
  value: unknown;
  onChange: (value: Record<string, any> | undefined) => void;
};

function JsonArgsField({ label, value, onChange }: JsonArgsFieldProps) {
  const pretty = useMemo(() => (value && typeof value === 'object' ? JSON.stringify(value, null, 2) : ''), [value]);
  const [text, setText] = useState(pretty);
  const [error, setError] = useState('');

  useEffect(() => {
    setText(pretty);
    setError('');
  }, [pretty]);

  const apply = () => {
    const parsed = parseJsonObject(text, label);
    if (!parsed.ok) {
      setError(parsed.error);
      return;
    }
    setError('');
    onChange(parsed.value);
  };

  return (
    <div className="pt-2">
      <TextAreaInput label={label} value={text} onChange={setText} rows={3} placeholder="{}" className="font-mono" />
      <div className="mt-1 flex items-center gap-2">
        <button type="button" onClick={apply} className="rounded-md bg-gray-700 px-3 py-1 text-xs hover:bg-gray-600">
          Apply JSON
        </button>
        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>
    </div>
  );
}

export function ForkOptimizerSelect({
  jobConfig,
  setJobConfig,
  className,
}: {
  jobConfig: JobConfig;
  setJobConfig: SetJobConfig;
  className?: string;
}) {
  const train = getTrain(jobConfig);
  return (
    <CreatableSelectInput
      label="Optimizer"
      className={className}
      value={String(train.optimizer ?? 'adamw8bit')}
      onChange={value => setJobConfig(value, FORK_CONTROL_PATHS.train.optimizer)}
      options={FORK_OPTIMIZER_OPTIONS}
      placeholder="easy:WarpAINO or a dotted optimizer class"
    />
  );
}

export function ForkTimestepTypeSelect({
  jobConfig,
  setJobConfig,
  className,
}: {
  jobConfig: JobConfig;
  setJobConfig: SetJobConfig;
  className?: string;
}) {
  const train = getTrain(jobConfig);
  return (
    <SelectInput
      label="Timestep Type"
      className={className}
      value={String(train.timestep_type ?? 'sigmoid')}
      onChange={value => setJobConfig(value, FORK_CONTROL_PATHS.train.timestepType)}
      options={FORK_TIMESTEP_TYPE_OPTIONS}
    />
  );
}

function SectionTitle({ title, note }: { title: string; note?: string }) {
  return (
    <div className="pt-4 pb-1">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-400">{title}</h3>
      {note && <p className="mt-1 text-xs text-gray-500">{note}</p>}
    </div>
  );
}

export default function ForkTrainingOptions({
  jobConfig,
  setJobConfig,
  datasetOptions,
  modelArch,
  embedded = false,
  includeOptimizer = true,
  includeTimestepType = true,
}: ForkTrainingOptionsProps) {
  const processConfig = getProcess(jobConfig);
  const network = getNetwork(jobConfig);
  const train = getTrain(jobConfig);
  const model = getModel(jobConfig);
  const modelKwargs = getModelKwargs(jobConfig);
  const extras = getTrainingExtras(jobConfig);
  const validation = getValidationConfig(jobConfig);
  const validationDatasets = getValidationDatasets(jobConfig);
  const datasets = getDatasets(jobConfig);
  const arch = String(modelArch?.name ?? model.arch ?? '');
  const networkType = String(network.type ?? 'lora').toLowerCase();
  const loraLike = isLoraLikeNetwork(networkType);
  const networkKwargs =
    network.network_kwargs && typeof network.network_kwargs === 'object' ? network.network_kwargs : {};
  const optimizer = String(train.optimizer ?? 'adamw8bit');
  const scheduler = String(train.lr_scheduler ?? 'constant');
  const isEasyOptimizer = optimizer.startsWith('easy:') || optimizer.includes('.');
  const isSaem = /simplifiedademamixexm|ademamixexm/i.test(optimizer);
  const timestepType = String(train.timestep_type ?? 'sigmoid');
  const schedulerParams =
    train.lr_scheduler_params && typeof train.lr_scheduler_params === 'object' ? train.lr_scheduler_params : {};
  const optimizerParams =
    train.optimizer_params && typeof train.optimizer_params === 'object' ? train.optimizer_params : {};
  const extra = extras ?? {};
  const extraLoss = extra.loss && typeof extra.loss === 'string' ? extra.loss : 'l2';
  const anchor = getExtrasSection(jobConfig, 'anchor');
  const hf = getExtrasSection(jobConfig, 'hf');
  const frequency = getExtrasSection(jobConfig, 'frequency');
  const topology = getExtrasSection(jobConfig, 'topology');
  const dynamic = getExtrasSection(jobConfig, 'dynamic');
  const edm2 = getExtrasSection(jobConfig, 'edm2');
  const wavelet = getExtrasSection(jobConfig, 'wavelet');
  const tokenMining = getExtrasSection(jobConfig, 'token_mining');

  const write = (path: string, value: any) => setJobConfig(value, path);
  const writeOptionalNumber = (path: string, value: number | null) =>
    setJobConfig(value === null ? undefined : value, path);
  const writeExtra = (path: string, value: any) => write(trainingExtrasPath(path), value);
  const writeExtraOptionalNumber = (path: string, value: number | null) =>
    setJobConfig(value === null ? undefined : value, trainingExtrasPath(path));
  const writeSchedulerParam = (key: string, value: any) =>
    setJobConfig(value, `${FORK_CONTROL_PATHS.train.lrSchedulerParams}.${key}`);
  const writeOptimizerParam = (key: string, value: any) =>
    setJobConfig(value, `${FORK_CONTROL_PATHS.train.optimizerParams}.${key}`);

  const enableTrainingExtras = (enabled: boolean) => {
    if (enabled) {
      if (!extras) setJobConfig({}, FORK_CONTROL_PATHS.train.trainingExtras);
    } else {
      setJobConfig(undefined, FORK_CONTROL_PATHS.train.trainingExtras);
    }
  };

  const enableValidationDatasets = () => {
    setJobConfig({ validation_items: [], datasets: [] }, FORK_CONTROL_PATHS.validation.config);
  };

  const content = (
    <div className="space-y-1">
      <SectionTitle
        title="Optimization"
        note="Fork optimizer and scheduler names use the backend resolver (easy:… or dotted classes). Unknown YAML keys are preserved."
      />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {includeOptimizer && <ForkOptimizerSelect jobConfig={jobConfig} setJobConfig={setJobConfig} />}
        <CreatableSelectInput
          label="LR Scheduler"
          value={scheduler}
          onChange={value => setJobConfig(value, FORK_CONTROL_PATHS.train.lrScheduler)}
          options={FORK_SCHEDULER_OPTIONS}
          placeholder="easy:RAWR or a dotted scheduler class"
        />
        {isEasyOptimizer && (
          <div className="pt-1">
            <Checkbox
              label="Compile optimizer step (compile_step)"
              checked={optimizerParams.compile_step === true}
              onChange={value => writeOptimizerParam('compile_step', value)}
            />
            {isSaem && (
              <NumberInput
                label="beta1_warmup"
                className="pt-2"
                value={numberOrNull(optimizerParams.beta1_warmup)}
                onChange={value => writeOptimizerParam('beta1_warmup', value === null ? undefined : value)}
                placeholder="unset = auto (run's optimizer steps)"
                min={1}
                allowEmpty
              />
            )}
          </div>
        )}
      </div>
      {scheduler === 'easy:RAWR' && (
        <div className="grid grid-cols-1 gap-4 pt-2 md:grid-cols-2 lg:grid-cols-5">
          <NumberInput
            label="RAWR d"
            value={numberOrNull(schedulerParams.d)}
            onChange={value => writeSchedulerParam('d', value === null ? undefined : value)}
            placeholder="0.9"
            min={0}
            max={2}
            allowEmpty
          />
          <NumberInput
            label="RAWR gamma"
            value={numberOrNull(schedulerParams.gamma)}
            onChange={value => writeSchedulerParam('gamma', value === null ? undefined : value)}
            placeholder="0.9"
            min={0}
            max={2}
            allowEmpty
          />
          <NumberInput
            label="RAWR min_lr"
            value={numberOrNull(schedulerParams.min_lr)}
            onChange={value => writeSchedulerParam('min_lr', value === null ? undefined : value)}
            placeholder="1e-6"
            min={0}
            allowEmpty
          />
          <NumberInput
            label="RAWR warmup_steps"
            value={numberOrNull(schedulerParams.warmup_steps)}
            onChange={value => writeSchedulerParam('warmup_steps', value === null ? undefined : value)}
            placeholder="0"
            min={0}
            allowEmpty
          />
          <NumberInput
            label="RAWR first_cycle_max_steps"
            value={numberOrNull(schedulerParams.first_cycle_max_steps)}
            onChange={value => writeSchedulerParam('first_cycle_max_steps', value === null ? undefined : value)}
            placeholder="unset = run steps"
            min={1}
            allowEmpty
          />
        </div>
      )}

      <SectionTitle title="Sampling, noise and reproducibility" />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        {includeTimestepType && <ForkTimestepTypeSelect jobConfig={jobConfig} setJobConfig={setJobConfig} />}
        <div className="pt-1">
          <Checkbox
            label="Allow TF32"
            checked={train.allow_tf32 !== false}
            onChange={value => write(FORK_CONTROL_PATHS.train.allowTf32, value)}
          />
          <Checkbox
            className="pt-2"
            label="Reseed per step"
            checked={train.reseed_per_step !== false}
            onChange={value => write(FORK_CONTROL_PATHS.train.reseedPerStep, value)}
          />
        </div>
        <SelectInput
          label="Mask resize mode"
          value={String(train.mask_resize_mode ?? 'nearest')}
          onChange={value => write(FORK_CONTROL_PATHS.train.maskResizeMode, value)}
          options={MASK_RESIZE_MODE_OPTIONS}
        />
        <div className="pt-1">
          <Checkbox
            label="IP noise random strength"
            checked={train.ip_noise_gamma_random_strength === true}
            onChange={value => write(FORK_CONTROL_PATHS.train.ipNoiseGammaRandomStrength, value)}
          />
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 pt-2 md:grid-cols-2 lg:grid-cols-4">
        <NumberInput
          label="Multires noise iterations"
          value={numberOrNull(train.multires_noise_iterations)}
          onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.train.multiresNoiseIterations, value)}
          placeholder="0 (off)"
          min={0}
          max={32}
          allowEmpty
        />
        <NumberInput
          label="Multires noise discount"
          value={numberOrNull(train.multires_noise_discount)}
          onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.train.multiresNoiseDiscount, value)}
          placeholder="0.1"
          min={0}
          max={1}
          allowEmpty
        />
        <NumberInput
          label="IP noise gamma"
          value={numberOrNull(train.ip_noise_gamma)}
          onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.train.ipNoiseGamma, value)}
          placeholder="0.0 (off)"
          min={0}
          allowEmpty
        />
        <NumberInput
          label="Noise augmentation seed"
          value={numberOrNull(train.noise_augmentation_seed)}
          onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.train.noiseAugmentationSeed, value)}
          placeholder="0"
          min={0}
          allowEmpty
        />
      </div>
      {timestepType === 'logit_normal' && (
        <div className="grid grid-cols-1 gap-4 pt-2 md:grid-cols-2 lg:grid-cols-5">
          <NumberInput
            label="Logit-normal mean"
            value={numberOrNull(train.logit_normal_mean)}
            onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.train.logitNormalMean, value)}
            placeholder="0.0"
            allowEmpty
          />
          <NumberInput
            label="Logit-normal std"
            value={numberOrNull(train.logit_normal_std)}
            onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.train.logitNormalStd, value)}
            placeholder="1.0"
            min={Number.MIN_VALUE}
            allowEmpty
          />
          <NumberInput
            label="Logit-normal uniform mix"
            value={numberOrNull(train.logit_normal_uniform)}
            onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.train.logitNormalUniform, value)}
            placeholder="0.0"
            min={0}
            max={1}
            allowEmpty
          />
          <NumberInput
            label="Logit-normal cutoff"
            value={numberOrNull(train.logit_normal_cutoff)}
            onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.train.logitNormalCutoff, value)}
            placeholder="1.0"
            min={Number.MIN_VALUE}
            max={1}
            allowEmpty
          />
          <TextInput
            label="Clean-share schedule"
            value={String(train.logit_normal_clean_schedule ?? '')}
            onChange={value =>
              write(FORK_CONTROL_PATHS.train.logitNormalCleanSchedule, value === '' ? undefined : value)
            }
            placeholder="e.g. 60:7:30:15:0 (optional)"
          />
        </div>
      )}

      <SectionTitle
        title="Network options"
        note="These are fork additions to NetworkConfig/network_kwargs. LoRA+ only applies to LoRA/DoRA/LoCon."
      />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div className="pt-1">
          <Checkbox
            label="Respect alpha/rank"
            checked={network.respect_alpha === true}
            onChange={value => write(FORK_CONTROL_PATHS.network.respectAlpha, value)}
          />
          <Checkbox
            className="pt-2"
            label="Scalarized LoRA factors"
            checked={networkKwargs.use_scalar === true}
            onChange={value => write(FORK_CONTROL_PATHS.network.useScalar, value)}
          />
        </div>
        <SelectInput
          label="LoRA compute dtype"
          value={String(network.compute_dtype ?? '')}
          onChange={value => write(FORK_CONTROL_PATHS.network.computeDtype, value === '' ? undefined : value)}
          options={NETWORK_COMPUTE_DTYPE_OPTIONS}
        />
        <div className="pt-1">
          <Checkbox
            label="Full named parameters"
            checked={networkKwargs.full_named_parameters === true}
            onChange={value => write(FORK_CONTROL_PATHS.network.fullNamedParameters, value)}
          />
          <Checkbox
            className="pt-2"
            label="Full LoCon"
            checked={networkKwargs.full_locon === true}
            onChange={value => write(FORK_CONTROL_PATHS.network.fullLocon, value)}
          />
        </div>
        <div className="pt-1">
          <Checkbox
            label="Use model-declared LoRA ignores"
            checked={networkKwargs.use_model_lora_ignores !== false}
            onChange={value => write(FORK_CONTROL_PATHS.network.useModelLoraIgnores, value)}
          />
          <p className="mt-1 text-xs text-gray-500">
            Turn off only to resume an older run that trained a model-declared ignore.
          </p>
        </div>
      </div>
      {loraLike && (
        <div className="grid grid-cols-1 gap-4 pt-2 md:grid-cols-2 lg:grid-cols-4">
          <NumberInput
            label="LoRA+ ratio"
            value={numberOrNull(network.lora_plus_ratio)}
            onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.network.loraPlusRatio, value)}
            placeholder="1.0 (off)"
            min={Number.MIN_VALUE}
            allowEmpty
          />
          <NumberInput
            label="LoRA+ text-encoder ratio"
            value={numberOrNull(network.lora_plus_text_encoder_ratio)}
            onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.network.loraPlusTextEncoderRatio, value)}
            placeholder="defaults to LoRA+ ratio"
            min={Number.MIN_VALUE}
            allowEmpty
          />
          {networkKwargs.use_scalar === true && (
            <NumberInput
              label="Scalar init value"
              value={numberOrNull(networkKwargs.scalar_init_value)}
              onChange={value => write(FORK_CONTROL_PATHS.network.scalarInitValue, value === null ? undefined : value)}
              placeholder="0.0"
              allowEmpty
            />
          )}
        </div>
      )}

      <SectionTitle title="Model options" note="Only options that this model reads are shown." />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div className="pt-1">
          <Checkbox
            label="Activation offloading"
            checked={model.activation_offloading === true}
            onChange={value => write(FORK_CONTROL_PATHS.model.activationOffloading, value)}
          />
          <p className="mt-1 text-xs text-gray-500">Eager-only, currently MiniMax-H3 (unbenchmarked on GPU). Requires compile off.</p>
        </div>
        {(isMossArch(arch) || arch.startsWith('minimax_h3') || isH3ImageArch(arch)) && (
          <NumberInput
            label="Max text length"
            value={numberOrNull(modelKwargs.max_text_length)}
            onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.model.maxTextLength, value)}
            placeholder="model default"
            min={1}
            allowEmpty
          />
        )}
        {isMossArch(arch) && (
          <div className="pt-1">
            <Checkbox
              label="Batched CFG previews"
              checked={modelKwargs.batched_cfg !== false}
              onChange={value => write(FORK_CONTROL_PATHS.model.batchedCfg, value)}
            />
            <Checkbox
              className="pt-2"
              label="cuDNN SDPA first"
              checked={modelKwargs.sdpa_cudnn !== false}
              onChange={value => write(FORK_CONTROL_PATHS.model.sdpaCudnn, value)}
            />
            <Checkbox
              className="pt-2"
              label="Fused pointwise"
              checked={modelKwargs.fused_pointwise !== false}
              onChange={value => write(FORK_CONTROL_PATHS.model.fusedPointwise, value)}
            />
          </div>
        )}
        {isH3ImageArch(arch) && (
          <>
            <NumberInput
              label="Reference min short edge"
              value={numberOrNull(modelKwargs.reference_min_short_edge)}
              onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.model.referenceMinShortEdge, value)}
              placeholder="model default"
              min={1}
              allowEmpty
            />
            <NumberInput
              label="Reference max short edge"
              value={numberOrNull(modelKwargs.reference_max_short_edge)}
              onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.model.referenceMaxShortEdge, value)}
              placeholder="model default"
              min={1}
              allowEmpty
            />
          </>
        )}
        {isKrea2Arch(arch) && (
          <>
            <TextInput
              label="Text encoder weight path"
              value={String(modelKwargs.text_encoder_weight_path ?? '')}
              onChange={value =>
                write(FORK_CONTROL_PATHS.model.textEncoderWeightPath, value === '' ? undefined : value)
              }
              placeholder="unset = model default"
            />
            <TextInput
              label="VAE config path"
              value={String(modelKwargs.vae_config_path ?? '')}
              onChange={value => write(FORK_CONTROL_PATHS.model.vaeConfigPath, value === '' ? undefined : value)}
              placeholder="unset = model default"
            />
          </>
        )}
      </div>

      <SectionTitle
        title="Dataset options"
        note="Random-crop percent is re-rolled on every image load; >0 disables latent caching."
      />
      {datasets.length === 0 && <p className="text-xs text-gray-500">Add a dataset first.</p>}
      {datasets.map((dataset, index) => (
        <div key={index} className="grid grid-cols-1 gap-4 rounded-md bg-gray-950/60 p-3 md:grid-cols-2 lg:grid-cols-4">
          <NumberInput
            label={`Dataset ${index + 1}: random crop %`}
            value={randomCropPercentFromConfig(dataset.random_crop_percent)}
            onChange={value => {
              const converted = randomCropPercentToConfig(value);
              setJobConfig(converted, datasetPath(index, 'random_crop_percent'));
            }}
            placeholder="0 (off)"
            min={0}
            max={90}
            allowEmpty
          />
          <NumberInput
            label={`Dataset ${index + 1}: bucket tolerance`}
            value={numberOrNull(dataset.bucket_tolerance)}
            onChange={value => setJobConfig(value === null ? undefined : value, datasetPath(index, 'bucket_tolerance'))}
            placeholder="unset = model divisibility"
            min={1}
            allowEmpty
          />
          {isLtx2Arch(arch) && (
            <div className="pt-5">
              <Checkbox
                label="Isolate audio/video modalities"
                checked={dataset.isolate_modalities === true}
                onChange={value => setJobConfig(value, datasetPath(index, 'isolate_modalities'))}
              />
            </div>
          )}
        </div>
      ))}

      <SectionTitle
        title="Dynamics and training extras"
        note="Opt-in loss bridge. Only the documented training_extras keys are exposed; arbitrary top-level keys are preserved."
      />
      <div className="flex flex-wrap items-center gap-6">
        <Checkbox label="Enable training_extras" checked={!!extras} onChange={enableTrainingExtras} />
      </div>
      {extras && (
        <div className="space-y-2 rounded-md bg-gray-950/60 p-3">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
            <SelectInput
              label="Base loss"
              value={extraLoss}
              onChange={value => writeExtra('loss', value)}
              options={TRAINING_EXTRAS_LOSS_OPTIONS}
            />
            <NumberInput
              label="Huber delta"
              value={numberOrNull(extra.delta)}
              onChange={value => writeExtraOptionalNumber('delta', value)}
              placeholder="0.01"
              min={Number.MIN_VALUE}
              allowEmpty
            />
            <SelectInput
              label="Delta schedule"
              value={String(extra.delta_schedule ?? 'constant')}
              onChange={value => writeExtra('delta_schedule', value)}
              options={DELTA_SCHEDULE_OPTIONS}
            />
            <NumberInput
              label="Loss scale"
              value={numberOrNull(extra.scale)}
              onChange={value => writeExtraOptionalNumber('scale', value)}
              placeholder="1.0"
              allowEmpty
            />
          </div>
          <div className="flex flex-wrap gap-6">
            <Checkbox
              label="Loss in float64"
              checked={extra.loss_float64 === true}
              onChange={value => writeExtra('loss_float64', value)}
            />
            <Checkbox
              label="x0_direct"
              checked={extra.x0_direct === true}
              onChange={value => writeExtra('x0_direct', value)}
            />
          </div>

          <SectionTitle title="Auxiliary losses" note="A scale of 0 (or unset) keeps that auxiliary disabled." />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
            <NumberInput
              label="Anchor scale"
              value={numberOrNull(anchor.scale)}
              onChange={value => writeExtraOptionalNumber('anchor.scale', value)}
              placeholder="0 = off (e.g. 0.01)"
              min={0}
              allowEmpty
            />
            {numberOr(anchor.scale, 0) > 0 && (
              <>
                <NumberInput
                  label="Anchor levels"
                  value={numberOrNull(anchor.levels)}
                  onChange={value => writeExtraOptionalNumber('anchor.levels', value)}
                  placeholder="3"
                  min={1}
                  allowEmpty
                />
                <div className="pt-5">
                  <Checkbox
                    label="Anchor SNR weight"
                    checked={anchor.snr_weight === true}
                    onChange={value => writeExtra('anchor.snr_weight', value)}
                  />
                </div>
              </>
            )}
            <NumberInput
              label="HF scale"
              value={numberOrNull(hf.scale)}
              onChange={value => writeExtraOptionalNumber('hf.scale', value)}
              placeholder="0 = off (e.g. 0.01)"
              min={0}
              allowEmpty
            />
            {numberOr(hf.scale, 0) > 0 && (
              <>
                <NumberInput
                  label="HF patch"
                  value={numberOrNull(hf.patch)}
                  onChange={value => writeExtraOptionalNumber('hf.patch', value)}
                  placeholder="4"
                  min={1}
                  allowEmpty
                />
                <NumberInput
                  label="HF exponent"
                  value={numberOrNull(hf.exponent)}
                  onChange={value => writeExtraOptionalNumber('hf.exponent', value)}
                  placeholder="1.0"
                  allowEmpty
                />
              </>
            )}
            <NumberInput
              label="Frequency scale"
              value={numberOrNull(frequency.scale)}
              onChange={value => writeExtraOptionalNumber('frequency.scale', value)}
              placeholder="0 = off (e.g. 0.01)"
              min={0}
              allowEmpty
            />
            {numberOr(frequency.scale, 0) > 0 && (
              <>
                <NumberInput
                  label="Frequency alpha"
                  value={numberOrNull(frequency.alpha)}
                  onChange={value => writeExtraOptionalNumber('frequency.alpha', value)}
                  placeholder="1.0"
                  allowEmpty
                />
                <NumberInput
                  label="Frequency patch factor"
                  value={numberOrNull(frequency.patch_factor)}
                  onChange={value => writeExtraOptionalNumber('frequency.patch_factor', value)}
                  placeholder="1"
                  min={1}
                  allowEmpty
                />
                <div className="pt-5">
                  <Checkbox
                    label="Ave spectrum"
                    checked={frequency.ave_spectrum === true}
                    onChange={value => writeExtra('frequency.ave_spectrum', value)}
                  />
                  <Checkbox
                    className="pt-2"
                    label="Log matrix"
                    checked={frequency.log_matrix === true}
                    onChange={value => writeExtra('frequency.log_matrix', value)}
                  />
                  <Checkbox
                    className="pt-2"
                    label="Batch matrix"
                    checked={frequency.batch_matrix === true}
                    onChange={value => writeExtra('frequency.batch_matrix', value)}
                  />
                </div>
              </>
            )}
            <NumberInput
              label="Topology scale"
              value={numberOrNull(topology.scale)}
              onChange={value => writeExtraOptionalNumber('topology.scale', value)}
              placeholder="0 = off (e.g. 0.01)"
              min={0}
              allowEmpty
            />
            {numberOr(topology.scale, 0) > 0 && (
              <>
                <NumberInput
                  label="Topology tau latent"
                  value={numberOrNull(topology.tau_latent)}
                  onChange={value => writeExtraOptionalNumber('topology.tau_latent', value)}
                  placeholder="0.1"
                  min={Number.MIN_VALUE}
                  allowEmpty
                />
                <NumberInput
                  label="Topology tau target"
                  value={numberOrNull(topology.tau_target)}
                  onChange={value => writeExtraOptionalNumber('topology.tau_target', value)}
                  placeholder="0.1"
                  min={Number.MIN_VALUE}
                  allowEmpty
                />
                <NumberInput
                  label="Topology scale levels"
                  value={numberOrNull(topology.scale_levels)}
                  onChange={value => writeExtraOptionalNumber('topology.scale_levels', value)}
                  placeholder="2"
                  min={1}
                  allowEmpty
                />
                <SelectInput
                  label="Topology loss type"
                  value={String(topology.loss_type ?? 'kl')}
                  onChange={value => writeExtra('topology.loss_type', value)}
                  options={TOPOLOGY_LOSS_OPTIONS}
                />
                <NumberInput
                  label="Topology chunk size"
                  value={numberOrNull(topology.chunk_size)}
                  onChange={value => writeExtraOptionalNumber('topology.chunk_size', value)}
                  placeholder="512"
                  min={1}
                  allowEmpty
                />
                <div className="pt-5">
                  <Checkbox
                    label="Apply channel norm"
                    checked={topology.apply_channel_norm !== false}
                    onChange={value => writeExtra('topology.apply_channel_norm', value)}
                  />
                </div>
              </>
            )}
          </div>

          <SectionTitle title="Dynamic weighting" />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
            <SelectInput
              label="Mode"
              value={String(dynamic.mode ?? 'none')}
              onChange={value => writeExtra('dynamic.mode', value)}
              options={DYNAMIC_MODE_OPTIONS}
            />
            {String(dynamic.mode ?? 'none') !== 'none' && (
              <>
                <NumberInput
                  label="User weight"
                  value={numberOrNull(dynamic.user_weight)}
                  onChange={value => writeExtraOptionalNumber('dynamic.user_weight', value)}
                  placeholder="1.0"
                  min={0}
                  allowEmpty
                />
                <NumberInput
                  label="DWA temperature"
                  value={numberOrNull(dynamic.dwa_temperature)}
                  onChange={value => writeExtraOptionalNumber('dynamic.dwa_temperature', value)}
                  placeholder="2.0"
                  min={0}
                  allowEmpty
                />
                <NumberInput
                  label="GradNorm alpha"
                  value={numberOrNull(dynamic.gradnorm_alpha)}
                  onChange={value => writeExtraOptionalNumber('dynamic.gradnorm_alpha', value)}
                  placeholder="1.5"
                  allowEmpty
                />
                <NumberInput
                  label="Max weight"
                  value={numberOrNull(dynamic.max_weight)}
                  onChange={value => writeExtraOptionalNumber('dynamic.max_weight', value)}
                  placeholder="10.0"
                  min={0}
                  allowEmpty
                />
                <NumberInput
                  label="Min weight"
                  value={numberOrNull(dynamic.min_weight)}
                  onChange={value => writeExtraOptionalNumber('dynamic.min_weight', value)}
                  placeholder="0.0"
                  min={0}
                  allowEmpty
                />
                <NumberInput
                  label="History size"
                  value={numberOrNull(dynamic.history_size)}
                  onChange={value => writeExtraOptionalNumber('dynamic.history_size', value)}
                  placeholder="2"
                  min={2}
                  allowEmpty
                />
              </>
            )}
          </div>

          <SectionTitle
            title="EDM2 adaptive weighting"
            note="Independent aux optimizer/scheduler; single GPU; FP16 is unsupported."
          />
          <div className="flex flex-wrap gap-6">
            <Checkbox
              label="Enable EDM2"
              checked={edm2.enabled === true}
              onChange={value => writeExtra('edm2.enabled', value)}
            />
          </div>
          {edm2.enabled === true && (
            <div className="grid grid-cols-1 gap-4 pt-2 md:grid-cols-2 lg:grid-cols-4">
              <NumberInput
                label="EDM2 channels"
                value={numberOrNull(edm2.channels)}
                onChange={value => writeExtraOptionalNumber('edm2.channels', value)}
                placeholder="128"
                min={1}
                allowEmpty
              />
              <NumberInput
                label="EDM2 LR"
                value={numberOrNull(edm2.lr)}
                onChange={value => writeExtraOptionalNumber('edm2.lr', value)}
                placeholder="0.015"
                min={Number.MIN_VALUE}
                allowEmpty
              />
              <CreatableSelectInput
                label="EDM2 optimizer"
                value={String(edm2.optimizer ?? 'easy:SCORN')}
                onChange={value => writeExtra('edm2.optimizer', value)}
                options={FORK_OPTIMIZER_OPTIONS}
              />
              <CreatableSelectInput
                label="EDM2 scheduler"
                value={String(edm2.scheduler ?? 'constant')}
                onChange={value => writeExtra('edm2.scheduler', value)}
                options={FORK_SCHEDULER_OPTIONS}
              />
              <div className="pt-5">
                <Checkbox
                  label="EDM2 float64"
                  checked={edm2.float64 === true}
                  onChange={value => writeExtra('edm2.float64', value)}
                />
                <Checkbox
                  className="pt-2"
                  label="EDM2 importance weights"
                  checked={edm2.importance === true}
                  onChange={value => writeExtra('edm2.importance', value)}
                />
              </div>
              {edm2.importance === true && (
                <>
                  <NumberInput
                    label="Importance max"
                    value={numberOrNull(edm2.importance_max)}
                    onChange={value => writeExtraOptionalNumber('edm2.importance_max', value)}
                    placeholder="10"
                    min={Number.MIN_VALUE}
                    allowEmpty
                  />
                  <NumberInput
                    label="Importance min SNR"
                    value={numberOrNull(edm2.importance_min_snr)}
                    onChange={value => writeExtraOptionalNumber('edm2.importance_min_snr', value)}
                    placeholder="1"
                    min={Number.MIN_VALUE}
                    allowEmpty
                  />
                </>
              )}
              <JsonArgsField
                label="EDM2 optimizer_args (JSON object)"
                value={edm2.optimizer_args}
                onChange={value => writeExtra('edm2.optimizer_args', value)}
              />
              <JsonArgsField
                label="EDM2 scheduler_args (JSON object)"
                value={edm2.scheduler_args}
                onChange={value => writeExtra('edm2.scheduler_args', value)}
              />
            </div>
          )}

          <SectionTitle title="Wavelet and token mining" />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div className="pt-5">
              <Checkbox
                label="Enable wavelet"
                checked={wavelet.enabled === true}
                onChange={value => writeExtra('wavelet.enabled', value)}
              />
            </div>
            {wavelet.enabled === true && (
              <NumberInput
                label="Wavelet minimum fraction"
                value={numberOrNull(wavelet.minimum_fraction)}
                onChange={value => writeExtraOptionalNumber('wavelet.minimum_fraction', value)}
                placeholder="0.3"
                min={0}
                max={1}
                allowEmpty
              />
            )}
            <div className="pt-5">
              <Checkbox
                label="Enable token mining"
                checked={tokenMining.enabled === true}
                onChange={value => writeExtra('token_mining.enabled', value)}
              />
            </div>
            {tokenMining.enabled === true && (
              <>
                <NumberInput
                  label="Token mining alpha"
                  value={numberOrNull(tokenMining.alpha)}
                  onChange={value => writeExtraOptionalNumber('token_mining.alpha', value)}
                  placeholder="1.0"
                  min={0}
                  allowEmpty
                />
                <NumberInput
                  label="Token mining min weight"
                  value={numberOrNull(tokenMining.min_weight)}
                  onChange={value => writeExtraOptionalNumber('token_mining.min_weight', value)}
                  placeholder="0.25"
                  min={Number.MIN_VALUE}
                  allowEmpty
                />
                <NumberInput
                  label="Token mining max weight"
                  value={numberOrNull(tokenMining.max_weight)}
                  onChange={value => writeExtraOptionalNumber('token_mining.max_weight', value)}
                  placeholder="4.0"
                  min={Number.MIN_VALUE}
                  allowEmpty
                />
                <div className="pt-5">
                  <Checkbox
                    label="Sigma gate"
                    checked={tokenMining.sigma_gate !== false}
                    onChange={value => writeExtra('token_mining.sigma_gate', value)}
                  />
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <SectionTitle
        title="Native validation datasets"
        note="Alternative to validation_items. Folder roots must not overlap the training data."
      />
      {!validation ? (
        <button
          type="button"
          onClick={enableValidationDatasets}
          className="rounded-md bg-gray-700 px-3 py-1.5 text-sm hover:bg-gray-600"
        >
          Enable validation datasets
        </button>
      ) : (
        <div className="space-y-2">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
            <NumberInput
              label="Validation max batches"
              value={numberOrNull(validation.max_batches)}
              onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.validation.maxBatches, value)}
              placeholder="5"
              min={1}
              allowEmpty
            />
            <NumberInput
              label="Validation seed"
              value={numberOrNull(validation.seed)}
              onChange={value => writeOptionalNumber(FORK_CONTROL_PATHS.validation.seed, value)}
              placeholder="42"
              allowEmpty
            />
          </div>
          {validationDatasets.map((dataset, index) => {
            const value = String(dataset.folder_path ?? '');
            return (
              <div key={index} className="grid grid-cols-1 gap-3 rounded-md bg-gray-950/60 p-3 md:grid-cols-[1fr_auto]">
                {datasetOptions && datasetOptions.length > 0 ? (
                  <SelectInput
                    label={`Validation dataset ${index + 1}`}
                    value={value}
                    onChange={next => setJobConfig(next, validationDatasetPath(index, 'folder_path'))}
                    options={[{ value: '', label: '- Select folder -' }, ...datasetOptions]}
                  />
                ) : (
                  <TextInput
                    label={`Validation dataset ${index + 1} folder`}
                    value={value}
                    onChange={next => setJobConfig(next, validationDatasetPath(index, 'folder_path'))}
                    placeholder="folder path"
                  />
                )}
                <button
                  type="button"
                  onClick={() =>
                    setJobConfig(
                      validationDatasets.filter((_, datasetIndex) => datasetIndex !== index),
                      FORK_CONTROL_PATHS.validation.datasets,
                    )
                  }
                  className="self-end rounded-md bg-red-700 px-3 py-1.5 text-sm hover:bg-red-600"
                >
                  Remove
                </button>
              </div>
            );
          })}
          <button
            type="button"
            onClick={() => {
              const next = [...validationDatasets, { folder_path: '' }];
              setJobConfig(next, FORK_CONTROL_PATHS.validation.datasets);
            }}
            className="rounded-md bg-gray-700 px-3 py-1.5 text-sm hover:bg-gray-600"
          >
            Add validation dataset
          </button>
        </div>
      )}
    </div>
  );

  return embedded ? (
    content
  ) : (
    <Card title="Fork Training Options" collapsible>
      {content}
    </Card>
  );
}
