import type { JobConfig, SelectOption } from '@/types';

/** The simple editor always edits the first process; same convention as SimpleJob.tsx. */
export const PROCESS_PATH = 'config.process[0]';

export type JsonObject = { [key: string]: any };

export const fieldPath = (section: string, key: string): string => `${PROCESS_PATH}.${section}.${key}`;
export const datasetPath = (index: number, key: string): string => `${PROCESS_PATH}.datasets[${index}].${key}`;
export const validationDatasetPath = (index: number, key: string): string =>
  `${PROCESS_PATH}.train.validation_config.datasets[${index}].${key}`;
export const trainingExtrasPath = (key?: string): string =>
  key ? `${PROCESS_PATH}.train.training_extras.${key}` : `${PROCESS_PATH}.train.training_extras`;

/**
 * Every committed fork-specific control this component owns. Tests assert this inventory and the component
 * renders exactly these (plus conditional sub-fields) rather than a catch-all config editor.
 */
export const FORK_CONTROL_PATHS = {
  network: {
    respectAlpha: fieldPath('network', 'respect_alpha'),
    loraPlusRatio: fieldPath('network', 'lora_plus_ratio'),
    loraPlusTextEncoderRatio: fieldPath('network', 'lora_plus_text_encoder_ratio'),
    computeDtype: fieldPath('network', 'compute_dtype'),
    useScalar: `${PROCESS_PATH}.network.network_kwargs.use_scalar`,
    scalarInitValue: `${PROCESS_PATH}.network.network_kwargs.scalar_init_value`,
    fullNamedParameters: `${PROCESS_PATH}.network.network_kwargs.full_named_parameters`,
    fullLocon: `${PROCESS_PATH}.network.network_kwargs.full_locon`,
    useModelLoraIgnores: `${PROCESS_PATH}.network.network_kwargs.use_model_lora_ignores`,
  },
  train: {
    optimizer: fieldPath('train', 'optimizer'),
    lrScheduler: fieldPath('train', 'lr_scheduler'),
    lrSchedulerParams: fieldPath('train', 'lr_scheduler_params'),
    optimizerParams: fieldPath('train', 'optimizer_params'),
    allowTf32: fieldPath('train', 'allow_tf32'),
    timestepType: fieldPath('train', 'timestep_type'),
    logitNormalMean: fieldPath('train', 'logit_normal_mean'),
    logitNormalStd: fieldPath('train', 'logit_normal_std'),
    logitNormalUniform: fieldPath('train', 'logit_normal_uniform'),
    logitNormalCutoff: fieldPath('train', 'logit_normal_cutoff'),
    logitNormalCleanSchedule: fieldPath('train', 'logit_normal_clean_schedule'),
    multiresNoiseIterations: fieldPath('train', 'multires_noise_iterations'),
    multiresNoiseDiscount: fieldPath('train', 'multires_noise_discount'),
    ipNoiseGamma: fieldPath('train', 'ip_noise_gamma'),
    ipNoiseGammaRandomStrength: fieldPath('train', 'ip_noise_gamma_random_strength'),
    noiseAugmentationSeed: fieldPath('train', 'noise_augmentation_seed'),
    reseedPerStep: fieldPath('train', 'reseed_per_step'),
    maskResizeMode: fieldPath('train', 'mask_resize_mode'),
    trainingExtras: trainingExtrasPath(),
  },
  model: {
    activationOffloading: fieldPath('model', 'activation_offloading'),
    batchedCfg: `${PROCESS_PATH}.model.model_kwargs.batched_cfg`,
    sdpaCudnn: `${PROCESS_PATH}.model.model_kwargs.sdpa_cudnn`,
    fusedPointwise: `${PROCESS_PATH}.model.model_kwargs.fused_pointwise`,
    maxTextLength: `${PROCESS_PATH}.model.model_kwargs.max_text_length`,
    referenceMinShortEdge: `${PROCESS_PATH}.model.model_kwargs.reference_min_short_edge`,
    referenceMaxShortEdge: `${PROCESS_PATH}.model.model_kwargs.reference_max_short_edge`,
    textEncoderWeightPath: `${PROCESS_PATH}.model.model_kwargs.text_encoder_weight_path`,
    vaeConfigPath: `${PROCESS_PATH}.model.model_kwargs.vae_config_path`,
  },
  dataset: {
    randomCropPercent: (index: number) => datasetPath(index, 'random_crop_percent'),
    bucketTolerance: (index: number) => datasetPath(index, 'bucket_tolerance'),
    isolateModalities: (index: number) => datasetPath(index, 'isolate_modalities'),
  },
  validation: {
    config: `${PROCESS_PATH}.train.validation_config`,
    datasets: `${PROCESS_PATH}.train.validation_config.datasets`,
    maxBatches: `${PROCESS_PATH}.train.validation_config.max_batches`,
    seed: `${PROCESS_PATH}.train.validation_config.seed`,
  },
} as const;

export const getProcess = (jobConfig: JobConfig | any): JsonObject => jobConfig?.config?.process?.[0] ?? {};
export const getTrain = (jobConfig: JobConfig | any): JsonObject => getProcess(jobConfig).train ?? {};
export const getNetwork = (jobConfig: JobConfig | any): JsonObject => getProcess(jobConfig).network ?? {};
export const getModel = (jobConfig: JobConfig | any): JsonObject => getProcess(jobConfig).model ?? {};
export const getModelKwargs = (jobConfig: JobConfig | any): JsonObject => getModel(jobConfig).model_kwargs ?? {};
export const getDatasets = (jobConfig: JobConfig | any): JsonObject[] => {
  const value = getProcess(jobConfig).datasets;
  return Array.isArray(value) ? value : [];
};
export const getValidationConfig = (jobConfig: JobConfig | any): JsonObject | undefined =>
  getTrain(jobConfig).validation_config;
export const getValidationDatasets = (jobConfig: JobConfig | any): JsonObject[] => {
  const value = getValidationConfig(jobConfig)?.datasets;
  return Array.isArray(value) ? value : [];
};
export const getTrainingExtras = (jobConfig: JobConfig | any): JsonObject | undefined => {
  const value = getTrain(jobConfig).training_extras;
  return value && typeof value === 'object' && !Array.isArray(value) ? value : undefined;
};
export const getExtrasSection = (jobConfig: JobConfig | any, key: string): JsonObject =>
  getTrainingExtras(jobConfig)?.[key] ?? {};

export const numberOrNull = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

export const numberOr = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

/** `random_crop_percent` accepts a fraction (0..1) or percent (0..100); the UI always edits percent. */
export const randomCropPercentFromConfig = (value: unknown): number | null => {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN;
  if (!Number.isFinite(n) || n <= 0) return null;
  return n > 1 ? Math.min(n, 90) : Math.min(n * 100, 90);
};

/** Backend clamps to 0.9; UI cap is 90%. Clearing/zero removes the key so the default stays the default. */
export const randomCropPercentToConfig = (value: number | null): number | undefined => {
  if (value === null || !Number.isFinite(value) || value <= 0) return undefined;
  return Math.min(value, 90) / 100;
};

export const hasForkTrainingExtras = (jobConfig: JobConfig | any): boolean => !!getTrainingExtras(jobConfig);

export type ParseJsonObjectResult = { ok: true; value: JsonObject | undefined } | { ok: false; error: string };

/** Parse the only two genuinely open-ended fork config dictionaries (EDM2 args). */
export const parseJsonObject = (text: string, label = 'JSON'): ParseJsonObjectResult => {
  const trimmed = text.trim();
  if (trimmed === '') return { ok: true, value: undefined };
  try {
    const parsed = JSON.parse(trimmed);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { ok: false, error: `${label} must be a JSON object.` };
    }
    return { ok: true, value: parsed as JsonObject };
  } catch (error: any) {
    return { ok: false, error: `${label}: ${error?.message || String(error)}` };
  }
};

export const STANDARD_OPTIMIZER_OPTIONS: SelectOption[] = [
  { value: 'adafactor', label: 'Adafactor' },
  { value: 'adam', label: 'Adam' },
  { value: 'adamw', label: 'AdamW' },
  { value: 'adamw8bit', label: 'AdamW8Bit' },
  { value: 'automagic', label: 'Automagic' },
  { value: 'automagic2', label: 'Automagic v2' },
  { value: 'automagic3', label: 'Automagic v3' },
  { value: 'automagic4', label: 'Automagic v4' },
  { value: 'prodigyopt', label: 'Prodigy' },
  { value: 'prodigy8bit', label: 'Prodigy8Bit' },
];

/** Documented fork-optimizer names; custom/dotted names remain enterable through CreatableSelectInput. */
export const FORK_OPTIMIZER_OPTIONS: SelectOption[] = [
  ...STANDARD_OPTIMIZER_OPTIONS,
  { value: 'easy:WarpAINO', label: 'Fork: WarpAINO' },
  { value: 'easy:SimplifiedAdEMAMixExM', label: 'Fork: SimplifiedAdEMAMixExM' },
  { value: 'easy:SCORN', label: 'Fork: SCORN (EDM2 aux default)' },
  { value: 'easy:AdEMAMix', label: 'Fork: AdEMAMix' },
  { value: 'easy:ADOPT', label: 'Fork: ADOPT' },
  { value: 'easy:CAME', label: 'Fork: CAME' },
  { value: 'easy:SOAP', label: 'Fork: SOAP' },
  { value: 'easy:StableSPAM', label: 'Fork: StableSPAM' },
  { value: 'easy:ScheduleFreeWrapper', label: 'Fork: ScheduleFreeWrapper' },
];

export const FORK_SCHEDULER_OPTIONS: SelectOption[] = [
  { value: 'constant', label: 'Constant' },
  { value: 'cosine', label: 'Cosine' },
  { value: 'cosine_with_restarts', label: 'Cosine with restarts' },
  { value: 'step', label: 'Step' },
  { value: 'linear', label: 'Linear' },
  { value: 'constant_with_warmup', label: 'Constant with warmup' },
  { value: 'easy:RAWR', label: 'Fork: RAWR (RexAnnealingWarmRestarts)' },
];

/** Backend-supported timestep types; the fork additions are logit_normal and shift_once. */
export const FORK_TIMESTEP_TYPE_OPTIONS: SelectOption[] = [
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'linear', label: 'Linear' },
  { value: 'shift', label: 'Shift' },
  { value: 'shift_once', label: 'Shift once (fork)' },
  { value: 'weighted', label: 'Weighted' },
  { value: 'logit_normal', label: 'Logit-normal (fork)' },
  { value: 'lognorm_blend', label: 'Log-normal blend' },
  { value: 'next_sample', label: 'Next sample' },
  { value: 'one_step', label: 'One step' },
];

export const TRAINING_EXTRAS_LOSS_OPTIONS: SelectOption[] = [
  { value: 'l2', label: 'L2 (MSE alias accepted)' },
  { value: 'l1', label: 'L1 (MAE alias accepted)' },
  { value: 'standard_pseudo_huber', label: 'Standard pseudo-Huber' },
  { value: 'standard_huber', label: 'Standard Huber' },
  { value: 'standard_smooth_l1', label: 'Standard smooth L1' },
  { value: 'huber', label: 'Huber' },
  { value: 'smooth_l1', label: 'Smooth L1' },
  { value: 'x_sigmoid', label: 'x sigmoid' },
  { value: 'log_cosh', label: 'Log cosh' },
  { value: 'squared_logarithmic', label: 'Squared logarithmic (signed-domain restriction)' },
  { value: 'soft_welsch', label: 'Soft Welsch' },
  { value: 'scaled_quadratic', label: 'Scaled quadratic' },
  { value: 'standard_deviation_loss', label: 'Standard deviation loss' },
  { value: 'psnr_loss', label: 'PSNR loss (negative PSNR; no EDM2/token/dynamic)' },
  { value: 'geman_mcclure_loss', label: 'Geman-McClure loss' },
  { value: 'smooth_l2_log', label: 'Smooth L2 log' },
  { value: 'mse_pyramid_2d', label: 'MSE pyramid 2D' },
];

export const DELTA_SCHEDULE_OPTIONS: SelectOption[] = [
  { value: 'constant', label: 'Constant' },
  { value: 'exponential', label: 'Exponential' },
  { value: 'snr', label: 'SNR (DDPM only)' },
];

export const DYNAMIC_MODE_OPTIONS: SelectOption[] = [
  { value: 'none', label: 'None (static scale)' },
  { value: 'dwa', label: 'DWA (dynamic weight averaging)' },
  { value: 'gradnorm', label: 'GradNorm (direct)' },
];

export const TOPOLOGY_LOSS_OPTIONS: SelectOption[] = [
  { value: 'kl', label: 'KL' },
  { value: 'ce', label: 'Cross entropy' },
  { value: 'cosine', label: 'Cosine' },
];

export const NETWORK_COMPUTE_DTYPE_OPTIONS: SelectOption[] = [
  { value: '', label: 'FP32 (default / unset)' },
  { value: 'bf16', label: 'BF16' },
  { value: 'fp16', label: 'FP16' },
];

export const MASK_RESIZE_MODE_OPTIONS: SelectOption[] = [
  { value: 'nearest', label: 'Nearest (legacy)' },
  { value: 'area', label: 'Area average' },
];

export const isLoraLikeNetwork = (networkType: unknown): boolean =>
  ['lora', 'dora', 'locon'].includes(String(networkType ?? 'lora').toLowerCase());

export const isMossArch = (arch: unknown): boolean => String(arch ?? '').startsWith('moss_image');
export const isH3ImageArch = (arch: unknown): boolean => {
  const value = String(arch ?? '');
  return value === 'minimax_h3_image' || value === 'p_h3_image';
};
export const isLtx2Arch = (arch: unknown): boolean => String(arch ?? '').startsWith('ltx2');
export const isKrea2Arch = (arch: unknown): boolean => String(arch ?? '').startsWith('krea2');
