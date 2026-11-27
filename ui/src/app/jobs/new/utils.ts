import { DpoFeedbackConfig, GroupedSelectOption, JobConfig, SelectOption } from '@/types';
import { ModelArch } from './options';
import { objectCopy } from '@/utils/basic';

export const DPO_BETA_DEFAULT = 100;
export const DPO_FEEDBACK_EVERY_DEFAULT = 10;
export const DPO_FEEDBACK_START_AFTER_DEFAULT = 0;
/** Used only when enabling feedback would otherwise leave no sample to rank. */
export const DPO_FEEDBACK_PROMPT_FALLBACK = 'a high quality, detailed image';

export const isMossImageDpoEligible = (archName: string | undefined): boolean =>
  archName === 'moss_image' || archName === 'moss_image:base';

export const isImageDpoEligible = (archName: string | undefined): boolean =>
  isMossImageDpoEligible(archName) || archName === 'krea2';

export const isValidDpoBeta = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

export const isValidDpoFeedbackEvery = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 1;

export const isValidDpoFeedbackStartAfter = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;

/** The persisted contract is absent/null = off and an object = on. Keep an
 * object visible even if a hand-edited config has invalid cadence values so
 * the editor can repair it instead of silently discarding the user's mode. */
export const isDpoFeedbackEnabled = (value: unknown): value is DpoFeedbackConfig =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const isValidDpoFeedbackConfig = (value: unknown): value is DpoFeedbackConfig =>
  isDpoFeedbackEnabled(value) &&
  isValidDpoFeedbackEvery((value as DpoFeedbackConfig).every) &&
  isValidDpoFeedbackStartAfter((value as DpoFeedbackConfig).start_after);

/** Enforce the transformer-LoRA invariants shared by offline and online image
 * preference training, without changing the selected ordinary loss. */
const applyTransformerLoraMode = (
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  setJobConfig('lora', 'config.process[0].network.type');
  setJobConfig(true, 'config.process[0].network.transformer_only');
  setJobConfig(true, 'config.process[0].train.train_unet');
  setJobConfig(false, 'config.process[0].train.train_text_encoder');
};

const isPositiveNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

const isNonNegativeNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;

/** Make the normal sampling block able to produce the images the gallery will
 * rank. Existing valid prompts and cadence are retained; only missing/invalid
 * essentials are filled in. */
const ensureFeedbackSampleGeneration = (
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const process = jobConfig.config.process[0];
  const sample = process.sample;

  setJobConfig(false, 'config.process[0].train.disable_sampling');
  if (typeof sample.sampler !== 'string' || sample.sampler.trim() === '') {
    setJobConfig('flowmatch', 'config.process[0].sample.sampler');
  }
  if (!isPositiveNumber(sample.sample_every)) {
    setJobConfig(250, 'config.process[0].sample.sample_every');
  }
  if (!isNonNegativeNumber(sample.sample_start_step)) {
    setJobConfig(0, 'config.process[0].sample.sample_start_step');
  }
  if (!isPositiveNumber(sample.width)) {
    setJobConfig(1024, 'config.process[0].sample.width');
  }
  if (!isPositiveNumber(sample.height)) {
    setJobConfig(1024, 'config.process[0].sample.height');
  }
  if (!isNonNegativeNumber(sample.guidance_scale)) {
    setJobConfig(4, 'config.process[0].sample.guidance_scale');
  }
  if (!isPositiveNumber(sample.sample_steps)) {
    setJobConfig(30, 'config.process[0].sample.sample_steps');
  }
  if (!isNonNegativeNumber(sample.seed)) {
    setJobConfig(42, 'config.process[0].sample.seed');
  }

  const samples = Array.isArray(sample.samples) ? sample.samples : [];
  const hasUsablePrompt = samples.some(
    item => typeof item?.prompt === 'string' && item.prompt.trim().length > 0,
  );
  if (!hasUsablePrompt) {
    setJobConfig(
      [...samples, { prompt: DPO_FEEDBACK_PROMPT_FALLBACK }],
      'config.process[0].sample.samples',
    );
  }
};

/** Turn on interleaved feedback while leaving the ordinary loss selectable and
 * the target dataset untouched. Cadence is measured in successful optimizer
 * updates, not raw optimizer steps or regular sample_every steps. */
export const applyDpoFeedbackMode = (
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
): boolean => {
  const process = jobConfig.config.process[0];
  // The trainer deliberately rejects interleaved feedback combined with the
  // offline DPO loss. The editor disables the switch in that state, and this
  // guard keeps programmatic callers from persisting an invalid combination.
  if (process.train.loss_type === 'dpo') {
    return false;
  }
  const current = isDpoFeedbackEnabled(process.train.dpo_feedback)
    ? process.train.dpo_feedback
    : null;
  const feedback: DpoFeedbackConfig = {
    every: isValidDpoFeedbackEvery(current?.every)
      ? current!.every
      : DPO_FEEDBACK_EVERY_DEFAULT,
    start_after: isValidDpoFeedbackStartAfter(current?.start_after)
      ? current!.start_after
      : DPO_FEEDBACK_START_AFTER_DEFAULT,
  };

  setJobConfig(feedback, 'config.process[0].train.dpo_feedback');
  if (!isValidDpoBeta(process.train.dpo_beta)) {
    setJobConfig(DPO_BETA_DEFAULT, 'config.process[0].train.dpo_beta');
  }
  applyTransformerLoraMode(jobConfig, setJobConfig);
  ensureFeedbackSampleGeneration(jobConfig, setJobConfig);
  return true;
};

/** Turn off only interleaved feedback. Offline DPO keeps its beta; a standard
 * loss job no longer has any consumer for dpo_beta, so remove the stale field. */
export const clearDpoFeedbackConfig = (
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const process = jobConfig.config.process[0];
  setJobConfig(undefined, 'config.process[0].train.dpo_feedback');
  if (process.train.loss_type !== 'dpo') {
    setJobConfig(undefined, 'config.process[0].train.dpo_beta');
  }
};

/** Apply the invariants offline DPO needs without touching unrelated training
 * options. Interleaved feedback is a separate mode and is not enabled here. */
export const applyDpoMode = (
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const process = jobConfig.config.process[0];
  // Offline DPO and interleaved feedback are mutually exclusive in the
  // trainer. Selecting offline DPO clears the online switch without touching
  // its beta, which this mode immediately defaults/validates below.
  setJobConfig(undefined, 'config.process[0].train.dpo_feedback');
  setJobConfig('dpo', 'config.process[0].train.loss_type');
  setJobConfig('flowmatch', 'config.process[0].train.noise_scheduler');
  if (!isValidDpoBeta(process.train.dpo_beta)) {
    setJobConfig(DPO_BETA_DEFAULT, 'config.process[0].train.dpo_beta');
  }
  applyTransformerLoraMode(jobConfig, setJobConfig);
};

/** Remove offline-DPO-only fields when leaving DPO or an eligible architecture.
 * Keep dpo_beta if interleaved feedback still needs it. */
export const clearDpoConfig = (
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const process = jobConfig.config.process[0];
  setJobConfig(undefined, 'config.process[0].train.dpo_reference_lora_path');
  setJobConfig(undefined, 'config.process[0].train.dpo_reference_lora_strength');
  if (!isDpoFeedbackEnabled(process.train.dpo_feedback)) {
    setJobConfig(undefined, 'config.process[0].train.dpo_beta');
  }
  jobConfig.config.process[0].datasets.forEach((_, i) => {
    setJobConfig(undefined, `config.process[0].datasets[${i}].rejected_path`);
  });
};

const expandDatasetDefaults = (
  defaults: { [key: string]: any },
  numDatasets: number,
): { [key: string]: any } => {
  // expands the defaults for datasets[x] to datasets[0], datasets[1], etc.
  const expandedDefaults: { [key: string]: any } = { ...defaults };
  for (const key in defaults) {
    if (key.includes('datasets[x].')) {
      for (let i = 0; i < numDatasets; i++) {
        const datasetKey = key.replace('datasets[x].', `datasets[${i}].`);
        const v = defaults[key];
        expandedDefaults[datasetKey] = Array.isArray(v) ? [...v] : objectCopy(v);
      }
      delete expandedDefaults[key];
    }
  }
  return expandedDefaults;
};

export const handleModelArchChange = (
  modelArchs: ModelArch[],
  currentArchName: string,
  newArchName: string,
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const currentArch = modelArchs.find(a => a.name === currentArchName);
  if (!currentArch || currentArch.name === newArchName) {
    return;
  }

  // update the defaults when a model is selected
  const newArch = modelArchs.find(model => model.name === newArchName);

  const newArchSupportsDpo = isImageDpoEligible(newArchName);

  // DPO fields are only valid on an eligible non-edit image text-to-image arch.
  // This covers both offline DPO and periodic ranked-sample feedback.
  if (!newArchSupportsDpo) {
    if (jobConfig.config.process[0].train.loss_type === 'dpo') {
      setJobConfig('mse', 'config.process[0].train.loss_type');
    }
    setJobConfig(undefined, 'config.process[0].train.dpo_beta');
    setJobConfig(undefined, 'config.process[0].train.dpo_feedback');
    setJobConfig(undefined, 'config.process[0].train.dpo_reference_lora_path');
    setJobConfig(undefined, 'config.process[0].train.dpo_reference_lora_strength');
  }

  // update vram setting
  if (!newArch?.additionalSections?.includes('model.low_vram')) {
    setJobConfig(false, 'config.process[0].model.low_vram');
  }

  // handle layer offloading setting
  if (!newArch?.additionalSections?.includes('model.layer_offloading')) {
    if ('layer_offloading' in jobConfig.config.process[0].model) {
      const newModel = objectCopy(jobConfig.config.process[0].model);
      delete newModel.layer_offloading;
      delete newModel.layer_offloading_text_encoder_percent;
      delete newModel.layer_offloading_transformer_percent;
      setJobConfig(newModel, 'config.process[0].model');
    }
  } else {
    // set to false if not set
    if (!('layer_offloading' in jobConfig.config.process[0].model)) {
      setJobConfig(false, 'config.process[0].model.layer_offloading');
      setJobConfig(1.0, 'config.process[0].model.layer_offloading_text_encoder_percent');
      setJobConfig(1.0, 'config.process[0].model.layer_offloading_transformer_percent');
    }
  }

  const numDatasets = jobConfig.config.process[0].datasets.length;

  let currentDefaults = expandDatasetDefaults(currentArch.defaults || {}, numDatasets);
  let newDefaults = expandDatasetDefaults(newArch?.defaults || {}, numDatasets);

  // set new model
  setJobConfig(newArchName, 'config.process[0].model.arch');

  // update datasets
  const hasControlPath = newArch?.additionalSections?.includes('datasets.control_path') || false;
  const hasMultiControlPaths = newArch?.additionalSections?.includes('datasets.multi_control_paths') || false;
  const hasNumFrames = newArch?.additionalSections?.includes('datasets.num_frames') || false;
  const hasAutoFrameCount = newArch?.additionalSections?.includes('datasets.auto_frame_count') || false;
  const controls = newArch?.controls ?? [];
  const datasets = jobConfig.config.process[0].datasets.map(dataset => {
    const newDataset = objectCopy(dataset);
    newDataset.controls = controls;
    if (!newArchSupportsDpo) {
      delete newDataset.rejected_path;
    }
    if (hasMultiControlPaths) {
      // make sure the config has the multi control paths
      newDataset.control_path_1 = newDataset.control_path_1 || null;
      newDataset.control_path_2 = newDataset.control_path_2 || null;
      newDataset.control_path_3 = newDataset.control_path_3 || null;
      // if we previously had a single control path and now
      // we selected a multi control model
      if (newDataset.control_path && newDataset.control_path !== '') {
        // only set if not overwriting
        if (!newDataset.control_path_1) {
          newDataset.control_path_1 = newDataset.control_path;
        }
      }
      delete newDataset.control_path; // remove single control path
    } else if (hasControlPath) {
      newDataset.control_path = newDataset.control_path || null;
      if (newDataset.control_path_1 && newDataset.control_path_1 !== '') {
        newDataset.control_path = newDataset.control_path_1;
      }
      if ('control_path_1' in newDataset) {
        delete newDataset.control_path_1;
      }
      if ('control_path_2' in newDataset) {
        delete newDataset.control_path_2;
      }
      if ('control_path_3' in newDataset) {
        delete newDataset.control_path_3;
      }
    } else {
      // does not have control images
      if ('control_path' in newDataset) {
        delete newDataset.control_path;
      }
      if ('control_path_1' in newDataset) {
        delete newDataset.control_path_1;
      }
      if ('control_path_2' in newDataset) {
        delete newDataset.control_path_2;
      }
      if ('control_path_3' in newDataset) {
        delete newDataset.control_path_3;
      }
    }
    if (!hasNumFrames) {
      newDataset.num_frames = 1; // reset num_frames if not applicable
    }
    if (!hasAutoFrameCount) {
      delete newDataset.auto_frame_count;
    }
    return newDataset;
  });
  setJobConfig(datasets, 'config.process[0].datasets');

  // update samples
  const hasSampleCtrlImg = newArch?.additionalSections?.includes('sample.ctrl_img') || false;
  const samples = jobConfig.config.process[0].sample.samples.map(sample => {
    const newSample = objectCopy(sample);
    if (!hasSampleCtrlImg) {
      delete newSample.ctrl_img; // remove ctrl_img if not applicable
    }
    return newSample;
  });
  setJobConfig(samples, 'config.process[0].sample.samples');

  // revert defaults from previous model
  for (const key in currentDefaults) {
    setJobConfig(currentDefaults[key][1], key);
  }

  for (const key in newDefaults) {
    setJobConfig(newDefaults[key][0], key);
  }
};
