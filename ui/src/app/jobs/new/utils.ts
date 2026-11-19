import { GroupedSelectOption, JobConfig, SelectOption } from '@/types';
import { ModelArch } from './options';
import { objectCopy } from '@/utils/basic';

export const DPO_BETA_DEFAULT = 100;

/** Only the text-to-image MOSS registrations are eligible; the edit variants
 * share the moss_image backend arch but are excluded from DPO. */
export const isMossImageDpoEligible = (archName: string | undefined): boolean =>
  archName === 'moss_image' || archName === 'moss_image:base';

export const isValidDpoBeta = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

/** Apply the invariants DPO needs without touching unrelated training options. */
export const applyDpoMode = (
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const process = jobConfig.config.process[0];
  setJobConfig('dpo', 'config.process[0].train.loss_type');
  setJobConfig('flowmatch', 'config.process[0].train.noise_scheduler');
  if (!isValidDpoBeta(process.train.dpo_beta)) {
    setJobConfig(DPO_BETA_DEFAULT, 'config.process[0].train.dpo_beta');
  }
  setJobConfig('lora', 'config.process[0].network.type');
  setJobConfig(true, 'config.process[0].network.transformer_only');
  setJobConfig(true, 'config.process[0].train.train_unet');
  setJobConfig(false, 'config.process[0].train.train_text_encoder');
};

/** Remove DPO-only fields when leaving DPO or an eligible architecture. */
export const clearDpoConfig = (
  jobConfig: JobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  setJobConfig(undefined, 'config.process[0].train.dpo_beta');
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

  const newArchSupportsDpo = isMossImageDpoEligible(newArchName);

  // DPO fields are only valid on an eligible non-edit MOSS text-to-image arch.
  if (!newArchSupportsDpo) {
    if (jobConfig.config.process[0].train.loss_type === 'dpo') {
      setJobConfig('mse', 'config.process[0].train.loss_type');
    }
    setJobConfig(undefined, 'config.process[0].train.dpo_beta');
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
