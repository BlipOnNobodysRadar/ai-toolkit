import copy
import inspect
import json
import random
import shutil
from collections import OrderedDict
import os
import re
import traceback
from typing import Union, List, Optional

import numpy as np
import yaml
from PIL import Image, ImageOps
from torchvision import transforms
from diffusers import T2IAdapter, ControlNetModel
from diffusers.training_utils import compute_density_for_timestep_sampling
from safetensors.torch import save_file, load_file
# from lycoris.config import PRESET
from torch.utils.data import DataLoader
import torch
import torch.backends.cuda
from huggingface_hub import HfApi, interpreter_login
from toolkit.memory_management import MemoryManager

from toolkit.basic import value_map
from toolkit.buckets import get_bucket_for_image_size
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.data_loader import get_dataloader_from_datasets, trigger_dataloader_setup_epoch
from toolkit.data_transfer_object.data_loader import FileItemDTO, DataLoaderBatchDTO
from toolkit.ema import ExponentialMovingAverage
from toolkit.embedding import Embedding
from toolkit.image_utils import show_tensors, show_latents, reduce_contrast
from toolkit.ip_adapter import IPAdapter
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.lorm import convert_diffusers_unet_to_lorm, count_parameters, print_lorm_extract_details, \
    lorm_ignore_if_contains, lorm_parameter_threshold, LORM_TARGET_REPLACE_MODULE
from toolkit.lycoris_special import LycorisSpecialNetwork
from toolkit.models.decorator import Decorator
from toolkit.network_mixins import Network
from toolkit.optimizer import get_optimizer
from toolkit.paths import CONFIG_ROOT
from toolkit.progress_bar import ToolkitProgressBar
from toolkit.prompt_utils import concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.sampler import get_sampler
from toolkit import sampling_state
from toolkit.saving import save_t2i_from_diffusers, load_t2i_model, save_ip_adapter_from_diffusers, \
    load_ip_adapter_model, load_custom_adapter_model

from toolkit.scheduler import get_lr_scheduler
from toolkit.sd_device_states_presets import get_train_sd_device_state_preset
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork

from jobs.process import BaseTrainProcess
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors, add_base_model_info_to_meta, \
    parse_metadata_from_safetensors
from toolkit.train_tools import get_torch_dtype, LearnableSNRGamma, apply_learnable_snr_gos, apply_snr_weight
from toolkit.timestep_indices import balanced_timestep_indices, last_train_index
import gc

from tqdm import tqdm

from toolkit.config_modules import SaveConfig, LoggingConfig, SampleConfig, NetworkConfig, TrainConfig, ModelConfig, \
    GenerateImageConfig, EmbeddingConfig, DatasetConfig, preprocess_dataset_raw_config, AdapterConfig, GuidanceConfig, validate_configs, \
    DecoratorConfig
from toolkit.logging_aitk import create_logger
from diffusers import FluxTransformer2DModel
from toolkit.accelerator import get_accelerator, unwrap_model
from toolkit.print import print_acc
from accelerate import Accelerator
from accelerate.optimizer import move_to_device
from accelerate.utils import DistributedType
import transformers
import diffusers
import hashlib

from toolkit.util.blended_blur_noise import get_blended_blur_noise
from toolkit.util.get_model import get_model_class
from toolkit.basic import flush
from toolkit import checkpoint_identity
from toolkit.optimizers import pending_grads


def mark_update_commit(trainer, reason):
    """From here on this loop iteration may be changing parameters or update state (optimizer.step(), a fused
    backward, the scheduler step). An OOM after this point cannot be skipped like a forward OOM
    (docs/STATE_MACHINES.md, M1 invariant W5)."""
    if getattr(trainer, '_update_commit_stage', None) is None:
        trainer._update_commit_stage = reason


_SAVE_STEP = re.compile(r'_(\d{9})(?!.*_\d{9})')


def save_write_rank(path):
    """Sort key for this run's checkpoints, newest last: when the save was last written, then its step.

    NTFS keeps a file's creation time when it is rewritten in place (torch.save over an existing .pt, a Diffusers
    folder whose files are overwritten), so ranking by getctime picked an older periodic save over a re-saved final
    one on Windows, for resume and for pruning (docs/STATE_MACHINES.md, OPUS-11/P-07). A Diffusers folder is ranked
    by its aitk_meta.yaml, which every save writes last; overwriting nested files does not touch the folder's own
    time. Equal times (coarse filesystem clocks) fall back to the step, and a save without one (the final save) last.
    """
    written = None
    if os.path.isdir(path):
        meta = os.path.join(path, 'aitk_meta.yaml')
        if os.path.exists(meta):
            written = os.path.getmtime(meta)
    if written is None:
        written = os.path.getmtime(path)
    step = _SAVE_STEP.search(os.path.basename(path))
    return written, int(step.group(1)) if step else float('inf')


# State files that share the folder and must never be taken for (or pruned as) a checkpoint
_NOT_CHECKPOINTS = ('optimizer.pt', 'training_extras.pt')


class SaveSet:
    """One save of one artifact kind: its files by model-declared suffix (Wan 2.2 14B writes _high_noise and
    _low_noise halves for one LoRA). Complete when every declared file exists."""

    def __init__(self, step, extension, files, suffixes):
        self.step, self.extension, self.files, self.suffixes = step, extension, files, suffixes
        self.complete = all(s in files for s in suffixes)
        self.rank = max(save_write_rank(p) for p in files.values())

    @property
    def path(self):
        # the first declared file; a split model's load_lora finds the other halves from it
        return self.files.get(self.suffixes[0]) or next(iter(self.files.values()))

    def missing(self):
        return [s for s in self.suffixes if s not in self.files]


def find_save_sets(root, name, suffixes=('',), post=''):
    """This name's saves in root, oldest first: exactly {name}, optionally _<9-digit step>, a declared suffix, post,
    and .safetensors/.pt or nothing (a Diffusers folder). Listed, never globbed: a glob treats a balanced [...] in the
    folder or the name as a character class, which missed this run's saves and matched other folders and other jobs
    (docs/STATE_MACHINES.md, OPUS-RES-01). Derived exports such as {name}_000000300-comfy never match."""
    if not os.path.isdir(root):
        return []
    flags = re.IGNORECASE if os.name == 'nt' else 0
    patterns = [(s, re.compile(rf'{re.escape(name)}(?:_(\d{{9}}))?{re.escape(s)}{re.escape(post)}(\.safetensors|\.pt)?',
                               flags)) for s in suffixes]
    grouped = {}
    for entry in os.listdir(root):
        if entry.lower() in _NOT_CHECKPOINTS:
            continue
        for suffix, pattern in patterns:
            match = pattern.fullmatch(entry)
            if match:
                key = (match.group(1), (match.group(2) or '').lower())
                grouped.setdefault(key, {})[suffix] = os.path.join(root, entry)
                break
    sets = [SaveSet(None if step is None else int(step), ext, files, tuple(suffixes))
            for (step, ext), files in grouped.items()]
    return sorted(sets, key=lambda s: s.rank)


def _is_cuda_oom(e):
    # torch.cuda.OutOfMemoryError is a RuntimeError; older paths only carry the message
    return isinstance(e, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(e)


class BaseSDTrainProcess(BaseTrainProcess):
    # What this trainer's hook_train_loop implements (docs/STATE_MACHINES.md M27). Trainers that zero gradients and
    # step in every hook call cannot accumulate across iterations; one that uses only the first batch of the list
    # cannot take gradient_accumulation > 1.
    supports_gradient_accumulation_steps = True
    supports_gradient_accumulation_batches = True

    def __init__(self, process_id: int, job, config: OrderedDict, custom_pipeline=None):
        super().__init__(process_id, job, config)
        self.accelerator: Accelerator = get_accelerator()
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_error()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        
        self.sd: StableDiffusion
        self.embedding: Union[Embedding, None] = None

        self.custom_pipeline = custom_pipeline
        self.step_num = 0
        self.start_step = 0
        self.epoch_num = 0
        self.last_save_step = 0
        # start at 1 so we can do a sample at the start
        self.grad_accumulation_step = 1
        # if true, then we do not do an optimizer step. We are accumulating gradients
        self.is_grad_accumulation_step = False
        self.device = str(self.accelerator.device)
        self.device_torch = self.accelerator.device
        network_config = self.get_conf('network', None)
        if network_config is not None:
            self.network_config = NetworkConfig(**network_config)
        else:
            self.network_config = None
        self.train_config = TrainConfig(**self.get_conf('train', {}))
        model_config = self.get_conf('model', {})
        self.modules_being_trained: List[torch.nn.Module] = []

        # update modelconfig dtype to match train
        model_config['dtype'] = self.train_config.dtype
        self.model_config = ModelConfig(**model_config)

        self.save_config = SaveConfig(**self.get_conf('save', {}))
        self.sample_config = SampleConfig(**self.get_conf('sample', {}))
        first_sample_config = self.get_conf('first_sample', None)
        if first_sample_config is not None:
            self.has_first_sample_requested = True
            self.first_sample_config = SampleConfig(**first_sample_config)
        else:
            self.has_first_sample_requested = False
            self.first_sample_config = self.sample_config
        self.logging_config = LoggingConfig(**self.get_conf('logging', {}))
        self.logger = create_logger(self.logging_config, config, self.save_root)
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler = None
        self.data_loader: Union[DataLoader, None] = None
        self.data_loader_reg: Union[DataLoader, None] = None
        self.trigger_word = self.get_conf('trigger_word', None)

        self.guidance_config: Union[GuidanceConfig, None] = None
        guidance_config_raw = self.get_conf('guidance', None)
        if guidance_config_raw is not None:
            self.guidance_config = GuidanceConfig(**guidance_config_raw)

        # store is all are cached. Allows us to not load vae if we don't need to
        self.is_latents_cached = True
        raw_datasets = self.get_conf('datasets', None)
        if raw_datasets is not None and len(raw_datasets) > 0:
            raw_datasets = preprocess_dataset_raw_config(raw_datasets)
        self.datasets = None
        self.datasets_reg = None
        self.dataset_configs: List[DatasetConfig] = []
        self.params = []
        
        # add dataset text embedding cache to their config
        if self.train_config.cache_text_embeddings:
            for raw_dataset in raw_datasets:
                raw_dataset['cache_text_embeddings'] = True

        # pass diff output preservation to the datasets so the data loader can build
        # and cache the DOP caption (dataset trigger word replaced with the class)
        if self.train_config.diff_output_preservation and raw_datasets is not None:
            for raw_dataset in raw_datasets:
                raw_dataset['diff_output_preservation'] = True
                raw_dataset['diff_output_preservation_class'] = self.train_config.diff_output_preservation_class
        
        if raw_datasets is not None and len(raw_datasets) > 0:
            for raw_dataset in raw_datasets:
                dataset = DatasetConfig(**raw_dataset)
                # handle trigger word per dataset
                if dataset.trigger_word is None and self.trigger_word is not None:
                    dataset.trigger_word = self.trigger_word
                is_caching = dataset.cache_latents or dataset.cache_latents_to_disk
                if not is_caching:
                    self.is_latents_cached = False
                if dataset.is_reg:
                    if self.datasets_reg is None:
                        self.datasets_reg = []
                    self.datasets_reg.append(dataset)
                else:
                    if self.datasets is None:
                        self.datasets = []
                    self.datasets.append(dataset)
                self.dataset_configs.append(dataset)
        
        self.is_caching_text_embeddings = any(
            dataset.cache_text_embeddings for dataset in self.dataset_configs
        )

        # name config keys that nothing reads (typos, options under the wrong section) instead of dropping them
        from toolkit.config_key_check import check_train_process
        self.unknown_config_keys = check_train_process(self)

        self.embed_config = None
        embedding_raw = self.get_conf('embedding', None)
        if embedding_raw is not None:
            self.embed_config = EmbeddingConfig(**embedding_raw)
        
        self.decorator_config: DecoratorConfig = None
        decorator_raw = self.get_conf('decorator', None)
        if decorator_raw is not None:
            if not self.model_config.is_flux:
                raise ValueError("Decorators are only supported for Flux models currently")
            self.decorator_config = DecoratorConfig(**decorator_raw)

        # t2i adapter
        self.adapter_config = None
        adapter_raw = self.get_conf('adapter', None)
        if adapter_raw is not None:
            self.adapter_config = AdapterConfig(**adapter_raw)
            # sdxl adapters end in _xl. Only full_adapter_xl for now
            if self.model_config.is_xl and not self.adapter_config.adapter_type.endswith('_xl'):
                self.adapter_config.adapter_type += '_xl'

        # to hold network if there is one
        self.network: Union[Network, None] = None
        self.adapter: Union[T2IAdapter, IPAdapter, ClipVisionAdapter, ReferenceAdapter, CustomAdapter, ControlNetModel, None] = None
        self.embedding: Union[Embedding, None] = None
        self.decorator: Union[Decorator, None] = None

        is_training_adapter = self.adapter_config is not None and self.adapter_config.train

        self.do_lorm = self.get_conf('do_lorm', False)
        self.lorm_extract_mode = self.get_conf('lorm_extract_mode', 'ratio')
        self.lorm_extract_mode_param = self.get_conf('lorm_extract_mode_param', 0.25)
        # 'ratio', 0.25)

        # get the device state preset based on what we are training
        self.train_device_state_preset = get_train_sd_device_state_preset(
            device=self.device_torch,
            train_unet=self.train_config.train_unet,
            train_text_encoder=self.train_config.train_text_encoder,
            cached_latents=self.is_latents_cached,
            train_lora=self.network_config is not None,
            train_adapter=is_training_adapter,
            train_embedding=self.embed_config is not None,
            train_decorator=self.decorator_config is not None,
            train_refiner=self.train_config.train_refiner,
            unload_text_encoder=self.train_config.unload_text_encoder or self.is_caching_text_embeddings,
            require_grads=False  # we ensure them later
        )
        
        self.get_params_device_state_preset = get_train_sd_device_state_preset(
            device=self.device_torch,
            train_unet=self.train_config.train_unet,
            train_text_encoder=self.train_config.train_text_encoder,
            cached_latents=self.is_latents_cached,
            train_lora=self.network_config is not None,
            train_adapter=is_training_adapter,
            train_embedding=self.embed_config is not None,
            train_decorator=self.decorator_config is not None,
            train_refiner=self.train_config.train_refiner,
            unload_text_encoder=self.train_config.unload_text_encoder or self.is_caching_text_embeddings,
            require_grads=True  # We check for grads when getting params
        )

        # fine_tuning here is for training actual SD network, not LoRA, embeddings, etc. it is (Dreambooth, etc)
        self.is_fine_tuning = True
        if self.network_config is not None or is_training_adapter or self.embed_config is not None or self.decorator_config is not None:
            self.is_fine_tuning = False

        self.named_lora = False
        if self.embed_config is not None or is_training_adapter:
            self.named_lora = True
        self.snr_gos: Union[LearnableSNRGamma, None] = None
        self.ema: ExponentialMovingAverage = None
        
        validate_configs(self.train_config, self.model_config, self.save_config, self.dataset_configs)
        
        do_profiler = self.get_conf('torch_profiler', False)
        self.torch_profiler = None if not do_profiler else torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        
        self.current_boundary_index = 0
        self.steps_this_boundary = 0
        self.num_consecutive_oom = 0
        self.additional_logs = {}
        # cached latents, prompt embeds, and fixed noise for validation
        self._validation_cache = None

    def post_process_generate_image_config_list(self, generate_image_config_list: List[GenerateImageConfig]):
        # override in subclass
        return generate_image_config_list

    def sample(self, step=None, is_first=False):
        if not self.accelerator.is_main_process:
            return
        flush()
        sample_folder = os.path.join(self.save_root, 'samples')
        gen_img_config_list = []

        sample_config = self.first_sample_config if is_first else self.sample_config
        start_seed = sample_config.seed
        current_seed = start_seed

        test_image_paths = []
        if self.adapter_config is not None and self.adapter_config.test_img_path is not None:
            test_image_path_list = self.adapter_config.test_img_path
            # divide up images so they are evenly distributed across prompts
            for i in range(len(sample_config.prompts)):
                test_image_paths.append(test_image_path_list[i % len(test_image_path_list)])

        for i in range(len(sample_config.prompts)):
            if sample_config.walk_seed:
                current_seed = start_seed + i

            step_num = ''
            if step is not None:
                # zero-pad 9 digits
                step_num = f"_{str(step).zfill(9)}"

            filename = f"[time]_{step_num}_[count].{self.sample_config.ext}"

            output_path = os.path.join(sample_folder, filename)

            prompt = sample_config.prompts[i]

            # add embedding if there is one
            # note: diffusers will automatically expand the trigger to the number of added tokens
            # ie test123 will become test123 test123_1 test123_2 etc. Do not add this yourself here
            if self.embedding is not None:
                prompt = self.embedding.inject_embedding_to_prompt(
                    prompt, expand_token=True, add_if_not_present=False
                )
            if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
                prompt = self.adapter.inject_trigger_into_prompt(
                    prompt, expand_token=True, add_if_not_present=False
                )
            if self.trigger_word is not None:
                prompt = self.sd.inject_trigger_into_prompt(
                    prompt, self.trigger_word, add_if_not_present=False
                )

            extra_args = {}
            if self.adapter_config is not None and self.adapter_config.test_img_path is not None:
                extra_args['adapter_image_path'] = test_image_paths[i]
            
            sample_item = sample_config.samples[i]
            if sample_item.seed is not None:
                current_seed = sample_item.seed

            gen_img_config_list.append(GenerateImageConfig(
                prompt=prompt,  # it will autoparse the prompt
                width=sample_item.width,
                height=sample_item.height,
                negative_prompt=sample_item.neg,
                seed=current_seed,
                guidance_scale=sample_item.guidance_scale,
                guidance_rescale=sample_config.guidance_rescale,
                num_inference_steps=sample_item.sample_steps,
                sigmas=sample_item.sigmas,
                network_multiplier=sample_item.network_multiplier,
                output_path=output_path,
                output_ext=sample_config.ext,
                adapter_conditioning_scale=sample_config.adapter_conditioning_scale,
                refiner_start_at=sample_config.refiner_start_at,
                extra_values=sample_config.extra_values,
                logger=self.logger,
                num_frames=sample_item.num_frames,
                fps=sample_item.fps,
                duration=sample_item.duration,
                ctrl_img=sample_item.ctrl_img,
                ctrl_idx=sample_item.ctrl_idx,
                ctrl_img_1=sample_item.ctrl_img_1,
                ctrl_img_2=sample_item.ctrl_img_2,
                ctrl_img_3=sample_item.ctrl_img_3,
                do_cfg_norm=sample_config.do_cfg_norm,
                **extra_args
            ))

        # post process
        gen_img_config_list = self.post_process_generate_image_config_list(gen_img_config_list)

        # if we have an ema, set it to validation mode (and give the raw weights back even if sampling fails)
        ema_was_training = self.ema is not None and self.ema._is_train_mode
        if self.ema is not None:
            self.ema.eval()

        # let adapter know we are sampling
        is_custom_adapter = self.adapter is not None and isinstance(self.adapter, CustomAdapter)
        was_sampling = getattr(self.adapter, 'is_sampling', False) if is_custom_adapter else None
        if is_custom_adapter:
            self.adapter.is_sampling = True
        give_back = []
        if is_custom_adapter:
            give_back.append(lambda: setattr(self.adapter, 'is_sampling', was_sampling))
        if ema_was_training:
            give_back.append(self.ema.train)
        try:
            if is_first and hasattr(self, '_first_sample_prompts_cache'):
                regular_cache = self.sd.sample_prompts_cache
                self.sd.sample_prompts_cache = self._first_sample_prompts_cache
                give_back.append(lambda: setattr(self.sd, 'sample_prompts_cache', regular_cache))
            # send to be generated
            self.sd.generate_images(gen_img_config_list, sampler=sample_config.sampler)
            if getattr(self, '_feedback_schedule', None) is not None:
                from toolkit.feedback_samples import capture_feedback_samples
                capture_feedback_samples(self, sample_config, gen_img_config_list, step=step)
        except BaseException as primary:
            sampling_state.settle(give_back, primary)
            raise
        sampling_state.settle(give_back)
        print_acc("") # add a line break

    def update_training_metadata(self):
        o_dict = OrderedDict({
            "training_info": self.get_training_info()
        })
        o_dict['ss_base_model_version'] = self.sd.get_base_model_version()

        # o_dict = add_base_model_info_to_meta(
        #     o_dict,
        #     is_v2=self.model_config.is_v2,
        #     is_xl=self.model_config.is_xl,
        # )
        o_dict['ss_output_name'] = self.job.name

        if self.trigger_word is not None:
            # just so auto1111 will pick it up
            o_dict['ss_tag_frequency'] = {
                f"1_{self.trigger_word}": {
                    f"{self.trigger_word}": 1
                }
            }

        self.add_meta(o_dict)

    def get_training_info(self):
        info = OrderedDict({
            'step': self.step_num,
            'epoch': self.epoch_num,
        })
        cursor = getattr(self, '_clean_share_cursor', None)
        if cursor is not None:
            info['clean_share_curriculum'] = cursor.state_dict()
        feedback = getattr(self, '_feedback_schedule', None)
        if feedback is not None:
            info['dpo_feedback'] = feedback.state_dict()
        reference = getattr(self, '_dpo_reference_identity', None)
        if reference is not None:
            info['dpo_reference_lora'] = reference
        return info

    def clean_up_saves(self):
        """Keep the newest max_step_saves_to_keep step saves of each artifact kind this run writes; return the newest
        save path. Only exact names this run's writers produce are candidates (find_save_sets), each kind has its own
        budget, and a split save counts once with all its files: a mixed budget deleted the newest LoRA next to an
        adapter, and a bare prefix glob took derived exports and other jobs' checkpoints (OPUS-PRUNE-01, OPUS-RES-01).
        The final (step-less) save is never pruned, as before. Saves older than the oldest kept complete save go, an
        incomplete one included; a newer incomplete one (an interrupted save) stays for the next save to replace."""
        if not self.accelerator.is_main_process:
            return
        latest_item = None
        if not os.path.exists(self.save_root):
            return latest_item
        keep = self.save_config.max_step_saves_to_keep
        newest = None
        for kind, name, suffixes in BaseSDTrainProcess._save_artifact_kinds(self):
            sets = [s for s in find_save_sets(self.save_root, name, suffixes) if s.step is not None]
            if sets and (newest is None or sets[-1].rank > newest.rank):
                newest = sets[-1]
            kept = [s for s in sets if s.complete][-keep:] if keep else []
            if not kept:
                continue
            oldest_kept = min(s.rank for s in kept)
            for save in sets:
                if save in kept or save.rank >= oldest_kept:
                    continue
                for item in save.files.values():
                    print_acc(f"Removing old save: {item}")
                    if os.path.isdir(item):
                        shutil.rmtree(item)
                    else:
                        os.remove(item)
                    # see if a yaml file with same name exists
                    yaml_file = os.path.splitext(item)[0] + ".yaml"
                    if os.path.exists(yaml_file):
                        os.remove(yaml_file)
        if newest is not None:
            latest_item = newest.path
        return latest_item

    # Artifact names: the one source for the writers in _save_checkpoint, resume discovery and pruning. Called through
    # the class, so trainer-like namespaces (tests) work.

    def _lora_save_name(self):
        return self.job.name + ('_LoRA' if getattr(self, 'named_lora', False) else '')

    def _lora_save_suffixes(self):
        """The files the model's save_lora writes per LoRA save ('' = the path itself)."""
        declared = getattr(getattr(self, 'sd', None), 'lora_save_suffixes', None)
        if declared is None or getattr(self, 'network', None) is None:
            return ('',)
        return tuple(declared())

    def _decorator_save_name(self):
        # alongside a LoRA the decorator once took the LoRA's own filename and overwrote it at every save (OPUS-SAVE-01)
        return self.job.name + ('_decorator' if getattr(self, 'network_config', None) is not None else '')

    def _adapter_save_name(self):
        name = self.job.name
        if getattr(self, 'network_config', None) is not None or getattr(self, 'embed_config', None) is not None:
            adapter_type = self.adapter_config.type
            if adapter_type == 't2i':
                name += '_t2i'
            elif adapter_type == 'control_net':
                name += '_cn'
            elif adapter_type == 'clip':
                name += '_clip'
            elif adapter_type.startswith('ip'):
                name += '_ip'
            else:
                name += '_adapter'
        return name

    def _save_artifact_kinds(self):
        """(kind, name, suffixes) for every artifact _save_checkpoint writes with this configuration."""
        job = self.job.name
        train_config = getattr(self, 'train_config', None)
        merge = getattr(train_config, 'merge_network_on_save', False)
        network_config = getattr(self, 'network_config', None)
        fine_tuning = getattr(self, 'is_fine_tuning', network_config is None)
        kinds = []
        if fine_tuning or merge:
            kinds.append(('model', job, ('',)))
            if getattr(train_config, 'train_refiner', False):
                kinds.append(('refiner', job + '_refiner', ('',)))
        else:
            if network_config is not None:
                kinds.append(('LoRA', BaseSDTrainProcess._lora_save_name(self),
                              BaseSDTrainProcess._lora_save_suffixes(self)))
            if getattr(self, 'embed_config', None) is not None:
                kinds.append(('embedding', self.embed_config.trigger, ('',)))
            if getattr(self, 'decorator_config', None) is not None:
                kinds.append(('decorator', BaseSDTrainProcess._decorator_save_name(self), ('',)))
            adapter_config = getattr(self, 'adapter_config', None)
            if adapter_config is not None and adapter_config.train:
                kinds.append(('adapter', BaseSDTrainProcess._adapter_save_name(self), ('',)))
        kinds.append(('critic', f'CRITIC_{job}', ('',)))
        return kinds

    def _refuse_colliding_save_names(self):
        """Two artifacts must not share a filename: one would overwrite the other at every save (OPUS-SAVE-01)."""
        seen = {}
        for kind, name, suffixes in BaseSDTrainProcess._save_artifact_kinds(self):
            for suffix in suffixes:
                key = (name + suffix).casefold() if os.name == 'nt' else name + suffix
                if key in seen:
                    raise ValueError(f"The {seen[key]} and the {kind} would both be saved as {name + suffix}"
                                     f"[_step].safetensors in {self.save_root}, and one would overwrite the other; "
                                     f"rename the job or the embedding trigger")
                seen[key] = kind

    def post_save_hook(self, save_path):
        # override in subclass
        pass
    
    def done_hook(self):
        pass

    def release_accelerator_state(self):
        # toolkit/accelerator.py shares one Accelerator per process, and prepare() keeps every model, optimizer and
        # scheduler in its own lists. Without this, a later job in the same run.py process (several configs, or -r
        # after a failure) starts with this job's network, optimizer state and base model still referenced.
        self.modules_being_trained.clear()
        self.optimizer = None
        self.lr_scheduler = None
        self.params = []
        self.accelerator.free_memory()
        flush()
    
    def end_step_hook(self):
        pass

    # Accumulation window and OOM protocol (docs/STATE_MACHINES.md, M1). run() calls these once per loop iteration.

    def _cleanup_batch_list(self, batch_list):
        for fetched in batch_list:
            if isinstance(fetched, DataLoaderBatchDTO):
                with self.timer('batch_cleanup'):
                    fetched.cleanup()

    def _main_loader_has_batches(self):
        try:
            self._main_epoch_len = len(self.data_loader) if self.data_loader is not None else 0
        except TypeError:  # an iterable loader with no length
            self._main_epoch_len = 0
        return self._main_epoch_len > 0

    def _setup_accumulation_iteration(self):
        """Decide whether this iteration runs the window's last microbatch (and so steps the optimizer).
        Whole-epoch accumulation (-1) is decided by the epoch rollover in the batch fetch instead."""
        if self.train_config.gradient_accumulation_steps == -1:
            if self.step_num >= self.train_config.steps - 1:
                # the run's last iteration applies the final (possibly partial) epoch window instead of dropping it
                self.is_grad_accumulation_step = False
                self.grad_accumulation_step = 0
            return
        # since optimizer step happens in the loop, we trigger it a step early
        # since we cannot reprocess it before them
        is_optimizer_step = self.grad_accumulation_step >= self.train_config.gradient_accumulation_steps
        self.is_grad_accumulation_step = not is_optimizer_step
        if is_optimizer_step:
            self.grad_accumulation_step = 0

    def _fetch_batch_list(self, step, dataloader, dataloader_iterator, dataloader_reg, dataloader_iterator_reg,
                          is_save_step, is_sample_step):
        """One loop iteration's batches (gradient_accumulation of them; regularization batches interleaved on even
        steps). Returns (batch_list, dataloader_iterator, dataloader_iterator_reg); an epoch end replaces an iterator.
        With whole-epoch accumulation (gradient_accumulation_steps -1) the iteration that fetches an epoch's last main
        batch is the update, counted against the loader's length (docs/STATE_MACHINES.md M1, C13)."""
        batch_list = []
        epoch_complete = False

        for b in range(self.train_config.gradient_accumulation):
            # keep track to alternate on an accumulation step for reg
            batch_step = step
            # don't do a reg step on sample or save steps as we dont want to normalize on those
            if batch_step % 2 == 0 and dataloader_reg is not None and not is_save_step and not is_sample_step:
                try:
                    with self.timer('get_batch:reg'):
                        batch = next(dataloader_iterator_reg)
                except StopIteration:
                    with self.timer('reset_batch:reg'):
                        # hit the end of an epoch, reset
                        self._seed_epoch_sampler(dataloader_reg, 2, step)
                        dataloader_iterator_reg = iter(dataloader_reg)
                        trigger_dataloader_setup_epoch(dataloader_reg)

                    with self.timer('get_batch:reg'):
                        batch = next(dataloader_iterator_reg)
                is_reg_step = True
            elif dataloader is not None:
                try:
                    with self.timer('get_batch'):
                        batch = next(dataloader_iterator)
                except StopIteration:
                    with self.timer('reset_batch'):
                        # hit the end of an epoch, reset
                        self._data_epoch += 1
                        self._seed_epoch_sampler(dataloader, 1, self._data_epoch)
                        dataloader_iterator = iter(dataloader)
                        trigger_dataloader_setup_epoch(dataloader)
                        self.epoch_num += 1
                        if (self.train_config.gradient_accumulation_steps == -1
                                and self._epoch_cursor < self._main_epoch_len):
                            raise RuntimeError(
                                f"The dataloader ended its epoch after {self._epoch_cursor} of the "
                                f"{self._main_epoch_len} batches its length reports; whole-epoch accumulation "
                                f"(gradient_accumulation_steps: -1) needs an accurate length")
                        self._epoch_cursor = 0
                        if self.train_config.gradient_accumulation_steps == -1:
                            # setup_epoch may rebuild the buckets, and with them the number of batches
                            self._main_loader_has_batches()
                    with self.timer('get_batch'):
                        batch = next(dataloader_iterator)
                self._epoch_cursor += 1
                if (self.train_config.gradient_accumulation_steps == -1
                        and self._epoch_cursor >= self._main_epoch_len):
                    # this batch completes the epoch: this iteration is the window's update (C13). (TrainConfig refuses
                    # gradient_accumulation > 1 with -1, so the list cannot hold more; the break only guards that.)
                    self.is_grad_accumulation_step = False
                    self.grad_accumulation_step = 0
                    epoch_complete = True
            else:
                batch = None
            batch_list.append(batch)
            batch_step += 1
            if epoch_complete:
                break
        return batch_list, dataloader_iterator, dataloader_iterator_reg

    def _install_update_commit_observers(self, optimizer):
        """Note when an iteration starts changing parameters, for any trainer's hook_train_loop.
        The handles are recorded as they are made, so run()'s finally removes them even after a partial install."""
        self._remove_update_commit_observers()
        self._update_commit_observable = False
        handles = self._update_commit_handles
        fused = pending_grads.steps_in_backward(optimizer)
        if fused:
            # a fused optimizer updates each parameter inside backward; the first gradient marks the commit
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        handles.append(p.register_hook(
                            lambda grad: mark_update_commit(self, 'a fused backward update')))
        register = getattr(getattr(optimizer, 'optimizer', optimizer), 'register_step_pre_hook', None)
        if register is not None:
            handles.append(register(lambda *args: mark_update_commit(self, 'optimizer.step()')))
        self._update_commit_observable = fused or register is not None

    def _remove_update_commit_observers(self):
        handles = getattr(self, '_update_commit_handles', None) or []
        self._update_commit_handles = []
        for handle in handles:
            handle.remove()  # RemovableHandle.remove only drops a dict entry

    def _discard_accumulation_window(self, optimizer):
        """Drop the whole unfinished window, hidden low-precision buffers included; the next iteration starts a
        fresh one, so no update averages a partial window."""
        optimizer.zero_grad(set_to_none=True)
        pending_grads.discard(optimizer)
        if getattr(self, 'training_extras', None) is not None:
            self.training_extras.edm.zero_grad()
        if self.train_config.gradient_accumulation_steps == -1:
            print_acc("OOM discarded the gradients accumulated so far this epoch; the next update covers only the "
                      "batches after it")
        # the end-of-iteration increment makes this 1: a window with no pending microbatches
        self.grad_accumulation_step = 0

    def _run_train_step(self, step, batch_list, optimizer):
        """Run hook_train_loop under the OOM protocol; returns (loss_dict, did_oom).

        An OOM before any update began skips the iteration and its whole accumulation window. The skipped loop step
        still counts as completed: it consumed its batch and per-step seeds exactly as the uninterrupted run did, so
        a save at this step resumes at the next one. An OOM after an update began cannot be skipped or retried; it
        aborts before anything is saved.
        """
        self._update_commit_stage = None
        try:
            with self.accelerator.accumulate(self.modules_being_trained):
                loss_dict = self.hook_train_loop(batch_list)
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise  # not an OOM; surface real errors
            oom = e
        else:
            self._checkpoint_completed_step = step + 1
            self.num_consecutive_oom = 0
            return loss_dict, False
        stage = self._update_commit_stage
        if stage is None and not self.is_grad_accumulation_step and not getattr(self, '_update_commit_observable', False):
            stage = 'an optimizer update this trainer cannot observe'
        if stage is not None:
            raise RuntimeError(f"OOM at step {step} after {stage} began changing parameters; the step cannot be "
                               f"skipped or retried. Resume from the last saved checkpoint.") from oom
        self.num_consecutive_oom += 1
        if self.num_consecutive_oom > 3:
            raise RuntimeError("OOM during training step 3 times in a row, aborting training") from oom
        del oom  # its traceback pins the failed step's activations; release them before flushing
        self._discard_accumulation_window(optimizer)
        self._checkpoint_completed_step = step + 1
        flush()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        # skip this step and keep going
        print_acc("")
        print_acc("################################################")
        print_acc(f"# OOM during training step, skipping batch {self.num_consecutive_oom}/3 #")
        print_acc("################################################")
        print_acc("")
        return None, True

    def _end_loop_iteration(self, step):
        self.step_num = step + 1
        self.grad_accumulation_step += 1

    def _save_at_loop_step(self):
        # print above the progress bar
        if self.progress_bar is not None:
            self.progress_bar.pause()
        print_acc(f"\nSaving at step {self.step_num}")
        self.save(self.step_num)
        self.ensure_params_requires_grad()
        # Saving must not discard the unfinished accumulation window, for any scheduler (RAWR checkpoints also
        # carry it). With gradient_accumulation_steps 1 nothing is pending here.
        flush()
        if self.progress_bar is not None:
            self.progress_bar.unpause()

    def _load_optimizer_state(self, optimizer, rawr_resume_enabled):
        """Load optimizer.pt (RAWR: the one paired with the selected model). Returns the RAWR capture or None."""
        from training_extras import rawr_checkpoint
        rawr_resume = None
        # check if it exists
        optimizer_state_filename = f'optimizer.pt'
        optimizer_state_file_path = os.path.join(self.save_root, optimizer_state_filename)
        if rawr_resume_enabled:
            optimizer_state_file_path = getattr(self, '_rawr_paired_optimizer_path', optimizer_state_file_path)
        if rawr_resume_enabled and self.step_num > 0 and not os.path.exists(optimizer_state_file_path):
            raise RuntimeError('RAWR resume requires the optimizer.pt paired with the selected model; start a new run for weights-only training')
        if os.path.exists(optimizer_state_file_path):
            # try to load
            # previous param groups
            # previous_params = copy.deepcopy(optimizer.param_groups)
            previous_lrs = []
            for group in optimizer.param_groups:
                previous_lrs.append(group['lr'])

            load_optimizer = True
            if self.network is not None:
                if self.network.did_change_weights:
                    # do not load optimizer if the network changed, it will result in
                    # a double state that will oom.
                    load_optimizer = False
                    print_acc("The LoRA rank changed since this checkpoint: continuing with fresh optimizer state "
                              "(not an exact continuation)")

            if load_optimizer:
                BaseSDTrainProcess._refuse_unresumed_optimizer_state(self, optimizer_state_file_path)
                optimizer_state_dict = None
                try:
                    print_acc(f"Loading optimizer state from {optimizer_state_file_path}")
                    optimizer_state_dict = torch.load(optimizer_state_file_path, weights_only=True,
                                                     map_location='cpu' if rawr_resume_enabled else None)
                except Exception as e:
                    if rawr_resume_enabled:
                        raise RuntimeError('Could not load RAWR optimizer/scheduler checkpoint; refusing a partial resume') from e
                    print_acc(f"Failed to load optimizer state from {optimizer_state_file_path}")
                    print_acc(e)
                if optimizer_state_dict is not None:
                    # checked before loading: torch maps saved state onto parameters by position only
                    identity = optimizer_state_dict.pop(checkpoint_identity.KEY, None)
                    ema_state = optimizer_state_dict.pop(checkpoint_identity.EMA_KEY, None)
                    BaseSDTrainProcess._check_optimizer_identity(self, identity, optimizer, optimizer_state_file_path)
                    try:
                        optimizer.load_state_dict(optimizer_state_dict)
                        if rawr_resume_enabled:
                            rawr_resume = rawr_checkpoint.capture(optimizer_state_dict)
                    except Exception as e:
                        if rawr_resume_enabled:
                            raise RuntimeError('Could not load RAWR optimizer/scheduler checkpoint; refusing a partial resume') from e
                        if identity is not None:
                            raise RuntimeError(f'Could not load {optimizer_state_file_path}, although it belongs to '
                                               f'this checkpoint and these parameters') from e
                        print_acc(f"Failed to load optimizer state from {optimizer_state_file_path}")
                        print_acc(e)
                    else:
                        self._resume_ema_state = ema_state
                    del optimizer_state_dict
                    flush()

            # update the optimizer LR from the params
            print_acc(f"Updating optimizer LR from params")
            if len(previous_lrs) > 0:
                for i, group in enumerate(optimizer.param_groups):
                    group['lr'] = previous_lrs[i]
                    group['initial_lr'] = previous_lrs[i]

            # Update the learning rates if they changed
            # optimizer.param_groups = previous_params

        if rawr_resume_enabled and self.step_num > 0 and rawr_resume is None:
            # e.g. a changed LoRA rank (network.did_change_weights) skipped the load above: RAWR would silently
            # restart its cycle and moments at this step (docs/STATE_MACHINES.md, SM-04)
            raise RuntimeError('RAWR resume requires the optimizer state paired with the selected model, but it was '
                               'not loaded (the network shape changed since that checkpoint); start a new run '
                               'for weights-only training')
        return rawr_resume

    def _refuse_unresumed_optimizer_state(self, path):
        """optimizer.pt with no checkpoint of this job resumed (a fresh start, or only a pretrained_lora_path) belongs
        to some earlier run: it used to be attached to the new weights, and the new saves then overwrote and pruned
        that run's checkpoints (docs/STATE_MACHINES.md, OPUS-RES-01/02)."""
        if (getattr(self, '_resume_record', None) is not None
                or getattr(self, '_rawr_paired_optimizer_path', None) is not None):
            return
        raise ValueError(f"{path} holds optimizer state from an earlier run, but no checkpoint of {self.job.name} "
                         f"was found to resume in {self.save_root}. Use a new output folder, or delete optimizer.pt "
                         f"(and that run's checkpoints) to start over in this one")

    def _check_optimizer_identity(self, identity, optimizer, path):
        """Refuse optimizer state that belongs to another save or other parameters, before it is loaded."""
        record = getattr(self, '_resume_record', None) or {'path': None, 'artifacts': []}
        checkpoint = record['path'] or 'the selected checkpoint'
        self._resume_identity_present = identity is not None
        if identity is None:
            if any(generation is not None for _, generation in record['artifacts']):
                raise ValueError(f"{path} has no save record (an older version wrote it), so it cannot belong to "
                                 f"{checkpoint}, which has one: that save was probably stopped before its optimizer "
                                 f"state was written. Move {path} out of the folder to continue with fresh optimizer "
                                 f"state, or restore the checkpoint it was saved with")
            print_acc(f"Legacy optimizer state: {path} does not record which checkpoint and parameters it belongs to, "
                      f"so pairing it with {checkpoint} cannot be verified. Resuming, but not certified as an exact "
                      f"continuation; the next save records it")
            return
        current = {'optimizer': checkpoint_identity.optimizer_class(optimizer),
                   'parameters': BaseSDTrainProcess._parameter_manifest(self, optimizer),
                   'base_model': checkpoint_identity.base_recipe(getattr(self, 'model_config', None))}
        start_step = getattr(getattr(self, 'train_config', None), 'start_step', None)
        checkpoint_identity.check(identity, current, record['artifacts'], self.step_num, start_step is None,
                                  checkpoint, path)
        # a matching save and parameter list, not a bit-exact continuation: save dtype, scheduler and accumulation
        # limits stay as warned elsewhere
        print_acc(f"Optimizer state is from the same save as {checkpoint} (step {identity['step']}) and matches "
                  f"this run's parameters")

    def _parameter_manifest(self, optimizer):
        manifest = getattr(self, '_resume_manifest', None)
        if manifest is None:
            manifest = self._resume_manifest = checkpoint_identity.parameter_manifest(
                optimizer, checkpoint_identity.trainer_parameter_names(self))
        return manifest

    def _restore_ema_resume_state(self):
        """After setup_ema: the resumed checkpoint holds EMA weights (the export). Put the raw training weights back
        into the parameters and the saved shadow into the EMA, so training continues exactly (OPUS-EMA-01)."""
        saved = getattr(self, '_resume_ema_state', None)
        self._resume_ema_state = None
        resumed = getattr(self, '_resume_record', None) is not None
        if saved is None:
            if not resumed:
                return
            if getattr(self, '_resume_identity_present', False):
                if self.ema is not None:
                    print_acc("EMA starts from the resumed weights (the checkpoint was saved without EMA)")
            elif self.ema is not None:
                print_acc("Legacy checkpoint: if it was trained with EMA, it holds the EMA weights, and training "
                          "continues from them with a fresh EMA (not an exact continuation)")
            return
        params = [p for group in self.optimizer.param_groups for p in group['params']]
        # the whole record first: a bad late entry must not leave some parameters restored (the identity check
        # already bound it to this save and to these parameters)
        checkpoint_identity.validate_ema_record(saved, params)
        if self.ema is not None:
            # changed EMA settings are the user's call, like a changed learning rate: continue under the new ones
            current = checkpoint_identity.ema_options(self.ema)
            changed = sorted(k for k in current if saved['options'].get(k) != current[k])
            if changed:
                was = saved['options']
                listed = ', '.join(f"{k}: {was.get(k)} -> {current[k]}" for k in changed)
                print_acc(f"EMA settings changed since this checkpoint ({listed}); the restored shadow continues "
                          f"under the new settings")
            num_updates = saved.get('num_updates') if current['use_num_updates'] else None
            if current['use_num_updates'] and num_updates is None:
                num_updates = 0
            self.ema.load_state_dict({'decay': self.ema.decay, 'num_updates': num_updates,
                                      'shadow_params': saved['shadow'], 'collected_params': None})
        with torch.no_grad():
            for p, value in zip(params, saved['raw']):
                p.data.copy_(value.to(device=p.device))
        if self.ema is not None:
            print_acc("Restored the raw training weights and the EMA shadow saved with this checkpoint")
        else:
            print_acc("Restored the raw training weights saved with this EMA checkpoint (EMA is now off; its shadow "
                      "is dropped)")

    def _check_accumulation_support(self, optimizer):
        """Refuse accumulation the optimizer setup cannot do (before training starts)."""
        gas = self.train_config.gradient_accumulation_steps
        if getattr(self.train_config, 'do_paramiter_swapping', False) and gas != 1:
            # swap_paramiters() reselects the trained parameters at the top of every loop step and clears their grads
            raise ValueError(
                f"do_paramiter_swapping reselects the trained parameters (clearing their gradients) every step, so it "
                f"cannot accumulate over gradient_accumulation_steps={gas}; use 1 (gradient_accumulation batches "
                f"within a step are fine)")
        if pending_grads.steps_in_backward(optimizer) and (gas != 1 or self.train_config.gradient_accumulation > 1):
            # each backward would apply its own update at 1/(window) scale instead of accumulating one
            raise ValueError(
                f"Optimizer '{self.train_config.optimizer}' updates parameters inside backward (fused), so it cannot "
                f"accumulate gradients over several backward passes (gradient_accumulation_steps={gas}, "
                f"gradient_accumulation={self.train_config.gradient_accumulation}). Use 1 for both, or an unfused "
                f"optimizer (e.g. automagic3 with optimizer_params fused: false).")

    def save(self, step=None):
        if not self.accelerator.is_main_process:
            return
        from training_extras import rawr_pair
        # first, as before: a refused save mode must not touch anything
        rawr_pair_save = rawr_pair.begin(self)
        # the EMA weights are installed only while saving, also when the save fails (docs/STATE_MACHINES.md, P-04)
        ema_was_training = self.ema is not None and self.ema._is_train_mode
        if self.ema is not None:
            # always save params as ema
            self.ema.eval()
        give_back = [self.ema.train] if ema_was_training else []
        try:
            # through the class: callers (and tests) may run save() unbound on a trainer-like namespace
            BaseSDTrainProcess._save_checkpoint(self, step, rawr_pair_save)
        except BaseException as primary:
            sampling_state.settle(give_back, primary, 'the raw weights after saving EMA weights')
            raise
        sampling_state.settle(give_back)
        flush()

    def _save_checkpoint(self, step, rawr_pair_save):
        from training_extras import rawr_pair
        rawr_model_alias = None
        rawr_model_export = None
        flush()

        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root, exist_ok=True)

        step_num = ''
        if step is not None:
            self.last_save_step = step
            # zeropad 9 digits
            step_num = f"_{str(step).zfill(9)}"

        self.update_training_metadata()
        # In-loop saves happen after the update but before step_num increments.
        # New RAWR checkpoints must resume at the next update, not replay one.
        from training_extras import rawr_checkpoint
        checkpoint_step = rawr_checkpoint.completed_step(
            self.step_num, getattr(self, '_checkpoint_completed_step', None),
            getattr(self, '_rawr_checkpoint_config', None))
        if checkpoint_step != self.step_num:
            self.meta['training_info']['step'] = checkpoint_step
        # Completed loop steps for every scheduler ('step' keeps its RAWR-checked meaning). Resume prefers this:
        # a non-RAWR periodic save records the index of the step it just trained, so resuming from 'step' repeats it.
        completed = getattr(self, '_checkpoint_completed_step', None)
        self.meta['training_info']['completed_steps'] = int(completed if completed is not None else self.step_num)
        filename = f'{self.job.name}{step_num}.safetensors'
        file_path = os.path.join(self.save_root, filename)

        save_meta = copy.deepcopy(self.meta)
        # get extra meta
        if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
            additional_save_meta = self.adapter.get_additional_save_metadata()
            if additional_save_meta is not None:
                for key, value in additional_save_meta.items():
                    save_meta[key] = value

        # prepare meta
        save_meta = get_meta_for_safetensors(save_meta, self.job.name)
        if rawr_pair_save is not None:
            save_meta['aitk_rawr_generation'] = json.dumps(rawr_pair_save['generation'])
        # every artifact of this save and its optimizer.pt carry one generation, so resume can tell they belong
        # together (docs/STATE_MACHINES.md, OPUS-RES-02)
        resume_generation = checkpoint_identity.new_generation()
        save_meta[checkpoint_identity.META_GENERATION] = json.dumps(resume_generation)
        unstamped = {}
        if not self.is_fine_tuning and not self.train_config.merge_network_on_save:
            if self.network is not None:
                lora_name = BaseSDTrainProcess._lora_save_name(self)

                filename = f'{lora_name}{step_num}.safetensors'
                file_path = os.path.join(self.save_root, filename)
                if rawr_pair_save is not None:
                    rawr_model_alias = file_path
                    file_path = rawr_pair_save['model']
                prev_multiplier = self.network.multiplier
                self.network.multiplier = 1.0
                try:
                    # if we are doing embedding training as well, add that
                    embedding_dict = self.embedding.state_dict() if self.embedding else None
                    self.network.save_weights(
                        file_path,
                        # Preserve training precision in the committed resume pair.
                        dtype=None if rawr_pair_save is not None else get_torch_dtype(self.save_config.dtype),
                        metadata=save_meta,
                        extra_state_dict=embedding_dict
                    )
                finally:
                    self.network.multiplier = prev_multiplier
                if rawr_pair_save is not None:
                    def rawr_model_export(path):
                        self.network.save_weights(
                            path, dtype=get_torch_dtype(self.save_config.dtype),
                            metadata=save_meta, extra_state_dict=embedding_dict)
                # if we have an embedding as well, pair it with the network

            # even if added to lora, still save the trigger version
            if self.embedding is not None:
                emb_filename = f'{self.embed_config.trigger}{step_num}.safetensors'
                emb_file_path = os.path.join(self.save_root, emb_filename)
                # for combo, above will get it
                # set current step: the completed count, which run() resumes from (see completed_steps above)
                self.embedding.step = self.meta['training_info']['completed_steps']
                # change filename to pt if that is set
                if self.embed_config.save_format == "pt":
                    # replace extension
                    emb_file_path = os.path.splitext(emb_file_path)[0] + ".pt"
                if os.path.basename(emb_file_path) == 'optimizer.pt':
                    # it would clobber the optimizer, and optimizer.pt may be a hard link to a committed RAWR slot
                    raise ValueError("An embedding trigger named 'optimizer' saved as .pt would overwrite "
                                     "optimizer.pt; rename the trigger")
                self.embedding.save(emb_file_path)
                # the embedding format has no room for the generation: optimizer.pt records its hash instead (small)
                unstamped[os.path.basename(emb_file_path)] = checkpoint_identity.sha256(emb_file_path)
            
            if self.decorator is not None:
                dec_filename = f'{BaseSDTrainProcess._decorator_save_name(self)}{step_num}.safetensors'
                dec_file_path = os.path.join(self.save_root, dec_filename)
                decorator_state_dict = self.decorator.state_dict()
                for key, value in decorator_state_dict.items():
                    if isinstance(value, torch.Tensor):
                        decorator_state_dict[key] = value.clone().to('cpu', dtype=get_torch_dtype(self.save_config.dtype))
                save_file(
                    decorator_state_dict,
                    dec_file_path,
                    metadata=save_meta,
                )

            if self.adapter is not None and self.adapter_config.train:
                adapter_name = BaseSDTrainProcess._adapter_save_name(self)

                filename = f'{adapter_name}{step_num}.safetensors'
                file_path = os.path.join(self.save_root, filename)
                # save adapter
                state_dict = self.adapter.state_dict()
                if self.adapter_config.type == 't2i':
                    save_t2i_from_diffusers(
                        state_dict,
                        output_file=file_path,
                        meta=save_meta,
                        dtype=get_torch_dtype(self.save_config.dtype)
                    )
                elif self.adapter_config.type == 'control_net':
                    # save in diffusers format
                    name_or_path = file_path.replace('.safetensors', '')
                    # move it to the new dtype and cpu
                    orig_device = self.adapter.device
                    orig_dtype = self.adapter.dtype

                    def adapter_back():
                        self.adapter = self.adapter.to(orig_device, dtype=orig_dtype)
                    self.adapter = self.adapter.to(torch.device('cpu'), dtype=get_torch_dtype(self.save_config.dtype))
                    try:
                        self.adapter.save_pretrained(
                            name_or_path,
                            dtype=get_torch_dtype(self.save_config.dtype),
                            safe_serialization=True
                        )
                        meta_path = os.path.join(name_or_path, 'aitk_meta.yaml')
                        with open(meta_path, 'w') as f:
                            yaml.dump({**self.meta, checkpoint_identity.META_GENERATION: resume_generation}, f)
                    except BaseException as primary:
                        sampling_state.settle([adapter_back], primary, 'the adapter placement')
                        raise
                    # move it back
                    adapter_back()
                else:
                    direct_save = False
                    if self.adapter_config.train_only_image_encoder:
                        direct_save = True
                    elif isinstance(self.adapter, CustomAdapter):
                        direct_save = self.adapter.do_direct_save
                    save_ip_adapter_from_diffusers(
                        state_dict,
                        output_file=file_path,
                        meta=save_meta,
                        dtype=get_torch_dtype(self.save_config.dtype),
                        direct_save=direct_save
                    )
        else:
            if self.network is not None and self.train_config.merge_network_on_save:
                # merge the network weights into a full model and save that.
                # torchao quantized weights can be force merged here (dequantize -> merge -> re-quantize)
                # even though can_merge_in is False (kept False so sampling never merges). quanto and
                # layer_offloading still cannot merge.
                from toolkit.util.quantize import get_torchao_config
                can_force_quantized_merge = (
                    self.model_config.quantize and not self.model_config.layer_offloading
                    and get_torchao_config(self.model_config.qtype) is not None
                )
                if not self.network.can_merge_in and not can_force_quantized_merge:
                    raise ValueError("Network cannot merge in weights. Cannot save full model.")

                print_acc("Merging network weights into full model for saving...")

                self.network.merge_in(merge_weight=self.train_config.merge_network_on_save_strength)
                # reset weights to zero
                self.network.reset_weights()
                self.network.is_merged_in = False
                
                print_acc("Done merging network weights. Saving model...")
                
            if self.save_config.save_format == "diffusers":
                # saving as a folder path
                file_path = file_path.replace('.safetensors', '')
                # convert it back to normal object
                save_meta = parse_metadata_from_safetensors(save_meta)

            if self.sd.refiner_unet and self.train_config.train_refiner:
                # save refiner
                refiner_name = self.job.name + '_refiner'
                filename = f'{refiner_name}{step_num}.safetensors'
                file_path = os.path.join(self.save_root, filename)
                self.sd.save_refiner(
                    file_path,
                    save_meta,
                    get_torch_dtype(self.save_config.dtype)
                )
            if self.train_config.train_unet or self.train_config.train_text_encoder:
                self.sd.save(
                    file_path,
                    save_meta,
                    get_torch_dtype(self.save_config.dtype)
                )

        # save learnable params as json if we have thim
        if self.snr_gos:
            json_data = {
                'offset_1': self.snr_gos.offset_1.item(),
                'offset_2': self.snr_gos.offset_2.item(),
                'scale': self.snr_gos.scale.item(),
                'gamma': self.snr_gos.gamma.item(),
            }
            path_to_save = file_path = os.path.join(self.save_root, 'learnable_snr.json')
            with open(path_to_save, 'w') as f:
                json.dump(json_data, f, indent=4)
        
        print_acc(f"Saved checkpoint to {file_path}")

        # save optimizer
        optimizer_failed = False
        if self.optimizer is not None:
            try:
                filename = f'optimizer.pt'
                file_path = os.path.join(self.save_root, filename)
                try:
                    state_dict = unwrap_model(self.optimizer).state_dict()
                except Exception as e:
                    state_dict = self.optimizer.state_dict()
                from training_extras import rawr_checkpoint
                state_dict = rawr_checkpoint.add_state(
                    state_dict, self.lr_scheduler,
                    getattr(self, '_rawr_checkpoint_config', None), checkpoint_step,
                    training_state=(rawr_checkpoint.training_state(self, checkpoint_step)
                                    if getattr(self, '_rawr_checkpoint_config', None) is not None else None))
                # which save, step, optimizer and parameters this state belongs to, checked before any resume loads it
                state_dict[checkpoint_identity.KEY] = checkpoint_identity.record(
                    self, resume_generation, self.meta['training_info']['completed_steps'], content=unstamped)
                # the checkpoint above holds the EMA weights (the export); the raw training weights and the shadow
                # travel here, in the same file and generation, so a resume continues training exactly (OPUS-EMA-01)
                ema_state = checkpoint_identity.ema_record(getattr(self, 'ema', None))
                if ema_state is not None:
                    state_dict[checkpoint_identity.EMA_KEY] = ema_state
                if rawr_pair_save is not None:
                    rawr_extras = None
                    if getattr(self, 'training_extras', None) is not None:
                        # The auxiliary loss state (EDM2 etc.) belongs to the same step as the pair; commit hashes it.
                        rawr_extras = rawr_pair.extras_path(rawr_pair_save)
                        self.training_extras.checkpoint(rawr_extras, curve_dir=self.save_root,
                                                        step=self.meta['training_info']['completed_steps'])
                    rawr_pair.commit(self.save_root, rawr_pair_save, state_dict,
                                     rawr_model_alias, lora_name, checkpoint_step, rawr_model_export,
                                     extras=rawr_extras)
                else:
                    # Write a new file and swap the name, never rewrite in place: an interrupted save keeps the
                    # previous optimizer.pt, and in a folder that once held RAWR pairs optimizer.pt may be a hard
                    # link to a committed slot (rawr_pair._hard_link), which an in-place write would corrupt.
                    rawr_checkpoint.atomic_save(state_dict, file_path)
                print_acc(f"Saved optimizer to {file_path}")
            except Exception as e:
                if getattr(self, '_rawr_checkpoint_config', None) is not None:
                    raise RuntimeError('Could not save RAWR optimizer/scheduler checkpoint') from e
                print_acc(e)
                print_acc("Could not save optimizer")
                optimizer_failed = True

        if getattr(self, 'training_extras', None) is not None and (rawr_pair_save is None or self.optimizer is None):
            # A committed pair already wrote (and aliased) its own copy above.
            # Stamped with completed_steps: the value a resume restores into step_num (checked on restore).
            self.training_extras.checkpoint(os.path.join(self.save_root, 'training_extras.pt'),
                                            step=self.meta['training_info']['completed_steps'])
        if optimizer_failed:
            # optimizer.pt still belongs to an earlier save; pruning could delete that save, the only one a resume can
            # pair with it (OPUS-RES-02). The next successful save prunes.
            print_acc("Keeping older saves: this save's optimizer state was not written")
        else:
            self.clean_up_saves()
        self.post_save_hook(file_path)

    # Called before the model is loaded
    def hook_before_model_load(self):
        # override in subclass
        pass

    def hook_after_model_load(self):
        # override in subclass
        pass

    def hook_add_extra_train_params(self, params):
        # override in subclass
        return params

    def hook_before_train_loop(self):
        if self.accelerator.is_main_process:
            self.logger.start()
        self.prepare_accelerator()
        
    def sample_step_hook(self, img_num, total_imgs):
        pass
    
    def prepare_optimizer_without_state_placement(self, optimizer):
        # accelerate's device placement round-trips the whole optimizer state through the accelerator device: on an
        # H3 resume that copied 4.6 GiB of restored moments onto the GPU before CPU-offloading optimizers moved them
        # back. Per-param state needs no help (torch's load_state_dict casts tensors, and dicts of them, to each
        # param's device; optimizers with custom state objects place them in their own load_state_dict). Tensor
        # hyperparameters in param_groups (compass, schedulefree) are not cast by torch, so they still move here
        # as before. DeepSpeed and Megatron-LM refuse a custom placement and keep accelerate's.
        if self.accelerator.distributed_type in (DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM):
            return self.accelerator.prepare(optimizer)
        for group in optimizer.param_groups:
            for key, value in group.items():
                if key != 'params':
                    group[key] = move_to_device(value, self.accelerator.device)
        return self.accelerator.prepare(optimizer, device_placement=[False])

    def prepare_accelerator(self):
        # set some config
        self.accelerator.even_batches=False
        
        # # prepare all the models stuff for accelerator (hopefully we dont miss any)
        if self.sd.vae is not None:
            self.sd.vae = self.accelerator.prepare(self.sd.vae)
        if self.sd.unet is not None:
            self.sd.unet = self.accelerator.prepare(self.sd.unet)
            # todo always tdo it?
            self.modules_being_trained.append(self.sd.unet)
        if self.sd.text_encoder is not None and self.train_config.train_text_encoder:
            if isinstance(self.sd.text_encoder, list):
                self.sd.text_encoder = [self.accelerator.prepare(model) for model in self.sd.text_encoder]
                self.modules_being_trained.extend(self.sd.text_encoder)
            else:
                self.sd.text_encoder = self.accelerator.prepare(self.sd.text_encoder)
                self.modules_being_trained.append(self.sd.text_encoder)
        if self.sd.refiner_unet is not None and self.train_config.train_refiner:
            self.sd.refiner_unet = self.accelerator.prepare(self.sd.refiner_unet)
            self.modules_being_trained.append(self.sd.refiner_unet)
        # todo, do we need to do the network or will "unet" get it?
        if self.sd.network is not None:
            self.sd.network = self.accelerator.prepare(self.sd.network)
            self.modules_being_trained.append(self.sd.network)
        if self.adapter is not None and self.adapter_config.train:
            # todo adapters may not be a module. need to check
            self.adapter = self.accelerator.prepare(self.adapter)
            self.modules_being_trained.append(self.adapter)
        
        # prepare other things
        self.optimizer = self.prepare_optimizer_without_state_placement(self.optimizer)
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        # self.data_loader = self.accelerator.prepare(self.data_loader)
        # if self.data_loader_reg is not None:
        #     self.data_loader_reg = self.accelerator.prepare(self.data_loader_reg)
            

    def ensure_params_requires_grad(self, force=False):
        if self.train_config.do_paramiter_swapping and not force:
            # the optimizer will handle this if we are not forcing
            return
        for group in self.params:
            for param in group['params']:
                if isinstance(param, torch.nn.Parameter):  # Ensure it's a proper parameter
                    param.requires_grad_(True)

    def setup_ema(self):
        if self.train_config.ema_config.use_ema:
            # our params are in groups. We need them as a single iterable
            params = []
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    params.append(param)
            self.ema = ExponentialMovingAverage(
                params,
                decay=self.train_config.ema_config.ema_decay,
                use_feedback=self.train_config.ema_config.use_feedback,
                feedback_rate=self.train_config.ema_config.feedback_rate,
                param_multiplier=self.train_config.ema_config.param_multiplier,
            )
            # expose to the model: models that run an EMA-teacher forward during training
            # (e.g. wan21_pixel self_flow) read it from here
            self.sd.ema = self.ema

    def before_dataset_load(self):
        pass

    def get_params(self):
        # you can extend this in subclass to get params
        # otherwise params will be gathered through normal means
        return None

    def hook_train_loop(self, batch):
        # return loss
        return 0.0
    
    def hook_after_sd_init_before_load(self):
        pass

    def get_latest_save_path(self, name=None, post='', include_pretrained_lora=True, suffixes=('',)):
        if name == None:
            name = self.job.name
        from training_extras import rawr_checkpoint, rawr_pair
        if rawr_checkpoint.is_rawr_name(getattr(self.train_config, 'lr_scheduler', None)) and not post:
            selected = rawr_pair.select(self.save_root, name)
            if selected is not None:
                if self.network_config is None or self.train_config.merge_network_on_save:
                    raise ValueError('Committed RAWR pair is a LoRA; use its original training configuration')
                self._rawr_paired_optimizer_path = selected['optimizer']
                self._rawr_paired_extras_path = selected.get('training_extras')
                self._rawr_pair_generation = selected['generation']
                return selected['model']
        # get latest saved step: the most recently written complete save of exactly this name (RAWR pairs were
        # selected above). find_save_sets also drops the _LoRA/_refiner/_t2i/_cn siblings and derived exports.
        latest_path = None
        sets = find_save_sets(self.save_root, name, suffixes, post)
        complete = [s for s in sets if s.complete]
        if sets and not complete:
            raise ValueError(f"{self.save_root} holds saves of {name}, but none is complete (e.g. "
                             f"{sets[-1].path} lacks its {', '.join(sets[-1].missing())} file). Restore the missing "
                             f"files or use a new output folder")
        if complete:
            chosen = BaseSDTrainProcess._committed_save(self, complete)
            for save in sets:
                if not save.complete and save.rank > chosen.rank:
                    print_acc(f"Skipping the incomplete save {save.path} (missing {', '.join(save.missing())}; "
                              f"its save was interrupted)")
            latest_path = chosen.path
            # the other files of a split save load with this path; the resume binds them to the optimizer state too
            files = getattr(self, '_latest_save_files', None) or {}
            files[latest_path] = list(chosen.files.values())
            self._latest_save_files = files

        if include_pretrained_lora and latest_path is None and self.network_config is not None and self.network_config.pretrained_lora_path is not None:
            # set pretrained lora path as load path if we do not have a checkpoint to resume from
            if os.path.exists(self.network_config.pretrained_lora_path):
                latest_path = self.network_config.pretrained_lora_path
                print_acc(f"Using pretrained lora path from config: {latest_path}")
            else:
                # no pretrained lora found
                print_acc(f"Pretrained lora path from config does not exist: {self.network_config.pretrained_lora_path}")

        return latest_path

    def _committed_save(self, complete):
        """The newest complete save, unless optimizer.pt was written by an earlier one. A save stopped after its
        checkpoint and before its optimizer state (a UI Stop during "Saving model") then resumes exactly from the last
        save whose optimizer state was written, instead of pairing the new weights with older moments (OPUS-RES-02)."""
        newest = complete[-1]
        optimizer_path = os.path.join(self.save_root, 'optimizer.pt')
        try:
            stat = os.stat(optimizer_path)
        except OSError:
            return newest
        cache = getattr(self, '_optimizer_identity_cache', None)
        key = (stat.st_mtime_ns, stat.st_size)
        if cache is None or cache[0] != key:
            cache = (key, checkpoint_identity.read_optimizer_identity(optimizer_path))
            self._optimizer_identity_cache = cache
        committed = cache[1]

        def belongs(save):  # every file of the save (both halves of a split LoRA)
            return all(checkpoint_identity.belongs(committed, p, checkpoint_identity.read_checkpoint_generation(p))
                       for p in save.files.values())
        if committed is None or belongs(newest):
            return newest
        for save in reversed(complete[:-1]):
            if belongs(save):
                print_acc(f"The newest save {newest.path} has no optimizer state (its save was interrupted after the "
                          f"checkpoint was written); resuming from {save.path}, the last complete save")
                return save
        return newest  # nothing matches: the optimizer check refuses with what to do

    def _note_resumed(self, path, meta=None):
        """Record a checkpoint this run resumed from, and its save generation, for the optimizer-state check."""
        record = getattr(self, '_resume_record', None)
        if record is None:
            record = self._resume_record = {'path': path, 'artifacts': []}
        generation = (meta or {}).get(checkpoint_identity.META_GENERATION)
        record['artifacts'].append((path, generation if isinstance(generation, str) else None))

    def load_training_state_from_metadata(self, path):
        if not self.accelerator.is_main_process:
            return
        if path is not None and self.network_config is not None and path == self.network_config.pretrained_lora_path:
            # dont load metadata from pretrained lora
            return
        meta = None
        # if path is folder, then it is diffusers
        if os.path.isdir(path):
            meta_path = os.path.join(path, 'aitk_meta.yaml')
            # load it
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
        else:
            meta = load_metadata_from_safetensors(path)
        if meta is not None and meta.get('aitk_rawr_generation') is not None:
            if meta['aitk_rawr_generation'] != getattr(self, '_rawr_pair_generation', None):
                raise ValueError('RAWR model requires its committed-pair manifest; refusing an unpaired export')
        training_info = (meta or {}).get('training_info')
        saved_reference = training_info.get('dpo_reference_lora') if isinstance(training_info, dict) else None
        if saved_reference != getattr(self, '_dpo_reference_identity', None):
            raise ValueError('Frozen DPO LoRA or strength differs from this checkpoint. '
                             'Restore the original reference to resume, or use a new job name.')
        BaseSDTrainProcess._note_resumed(self, path, meta)
        for sibling in (getattr(self, '_latest_save_files', None) or {}).get(path, []):
            if sibling != path:
                BaseSDTrainProcess._note_resumed(self, sibling, load_metadata_from_safetensors(sibling))
        # if 'training_info' in Orderdict keys
        if meta is not None and 'training_info' in meta and 'step' in meta['training_info'] and self.train_config.start_step is None:
            self.step_num = meta['training_info'].get('completed_steps', meta['training_info']['step'])
            if 'epoch' in meta['training_info']:
                self.epoch_num = meta['training_info']['epoch']
            self._clean_share_saved = meta['training_info'].get('clean_share_curriculum')
            self._saved_feedback_schedule = meta['training_info'].get('dpo_feedback')
            self.start_step = self.step_num
            print_acc(f"Found step {self.step_num} in metadata, starting from there")

    def _check_model_lora_ignores_on_resume(self, path, extra_weights):
        # A run saved before its model declared lora_ignore_if_contains (H3 adaln_proj) holds LoRA weights, and
        # optimizer/RAWR state, for modules the network now skips. The non-strict load drops those weights silently
        # and the optimizer load then fails on the group size (RAWR: "refusing a partial resume"; otherwise a silent
        # fresh optimizer). Say so up front. A pretrained_lora_path is a fresh start, so it only drops them.
        from toolkit.network_mixins import saved_keys_for_model_ignores
        hit = saved_keys_for_model_ignores(list(extra_weights or {}), getattr(self.network, 'model_lora_ignores', None))
        if not hit:
            return
        if self.network_config.pretrained_lora_path is not None and path == self.network_config.pretrained_lora_path:
            print_acc(f"Pretrained LoRA has {len(hit)} weights for modules this model skips "
                      f"({self.network.model_lora_ignores}); they are not loaded")
            return
        raise ValueError(
            f"{path} was trained with LoRA on modules this model now skips ({self.network.model_lora_ignores}, "
            f"{len(hit)} saved weights, e.g. {hit[0]}). To resume it as trained, set network.network_kwargs."
            f"use_model_lora_ignores: false; otherwise start a new output folder")

    def _check_resumed_network_keys(self, path):
        """Resume loads the network strictly: every trainable weight must come from the checkpoint and every saved
        weight must have a module. A same-size change of the trained modules (only_if_contains or ignore_if_contains
        moved to a sibling block) otherwise left the new modules at initialization, dropped the saved ones, and
        attached their optimizer moments by position (docs/STATE_MACHINES.md, OPUS-PARAM-ID-01). A
        pretrained_lora_path is the deliberate partial, weights-only load and is not checked."""
        if self.network_config.pretrained_lora_path is not None and path == self.network_config.pretrained_lora_path:
            return
        report = getattr(self.network, 'last_load_report', None)
        if report is None:
            return
        # a LoRA trained with an embedding carries the embedding's vectors; they load from the embedding file
        embedded = {'emb_params', 'clip_l', 'clip_g'} if self.embed_config is not None else set()
        not_loaded = report['not_loaded']
        unexpected = [k for k in report['unexpected'] if k not in embedded]
        if not not_loaded and not unexpected:
            return
        details = []
        if not_loaded:
            details.append(f"{len(not_loaded)} trainable weights are not in it (e.g. {not_loaded[0]})")
        if unexpected:
            details.append(f"{len(unexpected)} of its weights have no module in this network (e.g. {unexpected[0]})")
        raise ValueError(
            f"{path} was saved for a different network: {'; '.join(details)}. The network's target modules, type or "
            f"model changed since it was saved, so resuming would train the new modules from initialization under "
            f"the saved optimizer state. Resume with the original network settings; or, to start from its matching "
            f"weights only, use a new output folder with network.pretrained_lora_path: {path}")

    def load_weights(self, path):
        if self.network is not None:
            extra_weights = self.network.load_weights(path)
            self.load_training_state_from_metadata(path)
            return extra_weights
        else:
            print_acc("load_weights not implemented for non-network models")
            return None

    def apply_snr(self, seperated_loss, timesteps):
        if self.train_config.learnable_snr_gos:
            # add snr_gamma
            seperated_loss = apply_learnable_snr_gos(seperated_loss, timesteps, self.snr_gos)
        elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001:
            # add snr_gamma
            seperated_loss = apply_snr_weight(seperated_loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma, fixed=True)
        elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
            # add min_snr_gamma
            seperated_loss = apply_snr_weight(seperated_loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        return seperated_loss

    def load_lorm(self):
        latest_save_path = self.get_latest_save_path()
        if latest_save_path is not None:
            # hacky way to reload weights for now
            # todo, do this
            state_dict = load_file(latest_save_path, device=self.device)
            self.sd.unet.load_state_dict(state_dict)

            meta = load_metadata_from_safetensors(latest_save_path)
            # if 'training_info' in Orderdict keys
            if 'training_info' in meta and 'step' in meta['training_info']:
                self.step_num = meta['training_info'].get('completed_steps', meta['training_info']['step'])
                if 'epoch' in meta['training_info']:
                    self.epoch_num = meta['training_info']['epoch']
                self._clean_share_saved = meta['training_info'].get('clean_share_curriculum')
                self._saved_feedback_schedule = meta['training_info'].get('dpo_feedback')
                self.start_step = self.step_num
                print_acc(f"Found step {self.step_num} in metadata, starting from there")

    # def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
    #     self.sd.noise_scheduler.set_timesteps(1000, device=self.device_torch)
    #     sigmas = self.sd.noise_scheduler.sigmas.to(device=self.device_torch, dtype=dtype)
    #     schedule_timesteps = self.sd.noise_scheduler.timesteps.to(self.device_torch, )
    #     timesteps = timesteps.to(self.device_torch, )
    #
    #     # step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    #     step_indices = [t for t in timesteps]
    #
    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma

    def load_additional_training_modules(self, params):
        # override in subclass
        return params

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.sd.noise_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.sd.noise_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def get_optimal_noise(self, latents, dtype=torch.float32):
        # same distribution as keeping the best aligned of N noises, from one draw
        num_candidates = self.train_config.optimal_noise_pairing_samples
        noise = torch.randn_like(latents, dtype=dtype)
        # one pick per channel
        group_dims = tuple(range(2, latents.dim()))
        float_latents = latents.float()
        # clamp so an all zero latent gets plain noise instead of nan
        latent_direction = float_latents / torch.linalg.vector_norm(float_latents, dim=group_dims, keepdim=True).clamp_min(1e-8)
        current_alignment = (noise.float() * latent_direction).sum(dim=group_dims, keepdim=True)
        picked_alignment = torch.randn(*current_alignment.shape, abs(num_candidates), device=latents.device).amax(dim=-1)
        # negative count keeps the worst aligned instead
        if num_candidates < 0:
            picked_alignment = -picked_alignment
        correction = (picked_alignment - current_alignment) * latent_direction
        return noise + correction.to(dtype)
    
    def get_consistent_noise(self, latents, batch: 'DataLoaderBatchDTO', dtype=torch.float32):
        batch_num = latents.shape[0]
        chunks = torch.chunk(latents, batch_num, dim=0)
        noise_chunks = []
        for idx, chunk in enumerate(chunks):
            # get seed from path
            file_item = batch.file_items[idx]
            img_path = file_item.path
            # add augmentors
            if file_item.flip_x:
                img_path += '_fx'
            if file_item.flip_y:
                img_path += '_fy'
            seed = int(hashlib.md5(img_path.encode()).hexdigest(), 16) & 0xffffffff
            generator = torch.Generator("cpu").manual_seed(seed)
            noise_chunk = torch.randn(chunk.shape, generator=generator).to(chunk.device, dtype=dtype)
            noise_chunks.append(noise_chunk)
        noise = torch.cat(noise_chunks, dim=0).to(dtype=dtype)
        return noise
            

    def get_noise(
        self, 
        latents, 
        batch_size, 
        dtype=torch.float32, 
        batch: 'DataLoaderBatchDTO' = None,
        timestep=None,
    ):
        if abs(self.train_config.optimal_noise_pairing_samples) > 1:
            noise = self.get_optimal_noise(latents, dtype=dtype)
        elif self.train_config.force_consistent_noise:
            if batch is None:
                raise ValueError("Batch must be provided for consistent noise")
            noise = self.get_consistent_noise(latents, batch, dtype=dtype)
        else:
            if hasattr(self.sd, 'get_latent_noise_from_latents'):
                noise = self.sd.get_latent_noise_from_latents(
                    latents,
                    noise_offset=self.train_config.noise_offset
                ).to(self.device_torch, dtype=dtype)
            else:
                # get noise
                noise = self.sd.get_latent_noise(
                    height=latents.shape[2],
                    width=latents.shape[3],
                    num_channels=latents.shape[1],
                    batch_size=batch_size,
                    noise_offset=self.train_config.noise_offset,
                ).to(self.device_torch, dtype=dtype)
        
        if self.train_config.blended_blur_noise:
            noise = get_blended_blur_noise(
                latents, noise, timestep
            )

        if self.train_config.multires_noise_iterations and not getattr(self, '_disable_noise_augmentation', False):
            from toolkit.noise_augmentation import augmentation_seed, multires_noise_like
            noise = multires_noise_like(
                noise, self.train_config.multires_noise_iterations,
                self.train_config.multires_noise_discount,
                seed=augmentation_seed(self.train_config.noise_augmentation_seed,
                                       self.step_num, self.grad_accumulation_step, 1,
                                       batch_index=getattr(self, '_noise_augmentation_batch_index', 0)),
            )

        return noise

    def process_general_training_batch(self, batch: 'DataLoaderBatchDTO', *, shared_pair_noise=False):
        if (self.train_config.multires_noise_iterations or self.train_config.ip_noise_gamma) and not getattr(self, '_disable_noise_augmentation', False):
            from toolkit.noise_augmentation import begin_noise_augmentation_batch
            begin_noise_augmentation_batch(self)
        with torch.no_grad():
            with self.timer('prepare_prompt'):
                prompts = batch.get_caption_list()
                is_reg_list = batch.get_is_reg_list()

                is_any_reg = any([is_reg for is_reg in is_reg_list])

                do_double = self.train_config.short_and_long_captions and not is_any_reg

                if self.train_config.short_and_long_captions and do_double:
                    # dont do this with regs. No point

                    # double batch and add short captions to the end
                    prompts = prompts + batch.get_caption_short_list()
                    is_reg_list = is_reg_list + is_reg_list
                if self.model_config.refiner_name_or_path is not None and self.train_config.train_unet:
                    prompts = prompts + prompts
                    is_reg_list = is_reg_list + is_reg_list

                conditioned_prompts = []

                for prompt, is_reg in zip(prompts, is_reg_list):

                    # make sure the embedding is in the prompts
                    if self.embedding is not None:
                        prompt = self.embedding.inject_embedding_to_prompt(
                            prompt,
                            expand_token=True,
                            add_if_not_present=not is_reg,
                        )

                    if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                        prompt = self.adapter.inject_trigger_into_prompt(
                            prompt,
                            expand_token=True,
                            add_if_not_present=not is_reg,
                        )

                    # make sure trigger is in the prompts if not a regularization run
                    if self.trigger_word is not None:
                        prompt = self.sd.inject_trigger_into_prompt(
                            prompt,
                            trigger=self.trigger_word,
                            add_if_not_present=not is_reg,
                        )

                    if not is_reg and self.train_config.prompt_saturation_chance > 0.0:
                        # do random prompt saturation by expanding the prompt to hit at least 77 tokens
                        if random.random() < self.train_config.prompt_saturation_chance:
                            est_num_tokens = len(prompt.split(' '))
                            if est_num_tokens < 77:
                                num_repeats = int(77 / est_num_tokens) + 1
                                prompt = ', '.join([prompt] * num_repeats)


                    conditioned_prompts.append(prompt)

            with self.timer('prepare_latents'):
                dtype = get_torch_dtype(self.train_config.dtype)
                imgs = None
                is_reg = any(batch.get_is_reg_list())
                if batch.tensor is not None:
                    imgs = batch.tensor
                    # waveforms stay fp32 into the audio encoder
                    imgs = imgs.to(self.device_torch, dtype=torch.float32 if getattr(self.sd, 'is_audio_model', False) else dtype)
                    # dont adjust for regs.
                    if self.train_config.img_multiplier is not None and not is_reg:
                        # do it ad contrast
                        imgs = reduce_contrast(imgs, self.train_config.img_multiplier)
                if batch.latents is not None:
                    latents = batch.latents.to(self.device_torch, dtype=dtype)
                    batch.latents = latents
                else:
                    # normalize to
                    if self.train_config.standardize_images:
                        if self.sd.is_xl or self.sd.is_vega or self.sd.is_ssd:
                            target_mean_list = [0.0002, -0.1034, -0.1879]
                            target_std_list = [0.5436, 0.5116, 0.5033]
                        else:
                            target_mean_list = [-0.0739, -0.1597, -0.2380]
                            target_std_list = [0.5623, 0.5295, 0.5347]
                        # Mean: tensor([-0.0739, -0.1597, -0.2380])
                        # Standard Deviation: tensor([0.5623, 0.5295, 0.5347])
                        imgs_channel_mean = imgs.mean(dim=(2, 3), keepdim=True)
                        imgs_channel_std = imgs.std(dim=(2, 3), keepdim=True)
                        imgs = (imgs - imgs_channel_mean) / imgs_channel_std
                        target_mean = torch.tensor(target_mean_list, device=self.device_torch, dtype=dtype)
                        target_std = torch.tensor(target_std_list, device=self.device_torch, dtype=dtype)
                        # expand them to match dim
                        target_mean = target_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                        target_std = target_std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                        imgs = imgs * target_std + target_mean
                        batch.tensor = imgs

                        # show_tensors(imgs, 'imgs')

                    latents = self.sd.encode_images(imgs)
                    batch.latents = latents

                if batch.unconditional_tensor is not None and batch.unconditional_latents is None:
                    unconditional_imgs = batch.unconditional_tensor
                    unconditional_imgs = unconditional_imgs.to(self.device_torch, dtype=dtype)
                    unconditional_latents = self.sd.encode_images(unconditional_imgs)
                    batch.unconditional_latents = unconditional_latents

                unaugmented_latents = None
                if self.train_config.loss_target == 'differential_noise':
                    # we determine noise from the differential of the latents
                    unaugmented_latents = self.sd.encode_images(batch.unaugmented_tensor)

            with self.timer('prepare_scheduler'):
                
                batch_size = len(batch.file_items)
                min_noise_steps = self.train_config.min_denoising_steps
                max_noise_steps = self.train_config.max_denoising_steps
                if self.model_config.refiner_name_or_path is not None:
                    # if we are not training the unet, then we are only doing refiner and do not need to double up
                    if self.train_config.train_unet:
                        max_noise_steps = round(self.train_config.max_denoising_steps * self.model_config.refiner_start_at)
                        do_double = True
                    else:
                        min_noise_steps = round(self.train_config.max_denoising_steps * self.model_config.refiner_start_at)
                        do_double = False

                num_train_timesteps = self.train_config.num_train_timesteps

                if self.train_config.noise_scheduler in ['custom_lcm']:
                    # we store this value on our custom one
                    self.sd.noise_scheduler.set_timesteps(
                        self.sd.noise_scheduler.train_timesteps, device=self.device_torch
                    )
                elif self.train_config.noise_scheduler in ['lcm']:
                    self.sd.noise_scheduler.set_timesteps(
                        num_train_timesteps, device=self.device_torch, original_inference_steps=num_train_timesteps
                    )
                elif self.train_config.noise_scheduler == 'flowmatch':
                    linear_timesteps = any([
                        self.train_config.linear_timesteps,
                        self.train_config.linear_timesteps2,
                        self.train_config.timestep_type == 'linear',
                        self.train_config.timestep_type in ['one_step', 'two_step', 'four_step', 'eight_step'],
                    ])
                    
                    timestep_type = 'linear' if linear_timesteps else None
                    if timestep_type is None:
                        timestep_type = self.train_config.timestep_type
                    
                    if self.train_config.timestep_type == 'next_sample':
                        # simulate a sample
                        num_train_timesteps = self.train_config.next_sample_timesteps
                        timestep_type = 'shift'
                    
                    patch_size = 1
                    if self.sd.is_flux or 'flex' in self.sd.arch:
                        # flux is a patch size of 1, but latents are divided by 2, so we need to double it
                        patch_size = 2
                    elif hasattr(self.sd.unet, 'config') and hasattr(self.sd.unet.config, 'patch_size'):
                        patch_size = self.sd.unet.config.patch_size
                    
                    flow_kwargs = self.train_config.flow_timestep_kwargs(timestep_type)
                    if (self.train_config.logit_normal_clean_schedule is not None
                            and not getattr(batch, '_clean_share_validation', False)):
                        from toolkit.clean_share_curriculum import scheduled_mean
                        epoch = getattr(batch, '_clean_share_epoch', None)
                        if epoch is None:
                            raise ValueError('clean-share curriculum batch is missing epoch provenance')
                        flow_kwargs['logit_mean'] = scheduled_mean(
                            self.train_config.logit_normal_clean_schedule, epoch,
                            flow_kwargs['logit_std'], flow_kwargs['cutoff'], flow_kwargs['uniform_mix'])
                    self.sd.noise_scheduler.set_train_timesteps(
                        num_train_timesteps,
                        device=self.device_torch,
                        timestep_type=timestep_type,
                        latents=latents,
                        patch_size=patch_size,
                        **flow_kwargs,
                    )
                else:
                    self.sd.noise_scheduler.set_timesteps(
                        num_train_timesteps, device=self.device_torch
                    )
            if self.sd.is_multistage:
                with self.timer('adjust_multistage_timesteps'):
                    # get our current sample range
                    boundaries = [1] + self.sd.multistage_boundaries
                    boundary_max, boundary_min = boundaries[self.current_boundary_index], boundaries[self.current_boundary_index + 1]
                    asc_timesteps = torch.flip(self.sd.noise_scheduler.timesteps, dims=[0])
                    lo = len(asc_timesteps) - torch.searchsorted(asc_timesteps, torch.tensor(boundary_max * 1000, device=asc_timesteps.device), right=False)
                    hi = len(asc_timesteps) - torch.searchsorted(asc_timesteps, torch.tensor(boundary_min * 1000, device=asc_timesteps.device), right=True)
                    first_idx = (lo - 1).item() if hi > lo else 0
                    last_idx  = (hi - 1).item() if hi > lo else 999
                    min_noise_steps = first_idx
                    max_noise_steps = last_idx

            # clip min max indicies
            min_noise_steps = max(min_noise_steps, 0)
            max_noise_steps = min(max_noise_steps, last_train_index(num_train_timesteps, self.sd.noise_scheduler))
            
                    
            with self.timer('prepare_timesteps_indices'):

                content_or_style = self.train_config.content_or_style
                if is_reg:
                    content_or_style = self.train_config.content_or_style_reg

                if getattr(self.sd.noise_scheduler, 'fixed_train_sigmas', None) is not None:
                    # the scheduler's timesteps are the model's own sampling schedule
                    timestep_indices = torch.randint(
                        0,
                        len(self.sd.noise_scheduler.timesteps),
                        (batch_size,),
                        device=self.device_torch
                    )
                    timestep_indices = timestep_indices.long()
                elif self.train_config.timestep_type in ['two_step', 'four_step', 'eight_step']:
                    if self.train_config.timestep_type == 'two_step':
                        indice_choices = [0, 499]
                    elif self.train_config.timestep_type == 'four_step':
                        indice_choices = [0, 250, 500, 750]
                    elif self.train_config.timestep_type == 'eight_step':
                        indice_choices = [0, 125, 250, 375, 500, 625, 750, 875]
                    timestep_indices = torch.tensor(random.choices(indice_choices, k=batch_size), device=self.device_torch)
                    timestep_indices = timestep_indices.long()
                elif self.train_config.timestep_type == 'next_sample':
                    timestep_indices = torch.randint(
                            0,
                            num_train_timesteps - 2, # -1 for 0 idx, -1 so we can step
                            (batch_size,),
                            device=self.device_torch
                        )
                    timestep_indices = timestep_indices.long()
                elif self.train_config.timestep_type == 'one_step':
                    timestep_indices = torch.zeros((batch_size,), device=self.device_torch, dtype=torch.long)
                elif content_or_style in ['style', 'content']:
                    # this is from diffusers training code
                    # Cubic sampling for favoring later or earlier timesteps
                    # For more details about why cubic sampling is used for content / structure,
                    # refer to section 3.4 of https://arxiv.org/abs/2302.08453

                    # for content / structure, it is best to favor earlier timesteps
                    # for style, it is best to favor later timesteps

                    orig_timesteps = torch.rand((batch_size,), device=latents.device)

                    if content_or_style == 'content':
                        timestep_indices = orig_timesteps ** 3 * self.train_config.num_train_timesteps
                    elif content_or_style == 'style':
                        timestep_indices = (1 - orig_timesteps ** 3) * self.train_config.num_train_timesteps

                    timestep_indices = value_map(
                        timestep_indices,
                        0,
                        self.train_config.num_train_timesteps - 1,
                        min_noise_steps,
                        max_noise_steps
                    )
                    timestep_indices = timestep_indices.long().clamp(
                        min_noise_steps,
                        max_noise_steps
                    )
                    
                elif content_or_style == 'balanced':
                    # flowmatch: [min, max] inclusive, so the grid's last (lowest-sigma) index can be drawn
                    timestep_indices = balanced_timestep_indices(
                        min_noise_steps,
                        max_noise_steps,
                        batch_size,
                        flowmatch=self.train_config.noise_scheduler == 'flowmatch',
                        device=self.device_torch
                    )
                    timestep_indices = timestep_indices.long()
                else:
                    raise ValueError(f"Unknown content_or_style {content_or_style}")

                if self.train_config.first_timestep_chance > 0.0:
                    # index 0 is full noise; per-sample chance to force it
                    force_first = torch.rand((batch_size,), device=timestep_indices.device) < self.train_config.first_timestep_chance
                    timestep_indices = torch.where(force_first, torch.zeros_like(timestep_indices), timestep_indices)
            with self.timer('convert_timestep_indices_to_timesteps'):
                # convert the timestep_indices to a timestep
                timesteps = self.sd.noise_scheduler.timesteps[timestep_indices.long()]
                
            with self.timer('prepare_noise'):
                # get noise
                # A preference batch has B captions and 2B latents, ordered chosen then rejected.
                noise_latents = latents[:batch_size] if shared_pair_noise else latents
                noise = self.get_noise(noise_latents, batch_size, dtype=dtype, batch=batch, timestep=timesteps)

                # add dynamic noise offset. Dynamic noise is offsetting the noise to the same channelwise mean as the latents
                # this will negate any noise offsets
                if self.train_config.dynamic_noise_offset and not is_reg:
                    latents_channel_mean = latents.mean(dim=(2, 3), keepdim=True) / 2
                    # subtract channel mean to that we compensate for the mean of the latents on the noise offset per channel
                    noise = noise + latents_channel_mean

                if self.train_config.loss_target == 'differential_noise':
                    differential = latents - unaugmented_latents
                    # add noise to differential
                    # noise = noise + differential
                    noise = noise + (differential * 0.5)
                    # noise = value_map(differential, 0, torch.abs(differential).max(), 0, torch.abs(noise).max())
                    latents = unaugmented_latents

                noise_multiplier = self.train_config.noise_multiplier
                
                s = (noise.shape[0], noise.shape[1], 1, 1)
                if len(noise.shape) == 5:
                    # if we have a 5d tensor, then we need to do it on a per batch item, per channel basis, per frame
                    s = (noise.shape[0], noise.shape[1], noise.shape[2], 1, 1)
                
                noise = noise * noise_multiplier
                
                if self.train_config.do_signal_correction_noise:
                    # inverted copy of the latent, tapered to zero at pure noise so the input never inverts
                    scn_scale = self.train_config.signal_correction_noise_scale
                    scn_t = (timesteps.to(noise.device, dtype=torch.float32) / 1000.0).view(-1, *([1] * (noise.dim() - 1)))
                    scn_strength = scn_scale * (1.0 - scn_t) / (1.0 - scn_t + scn_scale * scn_t)
                    noise = noise - latents.to(noise.device, dtype=noise.dtype) * scn_strength.to(noise.dtype)
                
                if self.train_config.do_batch_noise_correction:
                    if latents.shape[0] == 1:
                        # if we only have a batch size of 1, then we cant do batch noise correction, so we skip it
                        print_acc("Skipping batch noise correction because batch size is 1, increase batch size and num_repeats to use this feature")
                    else:
                        # shuffle tensors ensuring that no tensor is in the same position as before
                        batch_noise = latents.clone().roll(shifts=torch.randint(1, latents.shape[0], (1,)).item(), dims=0).to(noise.device, dtype=noise.dtype)
                        batch_noise_scale = torch.randn(
                            batch_noise.shape[0], batch_noise.shape[1], 1, 1,
                            device=batch_noise.device,
                            dtype=batch_noise.dtype
                        ) * self.train_config.batch_noise_correction_scale
                        batch_noise = batch_noise * batch_noise_scale
                        noise = noise + batch_noise
                
                if self.train_config.random_noise_shift > 0.0:
                    # get random noise -1 to 1
                    noise_shift = torch.randn(
                        s,
                        device=noise.device,
                        dtype=noise.dtype
                    ) * self.train_config.random_noise_shift
                    # add to noise
                    noise += noise_shift
                
                if self.train_config.random_noise_multiplier > 0.0:
                    sigma = self.train_config.random_noise_multiplier
                    noise_multiplier = torch.exp(torch.randn(s, device=noise.device, dtype=noise.dtype) * sigma)
                    noise = noise * noise_multiplier
            with self.timer('make_noisy_latents'):

                # handle adaptive scaling mased on std
                if self.train_config.adaptive_scaling_factor:
                    std = latents.std(dim=(2, 3), keepdim=True)
                    latents = latents * (1 / (std + 1e-6))

                if self.train_config.do_blank_stabilization:
                    # zero out latents with blank prompts
                    blank_latent = torch.zeros_like(latents)
                    for i, prompt in enumerate(conditioned_prompts):
                        if prompt.strip() == '':
                            latents[i] = blank_latent[i]

                batch.latents = latents

                # normalize latents to a mean of 0 and an std of 1
                # mean_zero_latents = latents - latents.mean()
                # latents = mean_zero_latents / mean_zero_latents.std()

                input_noise = noise
                if self.train_config.ip_noise_gamma and not getattr(self, '_disable_noise_augmentation', False):
                    from toolkit.noise_augmentation import augmentation_seed, input_perturbed_noise
                    input_noise = input_perturbed_noise(
                        noise, self.train_config.ip_noise_gamma,
                        random_strength=self.train_config.ip_noise_gamma_random_strength,
                        seed=augmentation_seed(self.train_config.noise_augmentation_seed,
                                               self.step_num, self.grad_accumulation_step, 2,
                                               batch_index=getattr(self, '_noise_augmentation_batch_index', 0)),
                    )
                if shared_pair_noise:
                    timesteps = torch.cat((timesteps, timesteps))
                    if input_noise is noise:
                        noise = input_noise = torch.cat((noise, noise))
                    else:
                        input_noise = torch.cat((input_noise, input_noise))
                        noise = torch.cat((noise, noise))
                noisy_latents = self.sd.add_noise(latents, input_noise, timesteps)

                # determine scaled noise
                # todo do we need to scale this or does it always predict full intensity
                # noise = noisy_latents - latents

                # https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1170C17-L1171C77
                if self.train_config.loss_target == 'source' or self.train_config.loss_target == 'unaugmented':
                    sigmas = self.get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    # add it to the batch
                    batch.sigmas = sigmas
                    # todo is this for sdxl? find out where this came from originally
                    # noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

            def double_up_tensor(tensor: torch.Tensor):
                if tensor is None:
                    return None
                return torch.cat([tensor, tensor], dim=0)

            if do_double:
                if self.model_config.refiner_name_or_path:
                    # apply refiner double up
                    refiner_timesteps = torch.randint(
                        max_noise_steps,
                        self.train_config.max_denoising_steps,
                        (batch_size,),
                        device=self.device_torch
                    )
                    refiner_timesteps = refiner_timesteps.long()
                    # add our new timesteps on to end
                    timesteps = torch.cat([timesteps, refiner_timesteps], dim=0)

                    refiner_noisy_latents = self.sd.noise_scheduler.add_noise(latents, input_noise, refiner_timesteps)
                    noisy_latents = torch.cat([noisy_latents, refiner_noisy_latents], dim=0)

                else:
                    # just double it
                    noisy_latents = double_up_tensor(noisy_latents)
                    timesteps = double_up_tensor(timesteps)

                noise = double_up_tensor(noise)
                # prompts are already updated above
                imgs = double_up_tensor(imgs)
                batch.mask_tensor = double_up_tensor(batch.mask_tensor)
                batch.control_tensor = double_up_tensor(batch.control_tensor)

            noisy_latent_multiplier = self.train_config.noisy_latent_multiplier

            if noisy_latent_multiplier != 1.0:
                noisy_latents = noisy_latents * noisy_latent_multiplier

            # remove grads for these
            noisy_latents.requires_grad = False
            noisy_latents = noisy_latents.detach()
            noise.requires_grad = False
            noise = noise.detach()

        return noisy_latents, noise, timesteps, conditioned_prompts, imgs

    def setup_adapter(self):
        # t2i adapter
        is_t2i = self.adapter_config.type == 't2i'
        is_control_net = self.adapter_config.type == 'control_net'
        # the writer's name: discovery once looked for _ref (written as _adapter), ignored an embedding, and used the
        # process name instead of the job name the files carry
        latest_save_path = self.get_latest_save_path(BaseSDTrainProcess._adapter_save_name(self))
        
        if latest_save_path is not None and not self.adapter_config.train:
            # the save path is for something else since we are not training
            latest_save_path = self.adapter_config.name_or_path

        dtype = get_torch_dtype(self.train_config.dtype)
        if is_t2i:
            # if we do not have a last save path and we have a name_or_path,
            # load from that
            if latest_save_path is None and self.adapter_config.name_or_path is not None:
                self.adapter = T2IAdapter.from_pretrained(
                    self.adapter_config.name_or_path,
                    torch_dtype=get_torch_dtype(self.train_config.dtype),
                    varient="fp16",
                    # use_safetensors=True,
                )
            else:
                self.adapter = T2IAdapter(
                    in_channels=self.adapter_config.in_channels,
                    channels=self.adapter_config.channels,
                    num_res_blocks=self.adapter_config.num_res_blocks,
                    downscale_factor=self.adapter_config.downscale_factor,
                    adapter_type=self.adapter_config.adapter_type,
                )
        elif is_control_net:
            if self.adapter_config.name_or_path is None:
                raise ValueError("ControlNet requires a name_or_path to load from currently")
            load_from_path = self.adapter_config.name_or_path
            if latest_save_path is not None:
                load_from_path = latest_save_path
            self.adapter = ControlNetModel.from_pretrained(
                load_from_path,
                torch_dtype=get_torch_dtype(self.train_config.dtype),
            )
        elif self.adapter_config.type == 'clip':
            self.adapter = ClipVisionAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
            )
        elif self.adapter_config.type == 'reference':
            self.adapter = ReferenceAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
            )
        elif self.adapter_config.type.startswith('ip'):
            self.adapter = IPAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
            )
            if self.train_config.gradient_checkpointing:
                self.adapter.enable_gradient_checkpointing()
        else:
            self.adapter = CustomAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
                train_config=self.train_config,
            )
        self.adapter.to(self.device_torch, dtype=dtype)
        if latest_save_path is not None and not is_control_net:
            # load adapter from path
            print_acc(f"Loading adapter from {latest_save_path}")
            if is_t2i:
                loaded_state_dict = load_t2i_model(
                    latest_save_path,
                    self.device,
                    dtype=dtype
                )
                self.adapter.load_state_dict(loaded_state_dict)
            elif self.adapter_config.type.startswith('ip'):
                # ip adapter
                loaded_state_dict = load_ip_adapter_model(
                    latest_save_path,
                    self.device,
                    dtype=dtype,
                    direct_load=self.adapter_config.train_only_image_encoder
                )
                self.adapter.load_state_dict(loaded_state_dict)
            else:
                # custom adapter
                loaded_state_dict = load_custom_adapter_model(
                    latest_save_path,
                    self.device,
                    dtype=dtype
                )
                self.adapter.load_state_dict(loaded_state_dict)
        if latest_save_path is not None and self.adapter_config.train:
            self.load_training_state_from_metadata(latest_save_path)
        # set trainable params
        self.sd.adapter = self.adapter
    
    def setup_validation(self):
        # caches everything needed for validation (latents, prompt embeds, fixed noise)
        # must be called while the vae and text encoder are still loaded, they may be
        # dumped later to save memory
        val_config = self.train_config.validation_config
        if val_config is not None and val_config.datasets:
            from training_extras.toolkit_validation import setup
            setup(self)
            return
        if val_config is None:
            return
        if not self.accelerator.is_main_process:
            return
        validation_items = []
        for item in val_config.validation_items:
            if not item.image_path:
                print_acc("Skipping validation item with no image")
                continue
            if not os.path.exists(item.image_path):
                print_acc(f"Skipping validation item, image not found: {item.image_path}")
                continue
            validation_items.append(item)
        if len(validation_items) == 0:
            print_acc("Validation config has no valid validation_items, skipping validation")
            return
        print_acc(f"Caching validation latents and embeddings for {len(validation_items)} images")
        device = self.device_torch
        dtype = get_torch_dtype(self.train_config.dtype)
        resolution = val_config.resolution

        divisibility = self.sd.get_bucket_divisibility()

        image_list = []
        prompt_list = []
        for item in validation_items:
            img = Image.open(item.image_path)
            img = ImageOps.exif_transpose(img).convert('RGB')
            # deterministic resize that keeps the aspect ratio, matches the pixel budget
            # of the resolution and the bucket divisibility of the model
            bucket = get_bucket_for_image_size(
                img.width, img.height,
                resolution=resolution,
                divisibility=divisibility,
            )
            img = img.resize((bucket['width'], bucket['height']), Image.BICUBIC)
            tensor = transforms.ToTensor()(img) * 2.0 - 1.0
            image_list.append(tensor)
            prompt = item.prompt
            if self.trigger_word is not None:
                prompt = self.sd.inject_trigger_into_prompt(
                    prompt,
                    trigger=self.trigger_word,
                    add_if_not_present=False,
                )
            prompt_list.append(prompt)

        fork_devices = [device] if device.type == 'cuda' else []
        with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
            # encode the prompts one at a time so they can be reassembled per sigma later
            te = self.sd.text_encoder
            te_list = te if isinstance(te, list) else ([te] if te is not None else [])
            orig_te_devices = [next(t.parameters()).device for t in te_list]
            self.sd.text_encoder_to(device)
            embeds_list = [
                self.sd.encode_prompt([prompt]).to('cpu', dtype=torch.float32).detach()
                for prompt in prompt_list
            ]
            for t, te_device in zip(te_list, orig_te_devices):
                t.to(te_device)

            # seed so the vae latent dist sampling is always identical
            torch.manual_seed(42)
            orig_vae_device = self.sd.vae.device
            # images can have different aspect ratios so they are encoded one at a time
            latent_list = [
                self.sd.encode_images([image], device=device, dtype=dtype).to('cpu', dtype=torch.float32)
                for image in image_list
            ]
            self.sd.vae.to(orig_vae_device)

            # fixed noise per image, seeds start at 42 and increment for each image
            noise_list = []
            for i, latent in enumerate(latent_list):
                generator = torch.Generator(device='cpu').manual_seed(42 + i)
                noise_list.append(
                    torch.randn(latent.shape, generator=generator, dtype=torch.float32)
                )

        self._validation_cache = {
            'latents': latent_list,
            'noise': noise_list,
            'embeds': embeds_list,
        }
        flush()

    def validate(self):
        val_config = self.train_config.validation_config
        if val_config is not None and self._validation_cache is not None and 'native_batches' in self._validation_cache:
            from training_extras.toolkit_validation import validate
            validate(self)
            return
        if val_config is None or self._validation_cache is None:
            return
        if not self.accelerator.is_main_process:
            return
        device = self.device_torch
        dtype = get_torch_dtype(self.train_config.dtype)
        sigmas = val_config.validation_sigmas
        cache = self._validation_cache

        was_unet_training = self.sd.unet.training
        self.sd.unet.eval()
        # the network is only active inside its context, without this the base model is validated
        network = self.network if self.network is not None else BlankNetwork()
        start_multiplier = network.multiplier
        network.multiplier = 1.0
        with torch.no_grad(), network:
            if self.sd.is_flow_matching:
                timestep_values = [sigma * 1000.0 for sigma in sigmas]
            else:
                num_train_timesteps = self.sd.noise_scheduler.config.num_train_timesteps
                timestep_values = [
                    min(int(round(sigma * num_train_timesteps)), num_train_timesteps - 1)
                    for sigma in sigmas
                ]
            timesteps = torch.tensor(timestep_values, device=device)

            # images can have different aspect ratios, so each image is predicted as its
            # own batch with all sigmas at once
            losses = []
            for latents_cpu, noise_cpu, embeds_cpu in zip(cache['latents'], cache['noise'], cache['embeds']):
                latents = latents_cpu.to(device, dtype=dtype)
                noise = noise_cpu.to(device, dtype=dtype)
                batch_latents = torch.cat([latents] * len(sigmas), dim=0)
                batch_noise = torch.cat([noise] * len(sigmas), dim=0)
                batch_embeds = concat_prompt_embeds([embeds_cpu.clone().to(device, dtype=dtype)] * len(sigmas))

                noisy_latents = self.sd.add_noise(batch_latents, batch_noise, timesteps).detach()

                noise_pred = self.sd.predict_noise(
                    latents=noisy_latents.to(device, dtype=dtype),
                    conditional_embeddings=batch_embeds,
                    timestep=timesteps,
                    guidance_scale=1.0,
                    guidance_embedding_scale=self.train_config.cfg_scale,
                    bypass_guidance_embedding=self.train_config.bypass_guidance_embedding,
                )

                if self.sd.is_flow_matching:
                    target = batch_noise - batch_latents
                elif self.sd.prediction_type == 'v_prediction':
                    target = self.sd.noise_scheduler.get_velocity(batch_latents, batch_noise, timesteps)
                else:
                    target = batch_noise

                losses.append(torch.nn.functional.mse_loss(noise_pred.float(), target.float()))

            val_loss = torch.stack(losses).mean()
            self.additional_logs['val/loss'] = val_loss.item()
        network.multiplier = start_multiplier
        if was_unet_training:
            self.sd.unet.train()

    def _rng_seed(self, *parts):
        """A seed derived from training_seed and parts (stream, step or epoch); None when train.reseed_per_step is
        off or no training_seed is set. Deriving per step makes a resumed run draw what an uninterrupted one would,
        instead of replaying the opening steps from the process-start seed."""
        if self.training_seed is None or not getattr(self.train_config, 'reseed_per_step', True):
            return None
        seed = int(self.training_seed)
        for part in parts:
            seed = (seed * 1000003 + int(part) + 1) % (2 ** 61 - 1)
        return seed

    def _reseed_step(self, step):
        seed = self._rng_seed(0, step)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)  # all devices

    def _seed_epoch_sampler(self, dataloader, stream, epoch):
        # the shuffling sampler gets its own per-epoch generator, independent of the global RNG
        seed = self._rng_seed(stream, epoch)
        sampler = getattr(dataloader, 'sampler', None)
        if seed is not None and isinstance(sampler, torch.utils.data.RandomSampler):
            sampler.generator = torch.Generator().manual_seed(seed)

    def run(self):
        try:
            self._run_training()
        finally:
            # the loop's commit observers hold this trainer; an error, KeyboardInterrupt or failed save/sample must
            # not leave them on an optimizer a later job in this process may reuse (docs/STATE_MACHINES.md, R01)
            self._remove_update_commit_observers()
            # this run no longer writes its folder (OPUS-ONEWRITER-01); the job releases whatever is left
            BaseTrainProcess.release_save_root(self)

    def _check_trainer_accumulation_capability(self):
        """Refuse accumulation settings this trainer's hook_train_loop does not implement, before any model loads.
        Such a setting never worked: the trainer silently trained at a different effective batch size."""
        name = type(self).__name__
        gas = self.train_config.gradient_accumulation_steps
        if gas != 1 and not self.supports_gradient_accumulation_steps:
            raise ValueError(
                f"{name} steps the optimizer on every iteration, so gradient_accumulation_steps={gas} would be ignored; "
                f"use 1 (and a larger batch_size if you need a bigger effective batch)")
        batches = self.train_config.gradient_accumulation
        if batches > 1 and not self.supports_gradient_accumulation_batches:
            raise ValueError(
                f"{name} trains on the first batch of each iteration only, so gradient_accumulation={batches} would "
                f"fetch batches it never uses; use 1")

    def _run_training(self):
        # torch.autograd.set_detect_anomaly(True)
        # run base process run
        BaseTrainProcess.run(self)
        self._check_trainer_accumulation_capability()
        # process-wide, so set it both ways: run.py runs several jobs in one process
        torch.set_float32_matmul_precision('high' if self.train_config.allow_tf32 else 'highest')
        if self.train_config.allow_tf32:
            print_acc("TF32 matmuls enabled (train.allow_tf32)")
        params = []
        from training_extras import rawr_checkpoint, rawr_pair
        if not rawr_checkpoint.is_rawr_name(self.train_config.lr_scheduler):
            # before resuming or writing anything: a committed pair here would stay the resume point
            rawr_pair.refuse_stale_manifest(self.save_root)
        # before loading anything: two artifacts sharing a filename would overwrite each other at every save
        self._refuse_colliding_save_names()

        ### HOOK ###
        self.hook_before_model_load()
        model_config_to_load = copy.deepcopy(self.model_config)

        if self.is_fine_tuning or self.train_config.merge_network_on_save:
            # get the latest checkpoint
            # check to see if we have a latest save
            # exclude pretrained_lora_path here so a pretrained lora is not loaded as full model
            # weights. It is loaded as the initial lora later when building the network.
            latest_save_path = self.get_latest_save_path(include_pretrained_lora=False)

            if latest_save_path is not None:
                print_acc(f"#### IMPORTANT RESUMING FROM {latest_save_path} ####")
                model_config_to_load.name_or_path = latest_save_path
                self.load_training_state_from_metadata(latest_save_path)

        ModelClass = get_model_class(self.model_config)
        # if the model class has get_train_scheduler static method
        if hasattr(ModelClass, 'get_train_scheduler'):
            sampler = ModelClass.get_train_scheduler()
        else:
            # get the noise scheduler
            arch = 'sd'
            if self.model_config.is_pixart:
                arch = 'pixart'
            if self.model_config.is_flux:
                arch = 'flux'
            if self.model_config.is_lumina2:
                arch = 'lumina2'
            sampler = get_sampler(
                self.train_config.noise_scheduler,
                {
                    "prediction_type": "v_prediction" if self.model_config.is_v_pred else "epsilon",
                },
                arch=arch,
            )

        if self.train_config.train_refiner and self.model_config.refiner_name_or_path is not None and self.network_config is None:
            previous_refiner_save = self.get_latest_save_path(self.job.name + '_refiner')
            if previous_refiner_save is not None:
                model_config_to_load.refiner_name_or_path = previous_refiner_save
                self.load_training_state_from_metadata(previous_refiner_save)

        self.sd = ModelClass(
            # todo handle single gpu and multi gpu here
            # device=self.device,
            device=self.accelerator.device,
            model_config=model_config_to_load,
            dtype=self.train_config.dtype,
            custom_pipeline=self.custom_pipeline,
            noise_scheduler=sampler,
        )
        
        self.hook_after_sd_init_before_load()
        if hasattr(self.sd, "track_feedback_sources"):
            self.sd.track_feedback_sources = self.train_config.dpo_feedback is not None
        # run base sd process run
        self.sd.load_model()
        
        self.sd.add_after_sample_image_hook(self.sample_step_hook)

        dtype = get_torch_dtype(self.train_config.dtype)

        # model is loaded from BaseSDProcess
        unet = self.sd.unet
        vae = self.sd.vae
        tokenizer = self.sd.tokenizer
        text_encoder = self.sd.text_encoder
        noise_scheduler = self.sd.noise_scheduler

        if self.train_config.xformers:
            if vae is not None:
                vae.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
            if isinstance(text_encoder, list):
                for te in text_encoder:
                    # if it has it
                    if hasattr(te, 'enable_xformers_memory_efficient_attention'):
                        te.enable_xformers_memory_efficient_attention()
        
        if self.train_config.attention_backend != 'native':
            if hasattr(vae, 'set_attention_backend'):
                vae.set_attention_backend(self.train_config.attention_backend)
            if hasattr(unet, 'set_attention_backend'):
                unet.set_attention_backend(self.train_config.attention_backend)
            if isinstance(text_encoder, list):
                for te in text_encoder:
                    if hasattr(te, 'set_attention_backend'):
                        te.set_attention_backend(self.train_config.attention_backend)
            else:
                if hasattr(text_encoder, 'set_attention_backend'):
                    text_encoder.set_attention_backend(self.train_config.attention_backend)
        if self.train_config.sdp:
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # # check if we have sage and is flux
        # if self.sd.is_flux:
        #     # try_to_activate_sage_attn()
        #     try:
        #         from sageattention import sageattn
        #         from toolkit.models.flux_sage_attn import FluxSageAttnProcessor2_0
        #         model: FluxTransformer2DModel = self.sd.unet
        #         # enable sage attention on each block
        #         for block in model.transformer_blocks:
        #             processor = FluxSageAttnProcessor2_0()
        #             block.attn.set_processor(processor)
        #         for block in model.single_transformer_blocks:
        #             processor = FluxSageAttnProcessor2_0()
        #             block.attn.set_processor(processor)
                    
        #     except ImportError:
        #         print_acc("sage attention is not installed. Using SDP instead")

        if self.train_config.gradient_checkpointing:
            # if has method enable_gradient_checkpointing
            if hasattr(unet, 'enable_gradient_checkpointing'):
                unet.enable_gradient_checkpointing()
            elif hasattr(unet, 'gradient_checkpointing'):
                unet.gradient_checkpointing = True
            else:
                print("Gradient checkpointing not supported on this model")
            if isinstance(text_encoder, list):
                for te in text_encoder:
                    if hasattr(te, 'enable_gradient_checkpointing'):
                        te.enable_gradient_checkpointing()
                    if hasattr(te, "gradient_checkpointing_enable"):
                        te.gradient_checkpointing_enable()
            else:
                if hasattr(text_encoder, 'enable_gradient_checkpointing'):
                    text_encoder.enable_gradient_checkpointing()
                if hasattr(text_encoder, "gradient_checkpointing_enable"):
                    text_encoder.gradient_checkpointing_enable()

        if self.sd.refiner_unet is not None:
            self.sd.refiner_unet.to(self.device_torch, dtype=dtype)
            self.sd.refiner_unet.requires_grad_(False)
            self.sd.refiner_unet.eval()
            if self.train_config.xformers:
                self.sd.refiner_unet.enable_xformers_memory_efficient_attention()
            if self.train_config.gradient_checkpointing:
                self.sd.refiner_unet.enable_gradient_checkpointing()

        if isinstance(text_encoder, list):
            for te in text_encoder:
                te.requires_grad_(False)
                te.eval()
        elif text_encoder is not None:
            text_encoder.requires_grad_(False)
            text_encoder.eval()
        unet.to(self.device_torch, dtype=dtype)
        unet.requires_grad_(False)
        unet.eval()
        if vae is not None:
            vae = vae.to(torch.device('cpu'), dtype=dtype)
            vae.requires_grad_(False)
            vae.eval()
        if self.train_config.learnable_snr_gos:
            self.snr_gos = LearnableSNRGamma(
                self.sd.noise_scheduler, device=self.device_torch
            )
            # check to see if previous settings exist
            path_to_load = os.path.join(self.save_root, 'learnable_snr.json')
            from training_extras import rawr_checkpoint, rawr_pair
            paired_snr = (rawr_checkpoint.is_rawr_name(self.train_config.lr_scheduler)
                          and rawr_pair.select(self.save_root, self.job.name + ('_LoRA' if self.named_lora else ''))
                          is not None)
            # The committed optimizer includes authoritative SNR parameters,
            # AdamW state and loss history. Its legacy JSON alias can be newer,
            # stale or truncated after an interrupted save; do not read it.
            if os.path.exists(path_to_load) and not paired_snr:
                with open(path_to_load, 'r') as f:
                    json_data = json.load(f)
                    if 'offset' in json_data:
                        # legacy
                        self.snr_gos.offset_2.data = torch.tensor(json_data['offset'], device=self.device_torch)
                    else:
                        self.snr_gos.offset_1.data = torch.tensor(json_data['offset_1'], device=self.device_torch)
                        self.snr_gos.offset_2.data = torch.tensor(json_data['offset_2'], device=self.device_torch)
                    self.snr_gos.scale.data = torch.tensor(json_data['scale'], device=self.device_torch)
                    self.snr_gos.gamma.data = torch.tensor(json_data['gamma'], device=self.device_torch)

        self.hook_after_model_load()
        flush()
        if not self.is_fine_tuning:
            if self.network_config is not None:
                # TODO should we completely switch to LycorisSpecialNetwork?
                network_kwargs = self.network_config.network_kwargs
                is_lycoris = False
                is_lorm = self.network_config.type.lower() == 'lorm'
                # default to LoCON if there are any conv layers or if it is named
                NetworkClass = LoRASpecialNetwork
                if self.network_config.type.lower() == 'locon' or self.network_config.type.lower() == 'lycoris':
                    if network_kwargs.get('full_locon', False):
                        # Native LoCon factors plus full deltas for weighted norm
                        # layers. This route also supports quantized transformers
                        # and the toolkit's CPU adapter offloading.
                        NetworkClass = LoRASpecialNetwork
                    else:
                        NetworkClass = LycorisSpecialNetwork
                        is_lycoris = True

                if is_lorm:
                    network_kwargs['ignore_if_contains'] = lorm_ignore_if_contains
                    network_kwargs['parameter_threshold'] = lorm_parameter_threshold
                    network_kwargs['target_lin_modules'] = LORM_TARGET_REPLACE_MODULE

                # if is_lycoris:
                #     preset = PRESET['full']
                # NetworkClass.apply_preset(preset)
                
                if hasattr(self.sd, 'target_lora_modules'):
                    network_kwargs['target_lin_modules'] = self.sd.target_lora_modules

                self.network = NetworkClass(
                    text_encoder=text_encoder,
                    unet=self.sd.get_model_to_train(),
                    lora_dim=self.network_config.linear,
                    multiplier=1.0,
                    alpha=self.network_config.linear_alpha,
                    train_unet=self.train_config.train_unet,
                    train_text_encoder=self.train_config.train_text_encoder,
                    conv_lora_dim=self.network_config.conv,
                    conv_alpha=self.network_config.conv_alpha,
                    is_sdxl=self.model_config.is_xl or self.model_config.is_ssd,
                    is_v2=self.model_config.is_v2,
                    is_v3=self.model_config.is_v3,
                    is_pixart=self.model_config.is_pixart,
                    is_auraflow=self.model_config.is_auraflow,
                    is_flux=self.model_config.is_flux,
                    is_lumina2=self.model_config.is_lumina2,
                    is_ssd=self.model_config.is_ssd,
                    is_vega=self.model_config.is_vega,
                    dropout=self.network_config.dropout,
                    use_text_encoder_1=self.model_config.use_text_encoder_1,
                    use_text_encoder_2=self.model_config.use_text_encoder_2,
                    use_bias=is_lorm,
                    is_lorm=is_lorm,
                    network_config=self.network_config,
                    network_type=self.network_config.type,
                    transformer_only=self.network_config.transformer_only,
                    is_transformer=self.sd.is_transformer,
                    base_model=self.sd,
                    **network_kwargs
                )


                # todo switch everything to proper mixed precision like this
                # Offloaded adapters must not first allocate their entire FP32
                # master weights on CUDA; large ranks can OOM before attach().
                network_init_device = torch.device('cpu') if self.network_config.layer_offloading else self.device_torch
                self.network.force_to(network_init_device, dtype=torch.float32)
                # give network to sd so it can use it
                self.sd.network = self.network
                self.network._update_torch_multiplier()

                self.network.apply_to(
                    text_encoder,
                    unet,
                    self.train_config.train_text_encoder,
                    self.train_config.train_unet
                )

                # we cannot merge in if quantized or offloading. note: torchao quantized weights can
                # still be force merged at save time for the merge-and-reset method (see save logic),
                # but we keep can_merge_in False here so sampling never merges in/out.
                # models loaded from pre-quantized checkpoints (e.g. comfy convrot/nvfp4
                # imports) never set model_config.quantize, so detect their layers too
                model_is_prequantized = any(
                    getattr(m, 'is_ostris_quantized', False) for m in unet.modules()
                ) if unet is not None else False
                if self.model_config.quantize or self.model_config.layer_offloading or model_is_prequantized:
                    # todo find a way around this
                    self.network.can_merge_in = False
                # Training samples run the network through its hooks. A merge_in/merge_out round trip rounds the
                # frozen base weights through their dtype, so in bf16/fp16 every sample event would move them
                # (docs/STATE_MACHINES.md, SM-06). merge_network_on_save still merges.
                self.sd.merge_network_for_sampling = False

                if is_lorm:
                    self.network.is_lorm = True
                    # make sure it is on the right device
                    self.sd.unet.to(self.sd.device, dtype=dtype)
                    original_unet_param_count = count_parameters(self.sd.unet)
                    self.network.setup_lorm()
                    new_unet_param_count = original_unet_param_count - self.network.calculate_lorem_parameter_reduction()

                    print_lorm_extract_details(
                        start_num_params=original_unet_param_count,
                        end_num_params=new_unet_param_count,
                        num_replaced=len(self.network.get_all_modules()),
                    )

                self.network.prepare_grad_etc(text_encoder, unet)
                flush()

                # LyCORIS doesnt have default_lr
                config = {
                    'text_encoder_lr': self.train_config.lr,
                    'unet_lr': self.train_config.lr,
                }
                sig = inspect.signature(self.network.prepare_optimizer_params)
                if 'default_lr' in sig.parameters:
                    config['default_lr'] = self.train_config.lr
                if 'learning_rate' in sig.parameters:
                    config['learning_rate'] = self.train_config.lr
                params_net = self.network.prepare_optimizer_params(
                    **config
                )

                params += params_net

                if self.train_config.gradient_checkpointing:
                    self.network.enable_gradient_checkpointing()

                # the names the writer uses (job name, not process name; the model's split files), so the LoRA it saved
                # is the one resumed (OPUS-RES-01)
                lora_name = BaseSDTrainProcess._lora_save_name(self)

                latest_save_path = self.get_latest_save_path(
                    lora_name, suffixes=BaseSDTrainProcess._lora_save_suffixes(self))
                extra_weights = None
                if latest_save_path is not None and not self.train_config.merge_network_on_save:
                    print_acc(f"#### IMPORTANT RESUMING FROM {latest_save_path} ####")
                    print_acc(f"Loading from {latest_save_path}")
                    extra_weights = self.load_weights(latest_save_path)
                    self._check_model_lora_ignores_on_resume(latest_save_path, extra_weights)
                    self._check_resumed_network_keys(latest_save_path)
                    self.network.multiplier = 1.0
                elif self.train_config.merge_network_on_save and self.network_config.pretrained_lora_path is not None:
                    # with merge_network_on_save, saved checkpoints are full models that get loaded as the
                    # base model. Only load the pretrained lora as the initial lora when we are not resuming
                    # from a saved checkpoint (otherwise it is already merged into the loaded model).
                    resume_save_path = self.get_latest_save_path(include_pretrained_lora=False)
                    if resume_save_path is None and os.path.exists(self.network_config.pretrained_lora_path):
                        print_acc(f"Loading initial lora from pretrained lora path: {self.network_config.pretrained_lora_path}")
                        extra_weights = self.load_weights(self.network_config.pretrained_lora_path)
                        self.network.multiplier = 1.0
                
                if self.network_config.layer_offloading:
                    MemoryManager.attach(
                        self.network,
                        self.device_torch
                    )
                    # CPU initialization made the multiplier a CPU tensor. Once
                    # managed, derive its device from the layer's compute device.
                    self.network._update_torch_multiplier()

            if self.embed_config is not None:
                # we are doing embedding training as well
                self.embedding = Embedding(
                    sd=self.sd,
                    embed_config=self.embed_config
                )
                latest_save_path = self.get_latest_save_path(self.embed_config.trigger)
                # load last saved weights
                if latest_save_path is not None:
                    self.embedding.load_embedding_from_file(latest_save_path, self.device_torch)
                    BaseSDTrainProcess._note_resumed(self, latest_save_path)
                    if self.embedding.step > 1:
                        self.step_num = self.embedding.step
                        self.start_step = self.step_num

                # self.step_num = self.embedding.step
                # self.start_step = self.step_num
                params.append({
                    'params': list(self.embedding.get_trainable_params()),
                    'lr': self.train_config.embedding_lr
                })

                flush()
            
            if self.decorator_config is not None:
                self.decorator = Decorator(
                    num_tokens=self.decorator_config.num_tokens,
                    token_size=4096 # t5xxl hidden size for flux
                )
                latest_save_path = self.get_latest_save_path(BaseSDTrainProcess._decorator_save_name(self))
                # load last saved weights
                if latest_save_path is not None:
                    state_dict = load_file(latest_save_path)
                    self.decorator.load_state_dict(state_dict)
                    self.load_training_state_from_metadata(latest_save_path)
                    
                params.append({
                    'params': list(self.decorator.parameters()),
                    'lr': self.train_config.lr
                })
                
                # give it to the sd network
                self.sd.decorator = self.decorator
                self.decorator.to(self.device_torch, dtype=torch.float32)
                self.decorator.train()

                flush()

            if self.adapter_config is not None:
                self.setup_adapter()
                if self.adapter_config.train:

                    if isinstance(self.adapter, IPAdapter):
                        # we have custom LR groups for IPAdapter
                        adapter_param_groups = self.adapter.get_parameter_groups(self.train_config.adapter_lr)
                        for group in adapter_param_groups:
                            params.append(group)
                    else:
                        # set trainable params
                        params.append({
                            'params': list(self.adapter.parameters()),
                            'lr': self.train_config.adapter_lr
                        })

                if self.train_config.gradient_checkpointing:
                    self.adapter.enable_gradient_checkpointing()
                flush()

            params = self.load_additional_training_modules(params)

        else:  # no network, embedding or adapter
            # set the device state preset before getting params
            self.sd.set_device_state(self.get_params_device_state_preset)

            # params = self.get_params()
            if len(params) == 0:
                # will only return savable weights and ones with grad
                params = self.sd.prepare_optimizer_params(
                    unet=self.train_config.train_unet,
                    text_encoder=self.train_config.train_text_encoder,
                    text_encoder_lr=self.train_config.lr,
                    unet_lr=self.train_config.lr,
                    default_lr=self.train_config.lr,
                    refiner=self.train_config.train_refiner and self.sd.refiner_unet is not None,
                    refiner_lr=self.train_config.refiner_lr,
                )
            # we may be using it for prompt injections
            if self.adapter_config is not None and self.adapter is None:
                self.setup_adapter()
        flush()

        ### HOOK ###
        params = self.hook_add_extra_train_params(params)
        self.params = params
        # self.params = []

        # for param in params:
        #     if isinstance(param, dict):
        #         self.params += param['params']
        #     else:
        #         self.params.append(param)

        if self.train_config.start_step is not None:
            self.step_num = self.train_config.start_step
            self.start_step = self.step_num

        optimizer_type = self.train_config.optimizer.lower()
        
        # esure params require grad
        self.ensure_params_requires_grad(force=True)

        # run-length warmups (SimplifiedAdEMAMixExM beta1_warmup) default to the optimizer-step count
        from training_extras.optim import apply_run_length_defaults
        gas = self.train_config.gradient_accumulation_steps
        optimizer_steps = self.train_config.steps // gas if gas and gas > 0 else None
        optimizer_params = apply_run_length_defaults(optimizer_type, self.train_config.optimizer_params, optimizer_steps)
        optimizer = get_optimizer(self.params, optimizer_type, learning_rate=self.train_config.lr,
                                  optimizer_params=optimizer_params)
        self.optimizer = optimizer
        self._check_accumulation_support(optimizer)
        # the ordered (name, shape, dtype) of what this optimizer trains: saved with, and checked against, optimizer.pt;
        # refused here, before any resume or training, if a parameter has no module name to record
        checkpoint_identity.require_named(BaseSDTrainProcess._parameter_manifest(self, optimizer))

        # set it to do paramiter swapping
        if self.train_config.do_paramiter_swapping:
            # only works for adafactor, but it should have thrown an error prior to this otherwise
            self.optimizer.enable_paramiter_swapping(self.train_config.paramiter_swapping_factor)

        from training_extras import rawr_checkpoint
        rawr_resume_enabled = rawr_checkpoint.is_rawr_name(self.train_config.lr_scheduler)
        self._rawr_checkpoint_config = (rawr_checkpoint.configuration(
            optimizer, self.train_config.steps, self.train_config.lr_scheduler_params)
            if rawr_resume_enabled else None)
        self._rawr_scheduler_restored = False

        rawr_resume = self._load_optimizer_state(optimizer, rawr_resume_enabled)
        # schedule-free optimizers start in eval mode and refuse to step until train() (docs/STATE_MACHINES.md, C12)
        from training_extras.optim import enter_training_mode
        enter_training_mode(optimizer)

        # set up the ema now that the optimizer (and its params) are ready
        self.setup_ema()
        BaseSDTrainProcess._restore_ema_resume_state(self)
        # every save mode is known now (EMA, embedding, adapter): refuse a RAWR pair the writer cannot produce before
        # training toward a first save that would abort (OPUS-RAWR-01)
        rawr_pair.refuse_unsupported_writer(self)

        lr_scheduler_params = self.train_config.lr_scheduler_params

        # make sure it had bare minimum
        if 'max_iterations' not in lr_scheduler_params:
            lr_scheduler_params['total_iters'] = self.train_config.steps

        lr_scheduler = get_lr_scheduler(
            self.train_config.lr_scheduler,
            optimizer,
            **lr_scheduler_params
        )
        self.lr_scheduler = lr_scheduler
        # Construction advances RAWR's retained cycle fields. Restore them and
        # scheduler state together; non-RAWR schedulers retain their old path.
        self._rawr_scheduler_restored = rawr_checkpoint.restore(
            lr_scheduler, rawr_resume, self._rawr_checkpoint_config, self.step_num)

        # cache validation latents and embeddings now, the vae and text encoder
        # may be dumped before the train loop starts
        self.setup_validation()

        ### HOOk ###
        self.before_dataset_load()
        if getattr(self.sd, 'require_pixel_tensor_cache', False):
            # model needs pixel tensors at train time even with cached latents;
            # storing them changes the latent cache key (first run re-caches)
            for ds_list in [self.datasets, self.datasets_reg]:
                if ds_list is None:
                    continue
                for ds in ds_list:
                    if not (ds.cache_latents or ds.cache_latents_to_disk):
                        # live-loading datasets already have pixels on the batch
                        continue
                    if not ds.cache_tensors_to_disk:
                        print_acc(
                            f"Model requires cached pixel tensors: forcing "
                            f"cache_tensors_to_disk on dataset {ds.folder_path}"
                        )
                        ds.cache_tensors_to_disk = True
        # load datasets if passed in the root process
        if self.datasets is not None:
            self.data_loader = get_dataloader_from_datasets(
                self.datasets, self.train_config.batch_size, self.sd,
                pool_datasets=self.train_config.pool_datasets,
            )
        if self.datasets_reg is not None:
            self.data_loader_reg = get_dataloader_from_datasets(
                self.datasets_reg, self.train_config.batch_size, self.sd,
                pool_datasets=self.train_config.pool_datasets,
            )

        # a fresh text-embedding cache leaves the text encoder on the gpu (the
        # datasets don't restore placement: that would bounce it per dataset).
        # Park it once here, before accelerator.prepare brings the unet up.
        if (
            self.is_caching_text_embeddings
            and self.model_config.low_vram
            and self.sd.text_encoder is not None
        ):
            self.sd.text_encoder_to('cpu')

        flush()
        self.last_save_step = self.step_num
        ### HOOK ###
        self.hook_before_train_loop()

        # ============================================================
        # COMPILE
        #
        # compile: true
        #     -> whole-model torch.compile
        #
        # compile: true
        # block_compile: true
        #     -> block-level compilation
        # ============================================================
        if self.model_config.compile:
            compiled_refs = []  # (block_list, index, original_block) for rollback on failure
            try:
                inner_unet_check = unwrap_model(self.sd.unet)
                is_unet_offloaded = hasattr(inner_unet_check, '_memory_manager')

                text_encoder = getattr(self.sd, "text_encoder", None)
                text_encoder_check = unwrap_model(text_encoder) if text_encoder is not None else None
                is_te_offloaded = hasattr(text_encoder_check, '_memory_manager') if text_encoder_check is not None else False

                is_unet_quantized = getattr(self.model_config, 'quantize', False)
                is_quantized = is_unet_quantized or getattr(self.model_config, 'quantize_te', False)

                if not is_unet_offloaded:
                    self.sd.unet.to(self.device_torch)

                cache_size_limit = getattr(self.model_config, 'cache_size_limit', None)
                user_set_cache_limit = cache_size_limit is not None
                if user_set_cache_limit:
                    torch._dynamo.config.cache_size_limit = cache_size_limit
                torch._dynamo.config.suppress_errors = False
                # torch 2.9 inductor bug: the new memory-coalescing tiling analysis
                # crashes on some dynamic-shape index expressions (sympy PowByNatural
                # "assert p >= 0", seen with Qwen Image). The analysis doesn't apply
                # to dynamic shapes anyway, so turn it off.
                if hasattr(torch._inductor.config.triton, 'coalesce_tiling_analysis'):
                    torch._inductor.config.triton.coalesce_tiling_analysis = False

                compile_mode = getattr(self.model_config, 'compile_mode', 'default')
                compile_dynamic = getattr(self.model_config, 'compile_dynamic', True)
                compile_fullgraph = getattr(self.model_config, 'compile_fullgraph', False)
                block_compile = getattr(self.model_config, 'block_compile', False)

                # quantized + offloaded unet is incompatible with fullgraph; force it off
                if is_unet_quantized and is_unet_offloaded and compile_fullgraph:
                    print_acc(
                        "Quantized offloaded Transformer detected: fullgraph=True is incompatible, "
                        "switching to fullgraph=False."
                    )
                    compile_fullgraph = False

                cache_info = ""
                # ====================================================
                # BLOCK COMPILE
                # ====================================================
                if block_compile:
                    BLOCK_LIST_ATTRS = self.sd.get_transformer_block_names()

                    if BLOCK_LIST_ATTRS is None or len(BLOCK_LIST_ATTRS) == 0:
                        BLOCK_LIST_ATTRS = [
                            'layers',
                            'transformer_blocks',
                            'single_transformer_blocks',
                            'double_stream_blocks',
                            'single_stream_blocks',
                            'double_blocks',
                            'single_blocks',
                            'blocks',
                        ]
                    inner_unet = unwrap_model(self.sd.unet)

                    compiled_block_count = 0

                    for attr_name in BLOCK_LIST_ATTRS:
                        # attr_name may be a dotted path for models that nest their
                        # blocks (e.g. hidream_o1's "model.language_model.layers").
                        block_list = inner_unet
                        for part in attr_name.split('.'):
                            block_list = getattr(block_list, part, None)
                            if block_list is None:
                                break

                        if block_list is None:
                            continue

                        if not hasattr(block_list, '__len__'):
                            continue

                        for i, block in enumerate(block_list):
                            if not isinstance(block, torch.nn.Module):
                                continue

                            if hasattr(block, '_hf_hook'):
                                continue

                            compiled_refs.append((block_list, i, block))
                            block_list[i] = torch.compile(
                                block,
                                mode=compile_mode,
                                dynamic=compile_dynamic,
                                fullgraph=compile_fullgraph,
                            )
                            compiled_block_count += 1

                    if compiled_block_count > 0:
                        if user_set_cache_limit:
                            auto_cache_limit = max(cache_size_limit, compiled_block_count * 2)
                            if auto_cache_limit != cache_size_limit:
                                torch._dynamo.config.cache_size_limit = auto_cache_limit
                                cache_info = f", cache_size_limit={auto_cache_limit} (auto)"
                            else:
                                cache_info = f", cache_size_limit={cache_size_limit}"
                        else:
                            auto_cache_limit = compiled_block_count * 2
                            torch._dynamo.config.cache_size_limit = auto_cache_limit
                            cache_info = f", cache_size_limit={auto_cache_limit} (auto)"
                        print_acc(
                            f"Compiled {compiled_block_count} transformer block(s) "
                            f"with torch.compile (mode='{compile_mode}', fullgraph={compile_fullgraph}, dynamic={compile_dynamic}{cache_info})."
                        )
                        print_acc("The first forward pass will be slow during compile. This is normal.")
                        print_acc("If you are experiencing issues, disable block_compile.")
                    else:
                        print_acc(
                            f"No individual transformer blocks found; "
                            f"falling back to whole-model torch.compile "
                            f"(mode='{compile_mode}', fullgraph={compile_fullgraph}, dynamic={compile_dynamic}{cache_info})."
                        )
                        print_acc("The first forward pass will hang for a while. This is normal.")

                        if is_unet_quantized and not is_unet_offloaded and compile_fullgraph:
                            print_acc(
                                "Quantized model detected: fullgraph=True is incompatible "
                                "for whole-model compile, switching to fullgraph=False."
                            )
                            compile_fullgraph = False

                        if compile_mode == 'default':
                            self.sd.unet = torch.compile(
                                self.sd.unet,
                                dynamic=compile_dynamic,
                                fullgraph=compile_fullgraph,
                            )
                        else:
                            self.sd.unet = torch.compile(
                                self.sd.unet,
                                mode=compile_mode,
                                dynamic=compile_dynamic,
                                fullgraph=compile_fullgraph,
                            )

                # ====================================================
                # WHOLE MODEL COMPILE
                # ====================================================
                else:
                    print_acc("Compiling model with torch.compile (whole-model compile).")
                    print_acc("The first forward pass will hang for a while. This is normal.")

                    print_acc(
                        f"Using torch.compile settings: "
                        f"mode={compile_mode}, "
                        f"dynamic={compile_dynamic}, "
                        f"fullgraph={compile_fullgraph}{cache_info}"
                    )

                    if compile_fullgraph:
                        print_acc(
                            "fullgraph=True is incompatible with whole-model compile, "
                            "switching to fullgraph=False."
                        )
                        compile_fullgraph = False

                    if compile_mode == 'default':
                        self.sd.unet = torch.compile(
                            self.sd.unet,
                            dynamic=compile_dynamic,
                            fullgraph=compile_fullgraph,
                        )
                    else:
                        self.sd.unet = torch.compile(
                            self.sd.unet,
                            mode=compile_mode,
                            dynamic=compile_dynamic,
                            fullgraph=compile_fullgraph,
                        )

                if not is_unet_offloaded:
                    # once compiled, dynamo guards hold weakrefs to the params;
                    # .to() on quantized params requires swap_tensors, which fails
                    # on tensors with weakrefs. The model stays on device anyway,
                    # so make .to() a no-op.
                    unet_module = self.sd.unet
                    unet_module.to = lambda *args, **kwargs: unet_module

            except Exception as e:
                # undo any block-level compiles that happened before the failure,
                # so "continuing without compilation" is actually true
                if len(compiled_refs) > 0:
                    for block_list, i, original_block in compiled_refs:
                        block_list[i] = original_block

                if 'triton' in str(e).lower():
                    print_acc("WARNING: compile is disabled.")
                    print_acc("Triton is not available or not working on this system.")
                    print_acc("Install a working 'triton' package to use compile.")
                    print_acc("Continuing without compilation.")
                else:
                    print_acc(f"Failed to compile model: {e}")
                    print_acc("Continuing without compilation")

        feedback = None
        if self.train_config.dpo_feedback is not None:
            from toolkit.feedback_schedule import FeedbackSchedule
            self._feedback_schedule = feedback = FeedbackSchedule(self)

        if self.has_first_sample_requested and self.step_num <= 1 and not self.train_config.disable_sampling:
            print_acc("Generating first sample from first sample config")
            self.sample(0, is_first=True)

        # sample first
        if self.train_config.skip_first_sample or self.train_config.disable_sampling:
            print_acc("Skipping first sample due to config setting")
        elif self.step_num <= 1 or self.train_config.force_first_sample:
            print_acc("Generating baseline samples before training")
            self.sample(self.step_num)
        
        if self.accelerator.is_local_main_process:
            self.progress_bar = ToolkitProgressBar(
                total=self.train_config.steps,
                desc=self.job.name,
                leave=True,
                initial=self.step_num,
                iterable=range(0, self.train_config.steps),
            )
            self.progress_bar.pause()
        else:
            self.progress_bar = None

        rawr_stochastic_saved = None
        if rawr_resume_enabled:
            from training_extras.rawr_stochastic import StochasticResume
            if rawr_resume is not None and rawr_resume['payload'] is not None:
                rawr_stochastic_saved = rawr_resume['payload'].get('training_state', {}).get('stochastic')
            self._rawr_stochastic = StochasticResume(self, rawr_stochastic_saved)

        curriculum_cursor = None
        if self.train_config.logit_normal_clean_schedule is not None:
            from toolkit.clean_share_curriculum import prepare_curriculum
            from training_extras.rawr_stochastic import loader_identities
            curriculum_cursor = prepare_curriculum(self, loader_identities(self))

        self._data_epoch = 0
        if self.step_num > 0 and self._rng_seed(0, 0) is not None:
            concurrent = [d.folder_path for d in (self.datasets or []) + (self.datasets_reg or [])
                          if getattr(d, 'num_workers', 0) > 0 or getattr(d, 'batch_load_threads', 1) > 1]
            if concurrent or self.data_loader_reg is not None:
                print_acc("Resume: continuing close to, but not exactly like, an uninterrupted run. With dataloader "
                          "workers or batch_load_threads > 1, per-item randomness (caption dropout, token shuffle, "
                          "crops) can land on different items, and an epoch-boundary resume can shift that step's "
                          "noise and timesteps; with a reg dataset the image order restarts fresh. For an exact "
                          "continuation use num_workers: 0, batch_load_threads: 1 and no reg dataset.")
        if (self.step_num > 0 and self.train_config.gradient_accumulation_steps != 1
                and not getattr(self, '_rawr_scheduler_restored', False)):
            print_acc("Resume: this checkpoint does not store an unfinished gradient accumulation window. Unless it was "
                      "saved right after an optimizer update, the first update after this resume covers fewer "
                      "microbatches than the uninterrupted run would (a partial epoch with "
                      "gradient_accumulation_steps: -1). RAWR pairs carry the window for fixed-size accumulation.")
        if self.train_config.gradient_accumulation_steps == -1 and not self._main_loader_has_batches():
            raise ValueError("gradient_accumulation_steps: -1 accumulates over one epoch of the main dataset, which "
                             "needs a dataset with a known, nonzero number of batches")
        self._epoch_cursor = 0
        if self.data_loader is not None:
            dataloader = self.data_loader
            skip = 0
            feedback_recorded = False
            cursor = None  # main batches already taken from the current epoch (whole-epoch accumulation)
            if self.step_num > 0 and self.data_loader_reg is None and self._rng_seed(1, 0) is not None:
                # resumed: continue in the epoch, and at the batch, an uninterrupted run would be at
                per_epoch = len(dataloader)
                if per_epoch:
                    self._data_epoch, skip = divmod(self.step_num * self.train_config.gradient_accumulation, per_epoch)
            if feedback is not None and feedback.saved_position is not None:
                # Preference windows consumed no ordinary training images. Recorded
                # RAWR samplers restore their own cursor; never skip those twice.
                self._data_epoch, ordinary_skip = feedback.saved_position
                recorded = getattr(self, '_rawr_stochastic', None)
                feedback_recorded = (recorded is not None and not recorded.reason
                                     and bool(rawr_stochastic_saved and rawr_stochastic_saved.get('supported')))
                skip = 0 if feedback_recorded else ordinary_skip
            if curriculum_cursor is not None:
                self._data_epoch, skip = curriculum_cursor.position
            if feedback_recorded:
                dataloader_iterator = iter(dataloader)
                cursor = ordinary_skip
            elif skip == 0 and self._data_epoch > 0:
                # resumed exactly on an epoch boundary: let the first fetch take the in-loop epoch reset, like an
                # uninterrupted run (creating an iterator draws a worker base seed from the global RNG, so creating
                # it here instead would shift that step's noise and timesteps)
                self._data_epoch -= 1
                dataloader_iterator = iter(())
                cursor = len(dataloader)  # at that epoch's end: the first fetch takes the reset
            else:
                if self.step_num > 0 and self.data_loader_reg is not None:
                    # with reg batches interleaved the position can't be reconstructed: take a fresh order for
                    # this resume rather than replaying epoch 0's
                    self._seed_epoch_sampler(dataloader, 3, self.step_num)
                else:
                    self._seed_epoch_sampler(dataloader, 1, self._data_epoch)
                dataloader_iterator = iter(dataloader)
            self._epoch_cursor = skip if cursor is None else cursor
            for _ in range(skip):
                if dataloader.num_workers == 0 and hasattr(dataloader_iterator, '_next_index'):
                    dataloader_iterator._next_index()  # advance the sampler without loading the batch
                else:
                    next(dataloader_iterator)
        else:
            dataloader = None
            dataloader_iterator = None

        if self.data_loader_reg is not None:
            dataloader_reg = self.data_loader_reg
            # fresh (not replayed) order on resume; reg interleaving is not reconstructed exactly
            self._seed_epoch_sampler(dataloader_reg, 2, self.step_num)
            dataloader_iterator_reg = iter(dataloader_reg)
        else:
            dataloader_reg = None
            dataloader_iterator_reg = None

        # zero any gradients
        optimizer.zero_grad()

        if self._rawr_scheduler_restored:
            self.grad_accumulation_step = rawr_checkpoint.restore_training_state(
                self.optimizer, rawr_resume, self.train_config, self.step_num)
            rawr_checkpoint.restore_auxiliary_state(self, rawr_resume)  # SNR + data identity outside the recorded path
            rawr_resume = None  # release the saved CPU gradient copies

        if not self._rawr_scheduler_restored:
            self.lr_scheduler.step(self.step_num)

        self.sd.set_device_state(self.train_device_state_preset)
        flush()
        # self.step_num = 0

        # print_acc(f"Compiling Model")
        # torch.compile(self.sd.unet, dynamic=True)

        # make sure all params require grad
        self.ensure_params_requires_grad(force=True)


        if self._validation_cache is not None and 'native_batches' in self._validation_cache:
            from training_extras.toolkit_validation import validate
            validate(self, phase='initial')

        ###################################################################
        # TRAIN LOOP
        ###################################################################


        if rawr_resume_enabled:
            self._rawr_stochastic.restore_rng(rawr_stochastic_saved)

        start_step_num = self.step_num
        self._checkpoint_completed_step = self.step_num
        self._install_update_commit_observers(optimizer)
        did_first_flush = False
        flush_next = False
        for step in range(start_step_num, self.train_config.steps):
            self.step_num = step
            # before any per-step randomness: parameter swapping, random CFG, batch fetching, noise and timesteps
            self._reseed_step(step)
            if self.train_config.do_paramiter_swapping:
                self.optimizer.optimizer.swap_paramiters()
            self.timer.start('train_loop')
            if flush_next:
                flush()
                flush_next = False
            if self.train_config.do_random_cfg:
                self.train_config.do_cfg = True
                self.train_config.cfg_scale = value_map(random.random(), 0, 1, 1.0, self.train_config.max_cfg_scale)
            # default to true so various things can turn it off
            self.is_grad_accumulation_step = True
            if self.train_config.free_u:
                self.sd.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
            if self.progress_bar is not None:
                self.progress_bar.unpause()
            with torch.no_grad():
                # if is even step and we have a reg dataset, use that
                # todo improve this logic to send one of each through if we can buckets and batch size might be an issue
                is_reg_step = False
                is_save_step = self.save_config.save_every and self.step_num % self.save_config.save_every == 0
                is_sample_step = (
                    self.sample_config.sample_every
                    and self.step_num >= self.sample_config.sample_start_step
                    and self.step_num % self.sample_config.sample_every == 0
                )
                if self.train_config.disable_sampling:
                    is_sample_step = False

                batch_list = feedback.batches() if feedback is not None else None
                using_feedback = batch_list is not None
                if using_feedback:
                    if self.train_config.gradient_accumulation_steps == -1:
                        # Whole-epoch ordinary windows end in the loader. A preference
                        # window uses its selected pairs and leaves that loader parked.
                        self.is_grad_accumulation_step = False
                        self.grad_accumulation_step = 0
                else:
                    if curriculum_cursor is not None:
                        curriculum_cursor.begin()
                    batch_list, dataloader_iterator, dataloader_iterator_reg = self._fetch_batch_list(
                        step, dataloader, dataloader_iterator, dataloader_reg, dataloader_iterator_reg,
                        is_save_step, is_sample_step)
                    if curriculum_cursor is not None:
                        if len(dataloader) != curriculum_cursor.contract['batches_per_epoch']:
                            raise ValueError('primary loader length changed during clean-share curriculum')
                        for batch in batch_list:
                            curriculum_cursor.stamp(batch)

                # setup accumulation
                self._setup_accumulation_iteration()

            # flush()
            ### HOOK ###
            if self.torch_profiler is not None:
                self.torch_profiler.start()
            loss_dict, did_oom = self._run_train_step(step, batch_list, optimizer)
            if feedback is not None:
                feedback.finish_iteration(did_oom)
            if curriculum_cursor is not None and not using_feedback:
                if did_oom:
                    raise RuntimeError('OOM during clean-share curriculum; aborting rather than skipping curriculum data')
                curriculum_cursor.commit()
            if self.torch_profiler is not None:
                torch.cuda.synchronize()  # Make sure all CUDA ops are done
                self.torch_profiler.stop()
                
                print("\n==== Profile Results ====")
                print(self.torch_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
            self.timer.stop('train_loop')

            # run validation before any possible sampling/logging for this step
            if not did_oom and self.train_config.validation_config is not None and not self.is_grad_accumulation_step:
                val_config = self.train_config.validation_config
                is_validate_step = (
                    self.step_num == self.start_step
                    or (val_config.validate_every_n_steps and self.step_num % val_config.validate_every_n_steps == 0)
                )
                if is_validate_step:
                    with self.timer('validate'):
                        self.validate()

            if not did_first_flush:
                flush()
                did_first_flush = True
            # flush()
            # setup the networks to gradient checkpointing and everything works
            if self.adapter is not None and isinstance(self.adapter, ReferenceAdapter):
                self.adapter.clear_memory()

            with torch.no_grad():
                # torch.cuda.empty_cache()
                # if optimizer has get_lrs method, then use it
                learning_rate = 0.0
                if not did_oom and loss_dict is not None:
                    if hasattr(optimizer, 'get_avg_learning_rate'):
                        learning_rate = optimizer.get_avg_learning_rate()
                    elif hasattr(optimizer, 'get_learning_rates'):
                        learning_rate = optimizer.get_learning_rates()[0]
                    elif self.train_config.optimizer.lower().startswith('dadaptation') or \
                            self.train_config.optimizer.lower().startswith('prodigy'):
                        learning_rate = (
                                optimizer.param_groups[0]["d"] *
                                optimizer.param_groups[0]["lr"]
                        )
                    else:
                        learning_rate = optimizer.param_groups[0]['lr']

                    prog_bar_string = f"lr: {learning_rate:.1e}"
                    for key, value in loss_dict.items():
                        prog_bar_string += f" {key}: {value:.3e}"

                    if self.progress_bar is not None:
                        self.progress_bar.set_postfix_str(prog_bar_string)

                # if the batch is a DataLoaderBatchDTO, then we need to clean it up (every batch of the list: the
                # loop used to clean only the last one it fetched)
                self._cleanup_batch_list(batch_list)

                # don't do on first step
                if self.step_num != self.start_step:
                    if is_sample_step or is_save_step:
                        self.accelerator.wait_for_everyone()
                        
                    if is_save_step:
                        self._save_at_loop_step()
                        flush_next = True

                    if is_sample_step:
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        flush()
                        # print above the progress bar
                        if self.train_config.free_u:
                            self.sd.pipeline.disable_freeu()
                        self.sample(self.step_num)
                        if self.train_config.unload_text_encoder:
                            # make sure the text encoder is unloaded
                            self.sd.text_encoder_to('cpu')
                        flush()

                        self.ensure_params_requires_grad()
                        if self.progress_bar is not None:
                            self.progress_bar.unpause()

                    if self.logging_config.log_every and self.step_num % self.logging_config.log_every == 0:
                        with self.timer('log_to_tensorboard'):
                            # log to tensorboard
                            if self.accelerator.is_main_process:
                                if self.writer is not None:
                                    if loss_dict is not None:
                                        for key, value in loss_dict.items():
                                            self.writer.add_scalar(f"{key}", value, self.step_num)
                                        self.writer.add_scalar(f"lr", learning_rate, self.step_num)

                        if self.accelerator.is_main_process:
                            # log to logger
                            self.logger.log({
                                'learning_rate': learning_rate,
                            })
                            if loss_dict is not None:
                                for key, value in loss_dict.items():
                                    self.logger.log({
                                        f'loss/{key}': value,
                                    })
                            if self.additional_logs is not None:
                                for key, value in self.additional_logs.items():
                                    self.logger.log({
                                        key: value,
                                    })
                                self.additional_logs = {}
                    elif self.logging_config.log_every is None:
                        if self.accelerator.is_main_process:
                            # log every step
                            self.logger.log({
                                'learning_rate': learning_rate,
                            })
                            for key, value in loss_dict.items():
                                self.logger.log({
                                    f'loss/{key}': value,
                                })
                            if self.additional_logs is not None:
                                for key, value in self.additional_logs.items():
                                    self.logger.log({
                                        key: value,
                                    })
                                self.additional_logs = {}


                    if self.performance_log_every > 0 and self.step_num % self.performance_log_every == 0:
                        # print the timers and clear them
                        self.timer.print()
                        self.timer.reset()
                
                # commit log
                if self.accelerator.is_main_process:
                    with self.timer('commit_logger'):
                        self.logger.commit(step=self.step_num)

                # sets progress bar to match our step (step is complete, so completed count is step + 1)
                if self.progress_bar is not None:
                    self.progress_bar.update(step + 1 - self.progress_bar.n)

                #############################
                # End of step
                #############################

                # update various steps
                self._end_loop_iteration(step)
                self.end_step_hook()


        ###################################################################
        ##  END TRAIN LOOP
        ###################################################################
        self._remove_update_commit_observers()
        self.accelerator.wait_for_everyone()
        if self.progress_bar is not None:
            self.progress_bar.close()
        if self.train_config.free_u:
            self.sd.pipeline.disable_freeu()
        if self._validation_cache is not None and 'native_batches' in self._validation_cache:
            from training_extras.toolkit_validation import validate
            validate(self, phase='final')
        if self.accelerator.is_main_process:
            self.save()
        if not self.train_config.disable_sampling:
            self.sample(self.step_num)
            self.logger.commit(step=self.step_num)
        print_acc("")
        if self.accelerator.is_main_process:
            self.logger.finish()
        self.accelerator.end_training()

        if self.accelerator.is_main_process:
            # push to hub
            if self.save_config.push_to_hub:
                if("HF_TOKEN" not in os.environ):
                    interpreter_login(new_session=False, write_permission=True)
                self.push_to_hub(
                    repo_id=self.save_config.hf_repo_id,
                    private=self.save_config.hf_private
                )
        del (
            self.sd,
            unet,
            noise_scheduler,
            optimizer,
            self.network,
            tokenizer,
            text_encoder,
        )

        flush()
        self.done_hook()
        # after done_hook, so UI status/async work has finished with the job's objects
        self.release_accelerator_state()

    def push_to_hub(
    self,
    repo_id: str,
    private: bool = False,
    ):  
        if not self.accelerator.is_main_process:
            return
        readme_content = self._generate_readme(repo_id)
        readme_path = os.path.join(self.save_root, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        api = HfApi()

        api.create_repo(
            repo_id,
            private=private,
            exist_ok=True
        )

        api.upload_folder(
            repo_id=repo_id,
            folder_path=self.save_root,
            ignore_patterns=["*.yaml", "*.pt"],
            repo_type="model",
        )


    def _generate_readme(self, repo_id: str) -> str:
        """Generates the content of the README.md file."""

        # Gather model info
        base_model = self.model_config.name_or_path
        instance_prompt = self.trigger_word if hasattr(self, "trigger_word") else None
        if base_model == "black-forest-labs/FLUX.1-schnell":
            license = "apache-2.0"
        elif base_model == "black-forest-labs/FLUX.1-dev":
            license = "other"
            license_name = "flux-1-dev-non-commercial-license"
            license_link = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md"
        else:
            license = "creativeml-openrail-m"
        tags = [
            "text-to-image",
        ]
        if self.model_config.is_xl:
            tags.append("stable-diffusion-xl")
        if self.model_config.is_flux:
            tags.append("flux")
        if self.model_config.is_lumina2:
            tags.append("lumina2")
        if self.model_config.is_v3:
            tags.append("sd3")
        if self.network_config:
            tags.extend(
                [
                    "lora",
                    "diffusers",
                    "template:sd-lora",
                    "ai-toolkit",
                ]
            )

        # Generate the widget section
        widgets = []
        sample_image_paths = []
        samples_dir = os.path.join(self.save_root, "samples")
        if os.path.isdir(samples_dir):
            for filename in os.listdir(samples_dir):
                #The filenames are structured as 1724085406830__00000500_0.jpg
                #So here we capture the 2nd part (steps) and 3rd (index the matches the prompt)
                match = re.search(r"__(\d+)_(\d+)\.jpg$", filename)
                if match:
                    steps, index = int(match.group(1)), int(match.group(2))
                    #Here we only care about uploading the latest samples, the match with the # of steps
                    if steps == self.train_config.steps:
                        sample_image_paths.append((index, f"samples/{filename}"))

            # Sort by numeric index
            sample_image_paths.sort(key=lambda x: x[0])

            # Create widgets matching prompt with the index 
            for i, prompt in enumerate(self.sample_config.prompts):
                if i < len(sample_image_paths):
                    # Associate prompts with sample image paths based on the extracted index
                    _, image_path = sample_image_paths[i]
                    widgets.append(
                        {
                            "text": prompt,
                            "output": {
                                "url": image_path
                            },
                        }
                    )
        dtype = "torch.bfloat16" if self.model_config.is_flux else "torch.float16"
        # Construct the README content
        readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
license: {license}
{'license_name: ' + license_name if license == "other" else ""}
{'license_link: ' + license_link if license == "other" else ""}
---

# {self.job.name}
Model trained with [AI Toolkit by Ostris](https://github.com/ostris/ai-toolkit)
<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, etc.

Weights for this model are available in Safetensors format.

[Download](/{repo_id}/tree/main) them in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('{base_model}', torch_dtype={dtype}).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='{self.job.name}.safetensors')
image = pipeline('{instance_prompt if not widgets else self.sample_config.prompts[0]}').images[0]
image.save("my_image.png")
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

"""
        return readme_content
