"""Bounded stochastic continuation for serial CPU image training.

No worker replay, private DataLoader internals, GPU state, or cached stochastic
latents. Unsupported modes keep ordinary continuation and record the reason.
"""
import copy
import hashlib
import json
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, RandomSampler


def rng_state():
    n = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])]}


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


class RecordedSampler:
    """Record actual indices, not fetched batches or an estimate of RNG draws."""
    def __init__(self, sampler):
        self.sampler = sampler
        self.order = None
        self.cursor = 0

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        if self.order is None:
            self.order = list(self.sampler)
            self.cursor = 0
        while self.cursor < len(self.order):
            index = self.order[self.cursor]
            self.cursor += 1
            yield index
        self.order = None
        self.cursor = 0

    def state_dict(self):
        return {'order': copy.deepcopy(self.order), 'cursor': self.cursor}

    def load_state_dict(self, state):
        order, cursor = state.get('order'), state.get('cursor')
        if type(cursor) is not int or cursor < 0 or (order is None and cursor != 0):
            raise ValueError('Invalid RAWR sampler cursor')
        if order is not None:
            if not isinstance(order, list) or len(order) != len(self) or cursor > len(order):
                raise ValueError('Invalid RAWR sampler order')
            source = self.sampler.sampler if isinstance(self.sampler, BatchSampler) else self.sampler
            flat = [i for batch in order for i in batch] if isinstance(self.sampler, BatchSampler) else order
            if any(type(i) is not int or not 0 <= i < len(source.data_source) for i in flat):
                raise ValueError('Invalid RAWR sampler index')
        self.order, self.cursor = copy.deepcopy(order), cursor


ITEM_FIELDS = ('crop_x', 'crop_y', 'crop_width', 'crop_height',
               'scale_to_width', 'scale_to_height', 'flip_x', 'flip_y')


def _file_stamp(path):
    try:
        stat = os.stat(path)
        return [os.path.abspath(path), stat.st_size, stat.st_mtime_ns]
    except FileNotFoundError:
        return [os.path.abspath(path), None, None]


def _datasets(loader):
    return loader.dataset.datasets if isinstance(loader.dataset, ConcatDataset) else [loader.dataset]


def dataset_identity(dataset):
    files = []
    for item in dataset.file_list:
        # DatasetConfig normalizes the dot; match CaptionProcessingDTOMixin.load_caption.
        caption = os.path.splitext(item.path)[0] + item.dataset_config.caption_ext
        files.append([_file_stamp(item.path), _file_stamp(caption), item.flip_x, item.flip_y])
    configs = [vars(c) for c in dataset.source_configs]
    encoded = json.dumps({'files': files, 'configs': configs, 'captions': dataset.caption_dict},
                         sort_keys=True, default=lambda obj: vars(obj)).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dataset_state(dataset):
    return {'identity': dataset_identity(dataset), 'epoch': dataset.epoch_num,
            'batches': copy.deepcopy(dataset.batch_indices),
            'items': [{k: getattr(item, k) for k in ITEM_FIELDS if hasattr(item, k)}
                      for item in dataset.file_list]}


def _restore_dataset(dataset, state):
    if state['identity'] != dataset_identity(dataset):
        raise ValueError('RAWR dataset files, captions, order or configuration changed')
    batches, items = state['batches'], state['items']
    if len(items) != len(dataset.file_list) or len(batches) != len(dataset.batch_indices):
        raise ValueError('RAWR dataset layout changed')
    if any(type(i) is not int or not 0 <= i < len(items) for batch in batches for i in batch):
        raise ValueError('Invalid RAWR dataset batch index')
    dataset.batch_indices = copy.deepcopy(batches)
    for item, saved in zip(dataset.file_list, items):
        for key in ITEM_FIELDS:
            if key in saved:
                setattr(item, key, saved[key])
    dataset.epoch_num = state['epoch']


SNR_PARAMETERS = ('offset_1', 'offset_2', 'gamma', 'scale')


def snr_state(trainer):
    snr = getattr(trainer, 'snr_gos', None)
    if snr is None:
        return None
    return {'version': 1,
            'parameters': {key: getattr(snr, key).detach().cpu().clone() for key in SNR_PARAMETERS},
            'optimizer': copy.deepcopy(snr.optimizer.state_dict()),
            'buffer': [value.detach().cpu().clone() for value in snr.buffer],
            'max_buffer_size': snr.max_buffer_size}


def restore_snr(trainer, saved):
    snr = getattr(trainer, 'snr_gos', None)
    if (snr is None) != (saved is None):
        raise ValueError('Cannot restore RAWR stochastic state: learnable SNR configuration changed')
    if snr is None:
        return
    if (saved.get('version') != 1 or saved.get('max_buffer_size') != snr.max_buffer_size
            or set(saved.get('parameters', {})) != set(SNR_PARAMETERS)
            or not isinstance(saved.get('buffer'), list)
            or len(saved['buffer']) > snr.max_buffer_size):
        raise ValueError('Invalid RAWR learnable SNR state')
    for key in SNR_PARAMETERS:
        value, parameter = saved['parameters'][key], getattr(snr, key)
        if not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise ValueError('RAWR learnable SNR parameter layout changed')
    if any(not torch.is_tensor(value) or value.numel() != 1 for value in saved['buffer']):
        raise ValueError('Invalid RAWR learnable SNR loss history')
    snr.optimizer.load_state_dict(saved['optimizer'])
    with torch.no_grad():
        for key in SNR_PARAMETERS:
            getattr(snr, key).copy_(saved['parameters'][key])
    snr.optimizer.zero_grad()
    snr.buffer = [value.to(snr.device).clone() for value in saved['buffer']]


# train.reseed_per_step (BaseSDTrainProcess._rng_seed) already makes every step's draws and the data position a
# function of (training_seed, step/epoch), on GPU too. Exactly one mechanism may own the data stream, so this
# recorded-state path stands aside while reseeding is active and serves runs with reseeding off.
RESEEDED = 'per-step reseeding (train.reseed_per_step) owns the RNG and data position'


def loader_identities(trainer):
    """dataset_identity for every dataset of the training and reg loaders (file stamps, captions, config)."""
    result = []
    for attr in ('data_loader', 'data_loader_reg'):
        loader = getattr(trainer, attr, None)
        if loader is None:
            result.append(None)
            continue
        try:
            result.append([dataset_identity(d) for d in _datasets(loader)])
        except Exception:  # custom datasets without file_list/caption_dict
            result.append('unavailable')
    return result


def unsupported_reason(trainer):
    rng_seed = getattr(trainer, '_rng_seed', None)
    if getattr(trainer, 'training_seed', None) is not None and rng_seed is not None and rng_seed(0, 0) is not None:
        return RESEEDED
    if str(trainer.accelerator.device) != 'cpu' or trainer.accelerator.num_processes != 1:
        return 'only single-process CPU execution is supported'
    if (type(trainer).__module__ != 'extensions_built_in.sd_trainer.SDTrainer'
            or type(trainer.sd).__module__ != 'toolkit.stable_diffusion_model'):
        return 'requires the standard SDTrainer/StableDiffusion path; custom model and hook state is unavailable'
    if not trainer.train_config.disable_sampling or trainer.train_config.validation_config is not None:
        return 'sampling/validation after a checkpoint may consume unsaved RNG'
    if trainer.ema is not None or getattr(trainer, 'training_extras', None) is not None:
        return 'EMA/auxiliary state is outside this continuation path'
    snr = getattr(trainer, 'snr_gos', None)
    if snr is not None:
        from toolkit.train_tools import LearnableSNRGamma
        if type(snr) is not LearnableSNRGamma:
            return 'custom learnable SNR state is unavailable'
    if getattr(trainer.sd, 'is_multistage', False):
        return 'multistage model state is not recorded'
    if trainer.train_config.do_paramiter_swapping:
        return 'parameter-swapping state is not recorded'
    for loader in (trainer.data_loader, trainer.data_loader_reg):
        if loader is None:
            continue
        if type(loader) is not DataLoader or loader.num_workers != 0 or loader.generator is not None:
            return 'requires standard DataLoader, num_workers=0 and its default generator'
        if (type(loader.sampler) is not RandomSampler or loader.sampler.replacement
                or loader.sampler.generator is not None
                or (loader.batch_sampler is not None and type(loader.batch_sampler) is not BatchSampler)):
            return 'requires the standard non-replacement random sampler'
        for dataset in _datasets(loader):
            if type(dataset).__name__ != 'AiToolkitDataset' or type(dataset).__module__ != 'toolkit.data_loader':
                return 'custom dataset state is unavailable'
            if dataset.is_video or dataset.is_audio_model or dataset.is_multimodal_llm:
                return 'only image datasets are supported'
            for c in dataset.source_configs:
                if min(c.batch_load_threads, dataset.batch_size) > 1:
                    return 'parallel batch loading has independent RNG consumption'
                if any(getattr(c, k, False) for k in ('cache_latents', 'cache_latents_to_disk',
                        'cache_text_embeddings', 'cache_clip_vision_to_disk', 'cache_tensors_to_disk',
                        'augmentations', 'augments', 'clip_image_augmentations', 'controls',
                        'control_path', 'mask_path')):
                    return 'cached data, controls or stateful augmentations are unsupported'
    return None


class StochasticResume:
    def __init__(self, trainer, saved=None):
        self.trainer = trainer
        self.reason = unsupported_reason(trainer)
        self.streams = []
        if saved is not None and saved.get('supported') and self.reason:
            hint = ('; this checkpoint recorded its RNG and data state with train.reseed_per_step off, so resume it '
                    'with reseed_per_step: false' if self.reason == RESEEDED else '')
            raise ValueError('Cannot restore RAWR stochastic state: ' + self.reason + hint)
        if self.reason:
            if self.reason != RESEEDED:  # reseeding is the intended path, not a degradation
                warnings.warn('RAWR stochastic continuation unavailable: ' + self.reason, RuntimeWarning)
            return
        if saved is not None and saved.get('supported'):
            if saved.get('version') not in (1, 2):
                raise ValueError('Unsupported RAWR stochastic state version')
            if saved['version'] == 2 and 'snr' not in saved:
                raise ValueError('Missing RAWR learnable SNR state')
            # Version 1 did not capture SNR optimizer/history. Never silently
            # resume an SNR run from its lossy four-value legacy JSON instead.
            restore_snr(trainer, saved.get('snr'))
        elif getattr(trainer, 'snr_gos', None) is not None and getattr(trainer, 'step_num', 0) > 0:
            raise ValueError('Cannot restore RAWR stochastic state: checkpoint has no complete learnable SNR state')
        for attr in ('data_loader', 'data_loader_reg'):
            loader = getattr(trainer, attr)
            if loader is None:
                self.streams.append(None)
                continue
            sampler = RecordedSampler(loader.batch_sampler if loader.batch_sampler is not None else loader.sampler)
            kwargs = {'dataset': loader.dataset, 'collate_fn': loader.collate_fn, 'num_workers': 0}
            if loader.batch_sampler is None:
                kwargs.update(batch_size=None, sampler=sampler)
            else:
                kwargs['batch_sampler'] = sampler
            replacement = DataLoader(**kwargs)
            setattr(trainer, attr, replacement)
            self.streams.append((replacement, sampler))
        if saved is not None and saved.get('supported'):
            if len(saved['streams']) != len(self.streams):
                raise ValueError('RAWR data streams changed')
            for live, previous in zip(self.streams, saved['streams']):
                if (live is None) != (previous is None):
                    raise ValueError('RAWR data streams changed')
                if live is None:
                    continue
                loader, sampler = live
                if len(_datasets(loader)) != len(previous['datasets']):
                    raise ValueError('RAWR datasets changed')
                for dataset, state in zip(_datasets(loader), previous['datasets']):
                    _restore_dataset(dataset, state)
                sampler.load_state_dict(previous['sampler'])

    def state_dict(self):
        if self.reason:
            return {'supported': False, 'reason': self.reason}
        return {'supported': True, 'version': 2, 'rng': rng_state(), 'snr': snr_state(self.trainer),
                'streams': [None if stream is None else {
                    'sampler': stream[1].state_dict(),
                    'datasets': [_dataset_state(d) for d in _datasets(stream[0])]}
                    for stream in self.streams]}

    def restore_rng(self, saved):
        # Call only AFTER constructing DataLoader iterators and all startup work.
        if saved is not None and saved.get('supported'):
            restore_rng(saved['rng'])
        elif self.trainer.step_num > 0:
            warnings.warn('RAWR checkpoint has no restorable RNG/data state; stochastic continuation is approximate',
                          RuntimeWarning)
