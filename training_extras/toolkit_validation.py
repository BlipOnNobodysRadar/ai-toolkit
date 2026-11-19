"""Fixed held-out batches through the host's native image/video/audio path."""
import copy
import json
import os
import random
from contextlib import contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .optim import evaluation_optimizer
from toolkit.noise_augmentation import disable_noise_augmentation


@contextmanager
def fixed_rng(seed, device):
    py, np_state = random.getstate(), np.random.get_state()
    devices = [device] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def _raw_value(raw, key, default=None):
    if isinstance(raw, dict):
        return raw.get(key, default)
    return getattr(raw, key, default)


def _is_paired_validation(cfg):
    datasets = list(getattr(cfg, 'datasets', None) or [])
    return any(bool(_raw_value(d, 'rejected_path')) for d in datasets)


def _normalized_root(path):
    if path is None or not str(path).strip():
        return None
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _roots_overlap(left, right):
    left, right = _normalized_root(left), _normalized_root(right)
    if left is None or right is None:
        return False
    try:
        common = os.path.commonpath([left, right])
    except ValueError:
        # Different drives cannot overlap.
        return False
    return common in (left, right)


def _training_roots(trainer):
    roots = []
    datasets = (
        list(getattr(trainer, 'datasets', None) or [])
        + list(getattr(trainer, 'datasets_reg', None) or [])
        + list(getattr(trainer, 'dataset_configs', None) or [])
    )
    for dataset in datasets:
        roots.extend([
            getattr(dataset, 'folder_path', None) or getattr(dataset, 'dataset_path', None),
            getattr(dataset, 'rejected_path', None),
        ])
    return [root for root in roots if root]


def _validate_paired_validation_config(trainer, cfg):
    """Validate the first paired-validation constraints before loading data.

    Native held-out validation remains a separate loader.  Preference
    validation still needs the same pair invariants as training: one caption
    per chosen image, a rejected mate with matching geometry, and no
    conditioning/control branch that exists on only one arm.
    """
    datasets = list(getattr(cfg, 'datasets', None) or [])
    flags = [bool(_raw_value(raw, 'rejected_path')) for raw in datasets]
    if any(flags) and not all(flags):
        raise ValueError(
            'Cannot mix paired and unpaired validation datasets: either every '
            'validation dataset has rejected_path or none do'
        )
    if not any(flags):
        return False
    if getattr(trainer.train_config, 'loss_type', None) != 'dpo':
        raise ValueError(
            "Validation datasets[].rejected_path requires train.loss_type: dpo"
        )
    if getattr(trainer, 'network', None) is None:
        raise ValueError('Paired DPO validation requires a trainable LoRA network')

    # Reuse the production DPO option guard so held-out validation rejects the
    # same masks, controls, augmentations, and auxiliary branches as training.
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_loader import validate_dpo_config_options, validate_dpo_roots

    train_arms = _training_roots(trainer)
    for raw in datasets:
        if isinstance(raw, DatasetConfig):
            probe = raw
            chosen = raw.folder_path or raw.dataset_path
            rejected = raw.rejected_path
        else:
            probe = DatasetConfig(**raw)
            chosen = raw.get('folder_path') or raw.get('dataset_path')
            rejected = raw.get('rejected_path')
        if getattr(probe, 'is_reg', False):
            raise ValueError('Paired DPO validation cannot use regression datasets')
        validate_dpo_config_options(probe)
        validate_dpo_roots(chosen, rejected)

        val_arms = (
            ('chosen', chosen),
            ('rejected', rejected),
        )
        for arm_name, arm_root in val_arms:
            for train_root in train_arms:
                if _roots_overlap(arm_root, train_root):
                    raise ValueError(
                        f'Validation {arm_name} root overlaps a training root: '
                        f'validation={arm_root}, training={train_root}'
                    )
    return True


def _relative_chosen_image_id(file_item):
    """Return the chosen image path relative to its chosen dataset root."""
    path = getattr(file_item, 'path', None)
    if not path:
        return ''
    candidates = [getattr(file_item, 'dataset_root', None)]
    dataset_config = getattr(file_item, 'dataset_config', None)
    if dataset_config is not None:
        candidates.append(
            getattr(dataset_config, 'folder_path', None)
            or getattr(dataset_config, 'dataset_path', None)
        )
    for root in candidates:
        if not root:
            continue
        try:
            rel = os.path.relpath(os.path.abspath(str(path)), os.path.abspath(str(root)))
        except ValueError:
            continue
        if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
            return rel.replace(os.sep, '/')
    return os.path.basename(str(path))


def _embedding_batch_size(embeds, expected):
    """Validate that every cached caption tensor has one row per pair."""
    if not hasattr(embeds, 'keys'):
        return
    for key in embeds.keys():
        values = embeds[key]
        if isinstance(values, (list, tuple)) and len(values) != expected:
            raise ValueError(
                f'DPO validation caption embedding {key!r} has {len(values)} '
                f'rows for {expected} pairs'
            )
        if isinstance(values, torch.Tensor) and values.ndim > 0 and values.shape[0] not in (1, expected):
            raise ValueError(
                f'DPO validation caption embedding {key!r} has '
                f'{values.shape[0]} rows for {expected} pairs'
            )


def _validate_paired_batch_cache(batch, paired, prompts, noise, timesteps):
    pair_count = len(batch.file_items)
    if pair_count < 1:
        raise ValueError('DPO validation batch has no image pairs')
    if len(prompts) != pair_count:
        raise ValueError(
            'DPO validation requires exactly one chosen caption per image pair'
        )
    if any(getattr(item, 'rejected_item', None) is None for item in batch.file_items):
        raise ValueError('DPO validation batch has a chosen image without a rejected mate')
    if paired.latents is None:
        raise ValueError('DPO validation did not produce paired latents')
    expected_rows = pair_count * 2
    if paired.latents.shape[0] != expected_rows:
        raise ValueError(
            f'DPO validation produced {paired.latents.shape[0]} latents for '
            f'{pair_count} pairs; expected {expected_rows}'
        )
    if noise.shape[0] != expected_rows or timesteps.shape[0] != expected_rows:
        raise ValueError('DPO validation must return shared noise and timesteps for both arms')
    if not torch.equal(noise[:pair_count], noise[pair_count:]):
        raise ValueError('DPO validation chosen/rejected halves did not share noise')
    if not torch.equal(timesteps[:pair_count], timesteps[pair_count:]):
        raise ValueError('DPO validation chosen/rejected halves did not share timesteps')


def _timesteps_for_sigma(sd, sigma, batch_size, device):
    if not 0 <= sigma <= 1:
        raise ValueError('Validation sigmas must be in [0,1]')
    t = float(sigma) * sd.noise_scheduler.config.num_train_timesteps
    if not sd.is_flow_matching:
        t = min(round(t), sd.noise_scheduler.config.num_train_timesteps - 1)
    return torch.full(
        (batch_size,),
        t,
        device=device,
        dtype=torch.float32 if sd.is_flow_matching else torch.long,
    )


def _loss_target(sd, noise, batch, timesteps):
    """Use the same target entry point as the production training path."""
    if hasattr(sd, 'get_loss_target'):
        return sd.get_loss_target(noise=noise, batch=batch, timesteps=timesteps)
    if sd.is_flow_matching:
        return noise - batch.latents
    if sd.prediction_type == 'v_prediction':
        return sd.noise_scheduler.get_velocity(batch.latents, noise, timesteps)
    return noise


def _canonical_path(path):
    """Canonical, case-normalized path for component-wise containment checks."""
    if path is None:
        return None
    try:
        return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))
    except (TypeError, ValueError, OSError):
        return None


def _path_contains(parent, child):
    """True when child is parent or below parent, comparing path components."""
    parent = _canonical_path(parent)
    child = _canonical_path(child)
    if parent is None or child is None:
        return False
    try:
        return os.path.commonpath([parent, child]) == parent
    except ValueError:
        return False


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _config_path(obj):
    if isinstance(obj, dict):
        for key in ("folder_path", "dataset_path"):
            value = obj.get(key)
            if value:
                return value
        return None
    for key in ("folder_path", "dataset_path"):
        value = getattr(obj, key, None)
        if value:
            return value
    return None


def _iter_configs(obj, seen=None):
    """Yield configs reachable from a dataset/config/loader object."""
    if obj is None:
        return
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _iter_configs(value, seen)
        return
    if isinstance(obj, dict):
        yield obj
        return
    yield obj
    # A prepared AiToolkitDataset keeps its first source on dataset_config and
    # any pooled sources on source_configs.  Including both lets the root check
    # see the admitted folder(s), not only the per-file list.
    for attr in ("source_configs", "dataset_config", "datasets", "dataset"):
        value = getattr(obj, attr, None)
        if value is not None:
            yield from _iter_configs(value, seen)


def _iter_dataset_objects(obj, seen=None):
    """Yield actual dataset objects with admitted file lists, if present."""
    if obj is None:
        return
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _iter_dataset_objects(value, seen)
        return
    if isinstance(obj, dict):
        return
    if hasattr(obj, "file_list"):
        yield obj
    for attr in ("datasets", "dataset", "source_configs"):
        value = getattr(obj, attr, None)
        if value is not None:
            yield from _iter_dataset_objects(value, seen)


def _file_identity(path):
    """Stable file identity separate from path equality.

    Canonical path equality catches symlink aliases.  A hardlink has a different
    canonical path, so it is detected by the OS file identity instead.  Inode 0
    is not a usable identity on every filesystem; return None rather than making
    unrelated files compare equal through a sentinel (0, 0).
    """
    if path is None:
        return None
    try:
        st = os.stat(os.fspath(path))
    except (TypeError, ValueError, OSError):
        return None
    inode = int(getattr(st, "st_ino", 0))
    if inode == 0:
        return None
    return int(st.st_dev), inode


def _collect_files_from_root(root):
    """All regular files below a validation root, including JSON dataset keys."""
    canonical = _canonical_path(root)
    if canonical is None:
        return []
    if os.path.isfile(canonical):
        if canonical.lower().endswith(".json"):
            try:
                with open(canonical, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    paths = [p for p in data.keys() if isinstance(p, str) and os.path.isfile(p)]
                    if paths:
                        return paths
            except Exception:
                pass
        return [canonical]
    if not os.path.isdir(canonical):
        return []
    files = []
    for dirpath, dirnames, filenames in os.walk(canonical):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            full = os.path.join(dirpath, name)
            if os.path.isfile(full):
                files.append(full)
    return files


def _check_validation_disjoint(trainer):
    """Refuse held-out validation roots that overlap training data.

    Root containment is checked by path components, so a sibling with a shared
    string prefix (``data`` vs ``data2``) is accepted.  The actual admitted
    training file list and actual validation files are also compared by
    canonical path and by device/inode, which catches symlink aliases and
    hardlinks under disjoint roots.

    This is a conservative refusal.  A validation folder nested inside a
    training root is refused even when the training walker would skip hidden
    directories or ``_controls``; the layout is ambiguous and cannot be shown
    safe without an exact walk.  A refusal does not by itself prove every file
    in that layout leaked.
    """
    cfg = getattr(getattr(trainer, "train_config", None), "validation_config", None)
    if cfg is None:
        return
    validation_datasets = _as_list(getattr(cfg, "datasets", None))
    if not validation_datasets:
        return

    training_roots = []
    for attr in ("datasets", "datasets_reg", "dataset_configs"):
        for obj in _as_list(getattr(trainer, attr, None)):
            for config in _iter_configs(obj):
                path = _config_path(config)
                if path:
                    training_roots.append(path)
                rejected = _raw_value(config, "rejected_path")
                if rejected:
                    training_roots.append(rejected)
    # Some callers keep actual dataset objects on the loader rather than in
    # trainer.datasets; include their admitted file lists as well.
    for attr in ("datasets", "datasets_reg", "data_loader", "data_loader_reg"):
        for obj in _as_list(getattr(trainer, attr, None)):
            for dataset in _iter_dataset_objects(obj):
                for item in _as_list(getattr(dataset, "file_list", None)):
                    path = getattr(item, "path", None) or item
                    if path:
                        training_roots.append(path)
                    mate = getattr(item, "rejected_path", None)
                    if mate:
                        training_roots.append(mate)

    validation_roots = []
    for config in _iter_configs(validation_datasets):
        path = _config_path(config)
        if path:
            validation_roots.append(path)
        rejected = _raw_value(config, "rejected_path")
        if rejected:
            validation_roots.append(rejected)

    canonical_training = []
    for path in training_roots:
        canonical = _canonical_path(path)
        if canonical and canonical not in canonical_training:
            canonical_training.append(canonical)
    canonical_validation = []
    for path in validation_roots:
        canonical = _canonical_path(path)
        if canonical and canonical not in canonical_validation:
            canonical_validation.append(canonical)

    for training in canonical_training:
        for validation in canonical_validation:
            if _path_contains(training, validation) or _path_contains(validation, training):
                raise ValueError(
                    "Validation data overlaps training data: "
                    f"training root {training!r} and validation root {validation!r} "
                    "are equal or nested. Put validation files outside the training tree."
                )

    # Actual-file overlap catches canonical aliases and hardlinks that root
    # comparison cannot see when the roots are disjoint.
    training_paths = set()
    training_identities = set()
    for attr in ("datasets", "datasets_reg", "data_loader", "data_loader_reg"):
        for obj in _as_list(getattr(trainer, attr, None)):
            for dataset in _iter_dataset_objects(obj):
                for item in _as_list(getattr(dataset, "file_list", None)):
                    path = getattr(item, "path", None) or item
                    canonical = _canonical_path(path)
                    if canonical:
                        training_paths.add(canonical)
                    identity = _file_identity(path)
                    if identity:
                        training_identities.add(identity)
    for validation in canonical_validation:
        for path in _collect_files_from_root(validation):
            canonical = _canonical_path(path)
            identity = _file_identity(path)
            if canonical in training_paths or (identity is not None and identity in training_identities):
                raise ValueError(
                    "Validation data overlaps training data: "
                    f"validation file {canonical!r} is also an admitted training file"
                )


def setup(trainer):
    _check_validation_disjoint(trainer)
    from toolkit.data_loader import get_dataloader_from_datasets

    cfg = trainer.train_config.validation_config
    paired_validation = _is_paired_validation(cfg)
    if paired_validation:
        _validate_paired_validation_config(trainer, cfg)
    if trainer.train_config.train_text_encoder or getattr(trainer.sd, 'is_edit', False) or any(
            _raw_value(d, 'do_i2v', False) or _raw_value(d, 'control_path') for d in cfg.datasets):
        raise ValueError(
            'Cached held-out validation requires a frozen text encoder and '
            'plain image/video conditioning'
        )

    configs = []
    for raw in cfg.datasets:
        data = dict(raw) if isinstance(raw, dict) else vars(raw).copy()
        data.update(
            random_crop=False,
            shuffle_tokens=False,
            caption_dropout_rate=0,
            flip_x=False,
            flip_y=False,
            augments=[],
            augmentations=[],
            num_workers=0,
            cache_latents=not paired_validation,
            cache_latents_to_disk=False,
        )
        if paired_validation:
            # Encode only the selected held-out pairs, then keep their tensors on CPU.
            data.update(cache_text_embeddings=False, cache_tensors_to_disk=False,
                        random_crop_percent=0)
        if not paired_validation:
            if Path(data['folder_path']).resolve() in {
                    Path(d.folder_path).resolve() for d in trainer.datasets}:
                raise ValueError('Training and validation folders must be separate')
        configs.append(data)

    cache = []
    device, dtype = trainer.device_torch, trainer.sd.torch_dtype
    trainer.sd.set_device_state_preset('cache_latents')
    try:
        with torch.no_grad(), fixed_rng(cfg.seed, device), disable_noise_augmentation(trainer):
            loader = get_dataloader_from_datasets(configs, 1, trainer.sd)
            for batch in islice(loader, cfg.max_batches):
                # Validation uses explicit fixed sigmas below, never advances curriculum.
                batch._clean_share_validation = True
                if paired_validation:
                    from toolkit.diffusion_dpo import paired_batch

                    paired = paired_batch(batch)
                    _, noise, timesteps, prompts, _ = trainer.process_general_training_batch(
                        paired, shared_pair_noise=True
                    )
                    _validate_paired_batch_cache(batch, paired, prompts, noise, timesteps)
                    trainer.sd.text_encoder_to(device)
                    embeds = trainer.sd.encode_prompt(prompts).detach().to(
                        'cpu', dtype=torch.float32
                    )
                    pair_count = len(batch.file_items)
                    _embedding_batch_size(embeds, pair_count)
                    cache.append({
                        'latents': paired.latents.detach().to('cpu', dtype=torch.float32),
                        'noise': noise[:pair_count].detach().to('cpu', dtype=torch.float32),
                        'embeds': embeds,
                        'chosen_image_ids': [
                            _relative_chosen_image_id(item) for item in batch.file_items
                        ],
                    })
                else:
                    _, noise, _, prompts, _ = trainer.process_general_training_batch(batch)
                    trainer.sd.text_encoder_to(device)
                    embeds = trainer.sd.encode_prompt(prompts).detach().to(
                        'cpu', dtype=torch.float32
                    )
                    batch.latents = batch.latents.detach().to('cpu', dtype=torch.float32)
                    noise = noise.detach().to('cpu', dtype=torch.float32)
                    cache.append((batch, noise, embeds))
    finally:
        trainer.sd.restore_device_state()
    if not cache:
        raise ValueError('Validation dataset has no usable batches')
    trainer._validation_cache = {'native_batches': cache}
    if paired_validation:
        trainer._validation_cache['paired'] = True


def _validate_paired(trainer, cfg, phase, device, dtype):
    from toolkit.diffusion_dpo import (
        prediction_errors,
        preference_logits,
        reference_errors,
    )
    from toolkit.prompt_utils import concat_prompt_embeds

    cache = list(trainer._validation_cache.get('native_batches') or [])
    if not cache:
        raise ValueError('Validation dataset has no usable batches')
    if not getattr(cfg, 'validation_sigmas', None):
        raise ValueError('validation_sigmas must contain at least one sigma')
    for sigma in cfg.validation_sigmas:
        if not 0 <= sigma <= 1:
            raise ValueError('Validation sigmas must be in [0,1]')

    network = trainer.network
    if network is None:
        raise ValueError('Paired DPO validation requires a trainable LoRA network')
    was_training = trainer.sd.unet.training
    old_multiplier, old_active = network.multiplier, network.is_active
    optimizer = getattr(trainer, 'optimizer', None)
    optimizer_ctx = evaluation_optimizer(optimizer) if optimizer is not None else nullcontext()
    results = []
    global_pair_index = 0
    try:
        trainer.sd.unet.eval()
        network.multiplier = 1.0
        with torch.no_grad(), optimizer_ctx, network:
            for batch_index, cached in enumerate(cache):
                pair_latents_cpu = cached['latents']
                noise_cpu = cached['noise']
                embeds_cpu = cached['embeds']
                chosen_ids = cached['chosen_image_ids']
                pair_count = len(chosen_ids)
                if pair_latents_cpu.shape[0] != pair_count * 2:
                    raise ValueError('Cached DPO validation latents are not 2 x pair count')
                if noise_cpu.shape[0] == pair_count * 2:
                    if not torch.equal(noise_cpu[:pair_count], noise_cpu[pair_count:]):
                        raise ValueError('Cached DPO validation noise arms are not shared')
                    noise_cpu = noise_cpu[:pair_count]
                elif noise_cpu.shape[0] != pair_count:
                    raise ValueError('Cached DPO validation noise is not one value per pair')

                batch_start = global_pair_index
                global_pair_index += pair_count
                for sigma in cfg.validation_sigmas:
                    with fixed_rng(getattr(cfg, 'seed', 42) + batch_index, device):
                        # Every pair gets the same cached noise and timestep on
                        # both arms, and a fresh batch view leaves the cache
                        # reusable for the next sigma/pass.
                        latents = pair_latents_cpu.to(device, dtype=dtype).clone()
                        noise = noise_cpu.to(device, dtype=dtype)
                        noise_pair = torch.cat((noise, noise), dim=0)
                        timesteps = _timesteps_for_sigma(
                            trainer.sd, sigma, pair_count * 2, device
                        )
                        noisy = trainer.sd.add_noise(latents, noise_pair, timesteps).detach()

                        target = _loss_target(trainer.sd, noise_pair, SimpleNamespace(latents=latents), timesteps)
                        embeds = embeds_cpu.clone().to(device, dtype=dtype)
                        conditioning = concat_prompt_embeds([embeds, embeds])

                        reference = reference_errors(
                            trainer, noisy, timesteps, conditioning, target
                        )
                        policy_pred = trainer.predict_noise(noisy, timesteps, conditioning)
                        policy = prediction_errors(policy_pred, target)
                        margins = preference_logits(
                            policy, reference, trainer.train_config.dpo_beta
                        )
                        losses = -F.logsigmoid(margins)

                        for pair_index in range(pair_count):
                            rejected_index = pair_count + pair_index
                            chosen_gain = reference[pair_index] - policy[pair_index]
                            rejected_gain = reference[rejected_index] - policy[rejected_index]
                            strict_correct = bool(margins[pair_index].item() > 0)
                            results.append({
                                'item': batch_start + pair_index,
                                'batch': batch_index,
                                'pair_in_batch': pair_index,
                                'chosen_image_id': chosen_ids[pair_index],
                                'relative_chosen_image_id': chosen_ids[pair_index],
                                'sigma': float(sigma),
                                'loss': float(losses[pair_index].item()),
                                'dpo_loss': float(losses[pair_index].item()),
                                'margin': float(margins[pair_index].item()),
                                'scaled_margin': float(margins[pair_index].item()),
                                'accuracy': float(strict_correct),
                                'strict_preference_correct': strict_correct,
                                'policy_chosen_error': float(policy[pair_index].item()),
                                'policy_rejected_error': float(policy[rejected_index].item()),
                                'reference_chosen_error': float(reference[pair_index].item()),
                                'reference_rejected_error': float(reference[rejected_index].item()),
                                'chosen_gain': float(chosen_gain.item()),
                                'rejected_gain': float(rejected_gain.item()),
                            })

        trainer.additional_logs['val/loss'] = float(np.mean([r['dpo_loss'] for r in results]))
        trainer.additional_logs['val/dpo/loss'] = float(np.mean([r['dpo_loss'] for r in results]))
        trainer.additional_logs['val/dpo/margin'] = float(np.mean([r['margin'] for r in results]))
        trainer.additional_logs['val/dpo/accuracy'] = float(np.mean(
            [r['strict_preference_correct'] for r in results]
        ))
        for sigma in cfg.validation_sigmas:
            sigma_rows = [r for r in results if r['sigma'] == float(sigma)]
            suffix = f'/sigma_{sigma:g}'
            trainer.additional_logs[f'val/dpo/loss{suffix}'] = float(
                np.mean([r['dpo_loss'] for r in sigma_rows])
            )
            trainer.additional_logs[f'val/dpo/margin{suffix}'] = float(
                np.mean([r['margin'] for r in sigma_rows])
            )
            trainer.additional_logs[f'val/dpo/accuracy{suffix}'] = float(np.mean(
                [r['strict_preference_correct'] for r in sigma_rows]
            ))

        if getattr(getattr(trainer, 'accelerator', None), 'is_main_process', True):
            path = Path(trainer.save_root) / 'validation.jsonl'
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'step': trainer.step_num, 'phase': phase, 'raw_metrics': results}) + '\n')
    finally:
        network.multiplier, network.is_active = old_multiplier, old_active
        trainer.sd.unet.train(was_training)


def validate(trainer, phase='periodic'):
    cfg = trainer.train_config.validation_config
    cache = getattr(trainer, '_validation_cache', None) or {}
    paired_cache = bool(cache.get('paired')) if isinstance(cache, dict) else False
    if _is_paired_validation(cfg) or paired_cache:
        if getattr(trainer.train_config, 'loss_type', None) != 'dpo':
            raise ValueError(
                "Validation datasets[].rejected_path requires train.loss_type: dpo"
            )
        if not cache or not cache.get('native_batches'):
            raise ValueError('Paired validation cache was not initialized by setup()')
        _validate_paired(
            trainer, cfg, phase, trainer.device_torch, trainer.sd.torch_dtype
        )
        return

    device, dtype = trainer.device_torch, trainer.sd.torch_dtype
    network = trainer.network
    was_training = trainer.sd.unet.training
    old_multiplier = network.multiplier if network is not None else None
    # `with network` below forces is_active True then False; give back the entry value instead
    was_active = network.is_active if network is not None else None
    results = []
    try:
        trainer.sd.unet.eval()
        if network is not None:
            network.multiplier = 1.0
        from toolkit.stable_diffusion_model import BlankNetwork
        with torch.no_grad(), evaluation_optimizer(trainer.optimizer), network if network is not None else BlankNetwork():
            for item, (cached, noise_cpu, embeds_cpu) in enumerate(trainer._validation_cache['native_batches']):
                for sigma in cfg.validation_sigmas:
                    if not 0 <= sigma <= 1:
                        raise ValueError('Validation sigmas must be in [0,1]')
                    with fixed_rng(getattr(cfg, 'seed', 42) + item, device):
                        batch = copy.copy(cached)
                        batch.latents = cached.latents.clone().to(device, dtype=dtype)
                        noise = noise_cpu.to(device, dtype=dtype)
                        timesteps = _timesteps_for_sigma(
                            trainer.sd, sigma, batch.latents.shape[0], device
                        )
                        noisy = trainer.sd.add_noise(batch.latents, noise, timesteps).detach()
                        pred = trainer.sd.predict_noise(
                            latents=noisy,
                            conditional_embeddings=embeds_cpu.clone().to(device, dtype=dtype),
                            timestep=timesteps,
                            guidance_scale=1.0,
                            batch=batch,
                            guidance_embedding_scale=trainer.train_config.cfg_scale,
                            bypass_guidance_embedding=trainer.train_config.bypass_guidance_embedding,
                        )
                        target = _loss_target(trainer.sd, noise, batch, timesteps)
                        row = {
                            'sigma': sigma,
                            'item': item,
                            'video_mse': float((pred.float() - target.float()).square().mean()),
                        }
                        if hasattr(pred, 'get') and pred.get('audio') is not None:
                            row['audio_mse'] = float(
                                (pred.get('audio').float() - pred.get('audio_target').float()).square().mean()
                            )
                        results.append(row)
        trainer.additional_logs['val/loss'] = float(np.mean([r['video_mse'] for r in results]))
        for sigma in cfg.validation_sigmas:
            trainer.additional_logs[f'val/raw_mse/sigma_{sigma:g}'] = float(np.mean(
                [r['video_mse'] for r in results if r['sigma'] == sigma]
            ))
        audio = [r['audio_mse'] for r in results if 'audio_mse' in r]
        if audio:
            trainer.additional_logs['val/audio_mse'] = float(np.mean(audio))
        path = Path(trainer.save_root) / 'validation.jsonl'
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({
                'step': trainer.step_num,
                'phase': phase,
                'raw_metrics': results,
            }) + '\n')
    finally:
        if network is not None:
            network.multiplier = old_multiplier
            network.is_active = was_active
        trainer.sd.unet.train(was_training)
