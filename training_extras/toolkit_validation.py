"""Fixed held-out batches through the host's native image/video/audio path."""
import copy, json, os, random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from .optim import evaluation_optimizer
from toolkit.noise_augmentation import disable_noise_augmentation

@contextmanager
def fixed_rng(seed, device):
    py, np_state = random.getstate(), np.random.get_state()
    devices = [device] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py); np.random.set_state(np_state)

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
    # Some callers keep actual dataset objects on the loader rather than in
    # trainer.datasets; include their admitted file lists as well.
    for attr in ("datasets", "datasets_reg", "data_loader", "data_loader_reg"):
        for obj in _as_list(getattr(trainer, attr, None)):
            for dataset in _iter_dataset_objects(obj):
                for item in _as_list(getattr(dataset, "file_list", None)):
                    path = getattr(item, "path", None) or item
                    if path:
                        training_roots.append(path)

    validation_roots = []
    for config in _iter_configs(validation_datasets):
        path = _config_path(config)
        if path:
            validation_roots.append(path)

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
    if trainer.train_config.train_text_encoder or getattr(trainer.sd,'is_edit',False) or any(d.get('do_i2v',False) or d.get('control_path') for d in cfg.datasets):
        raise ValueError('Cached held-out validation requires a frozen text encoder and plain image/video conditioning')
    configs = []
    for raw in cfg.datasets:
        data = dict(raw)
        data.update(random_crop=False,shuffle_tokens=False,caption_dropout_rate=0,flip_x=False,
            flip_y=False,augments=[],augmentations=[],num_workers=0,cache_latents=True,
            cache_latents_to_disk=False)
        if Path(data['folder_path']).resolve() in {Path(d.folder_path).resolve() for d in trainer.datasets}:
            raise ValueError('Training and validation folders must be separate')
        configs.append(data)
    cache = []
    device, dtype = trainer.device_torch, trainer.sd.torch_dtype
    trainer.sd.set_device_state_preset('cache_latents')
    try:
        with torch.no_grad(), fixed_rng(cfg.seed, device), disable_noise_augmentation(trainer):
            loader = get_dataloader_from_datasets(configs, 1, trainer.sd)
            for i, batch in enumerate(loader):
                if i >= cfg.max_batches: break
                # Validation uses explicit fixed sigmas below, never advances curriculum.
                batch._clean_share_validation = True
                _, noise, _, prompts, _ = trainer.process_general_training_batch(batch)
                trainer.sd.text_encoder_to(device)
                embeds = trainer.sd.encode_prompt(prompts).detach().to('cpu', dtype=torch.float32)
                batch.latents = batch.latents.detach().to('cpu', dtype=torch.float32)
                noise = noise.detach().to('cpu', dtype=torch.float32)
                cache.append((batch,noise,embeds))
    finally:
        trainer.sd.restore_device_state()
    if not cache: raise ValueError('Validation dataset has no usable batches')
    trainer._validation_cache = {'native_batches':cache}

def validate(trainer, phase='periodic'):
    cfg = trainer.train_config.validation_config
    device, dtype = trainer.device_torch, trainer.sd.torch_dtype
    network = trainer.network
    was_training = trainer.sd.unet.training
    old_multiplier = network.multiplier if network is not None else None
    # `with network` below forces is_active True then False; give back the entry value instead
    was_active = network.is_active if network is not None else None
    results = []
    try:
        trainer.sd.unet.eval()
        if network is not None: network.multiplier = 1.0
        from toolkit.stable_diffusion_model import BlankNetwork
        with torch.no_grad(), evaluation_optimizer(trainer.optimizer), network if network is not None else BlankNetwork():
            for item, (cached, noise_cpu, embeds_cpu) in enumerate(trainer._validation_cache['native_batches']):
                for sigma in cfg.validation_sigmas:
                    if not 0 <= sigma <= 1: raise ValueError('Validation sigmas must be in [0,1]')
                    with fixed_rng(cfg.seed + item, device):
                        batch = copy.copy(cached)
                        batch.latents = cached.latents.clone().to(device, dtype=dtype)
                        noise = noise_cpu.to(device,dtype=dtype)
                        t = float(sigma) * trainer.sd.noise_scheduler.config.num_train_timesteps
                        if not trainer.sd.is_flow_matching: t = min(round(t),trainer.sd.noise_scheduler.config.num_train_timesteps-1)
                        timesteps = torch.full((batch.latents.shape[0],),t,device=device,
                            dtype=torch.float32 if trainer.sd.is_flow_matching else torch.long)
                        noisy = trainer.sd.add_noise(batch.latents,noise,timesteps).detach()
                        pred = trainer.sd.predict_noise(latents=noisy,conditional_embeddings=embeds_cpu.clone().to(device,dtype=dtype),
                            timestep=timesteps,guidance_scale=1.0,batch=batch,
                            guidance_embedding_scale=trainer.train_config.cfg_scale,
                            bypass_guidance_embedding=trainer.train_config.bypass_guidance_embedding)
                        if hasattr(trainer.sd,'get_loss_target'):
                            target = trainer.sd.get_loss_target(noise=noise,batch=batch,timesteps=timesteps)
                        elif trainer.sd.is_flow_matching: target = noise - batch.latents
                        elif trainer.sd.prediction_type == 'v_prediction':
                            target = trainer.sd.noise_scheduler.get_velocity(batch.latents,noise,timesteps)
                        else: target = noise
                        row = {'sigma':sigma,'item':item,'video_mse':float((pred.float()-target.float()).square().mean())}
                        if hasattr(pred,'get') and pred.get('audio') is not None:
                            row['audio_mse'] = float((pred.get('audio').float()-pred.get('audio_target').float()).square().mean())
                        results.append(row)
        trainer.additional_logs['val/loss'] = float(np.mean([r['video_mse'] for r in results]))
        for sigma in cfg.validation_sigmas:
            trainer.additional_logs[f'val/raw_mse/sigma_{sigma:g}'] = float(np.mean([r['video_mse'] for r in results if r['sigma']==sigma]))
        audio = [r['audio_mse'] for r in results if 'audio_mse' in r]
        if audio: trainer.additional_logs['val/audio_mse'] = float(np.mean(audio))
        path = Path(trainer.save_root) / 'validation.jsonl'
        with path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'step':trainer.step_num,'phase':phase,'raw_metrics':results})+'\n')
    finally:
        if network is not None:
            network.multiplier = old_multiplier
            network.is_active = was_active
        trainer.sd.unet.train(was_training)
