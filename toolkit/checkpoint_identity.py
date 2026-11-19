"""What a checkpoint's optimizer state belongs to (docs/STATE_MACHINES.md, OPUS-RES-02, OPUS-PARAM-ID-01, OPUS-EMA-01).

PyTorch's Optimizer.load_state_dict maps saved state to parameters by position only; names never take part. So each
save stamps one generation id into the model checkpoint's metadata and an identity record into optimizer.pt: the
generation, the completed step, the optimizer class, the ordered (name, shape, dtype) list of every parameter group
and the base-model recipe; artifacts whose format has no metadata (embedding files) are recorded by hash instead.
Resume compares the record with every resumed artifact and the current run *before* loading, and refuses a mismatch
instead of attaching moments to other weights. A match certifies the pairing, not a bit-exact continuation. With
EMA, optimizer.pt also carries the raw training weights and the EMA shadow, since the exported checkpoint holds the
EMA weights.
"""
import hashlib
import json
import os
import uuid
import warnings

import torch

KEY = '_aitk_resume_identity'
EMA_KEY = '_aitk_ema_resume'
META_GENERATION = 'aitk_resume_generation'
VERSION = 1

# What the base model was built from. Compared warn-only: paths legitimately move between machines, and hashing
# multi-GB weights at every save is not worth it.
_RECIPE_FIELDS = ('name_or_path', 'arch', 'dtype', 'quantize', 'qtype', 'quantize_te', 'qtype_te',
                  'te_name_or_path', 'model_kwargs', 'assistant_lora_path', 'accuracy_recovery_adapter',
                  'refiner_name_or_path', 'lora_path')


def new_generation():
    return uuid.uuid4().hex


def unwrap_optimizer(optimizer):
    from accelerate.optimizer import AcceleratedOptimizer
    while isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer
    return optimizer


def optimizer_class(optimizer):
    cls = type(unwrap_optimizer(optimizer))
    return f'{cls.__module__}.{cls.__qualname__}'


def parameter_names(modules):
    """id(param) -> dotted name, from (prefix, module or list of modules) pairs; the first name wins."""
    names = {}
    for prefix, module in modules:
        group = module if isinstance(module, (list, tuple)) else [module]
        for i, m in enumerate(group):
            if not isinstance(m, torch.nn.Module):
                continue
            label = prefix if len(group) == 1 else f'{prefix}{i}'
            for name, p in m.named_parameters(remove_duplicate=False):
                names.setdefault(id(p), f'{label}.{name}')
    return names


def parameter_manifest(optimizer, names):
    """Ordered (name, shape, dtype) per param group; unnamed parameters keep their position as their name."""
    return [[[names.get(id(p), f'?{gi}.{pi}'), list(p.shape), str(p.dtype).removeprefix('torch.')]
             for pi, p in enumerate(group['params'])]
            for gi, group in enumerate(unwrap_optimizer(optimizer).param_groups)]


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


def base_recipe(model_config):
    if model_config is None:
        return None
    return _jsonable({k: getattr(model_config, k, None) for k in _RECIPE_FIELDS})


def _module_attributes(label, owner):
    """(label.attr, module or list of modules) for every module an object holds, in a fixed (sorted) order."""
    found = []
    for attr, value in sorted(getattr(owner, '__dict__', {}).items()):
        if isinstance(value, torch.nn.Module) or (isinstance(value, (list, tuple)) and value
                                                  and all(isinstance(v, torch.nn.Module) for v in value)):
            found.append((f'{label}.{attr}', value))
    return found


def trainer_parameter_names(trainer):
    """Names for every parameter a module of the trainer or its model owns. The usual producers come first, so their
    names are stable (network, adapter, decorator; the model, text encoders (the embedding's token vectors) and
    refiner); any other module the trainer or model holds names the rest."""
    sd = getattr(trainer, 'sd', None)
    modules = [('network', getattr(trainer, 'network', None)), ('adapter', getattr(trainer, 'adapter', None)),
               ('decorator', getattr(trainer, 'decorator', None))]
    if sd is not None:
        modules += [('unet', getattr(sd, 'unet', None)), ('text_encoder', getattr(sd, 'text_encoder', None)),
                    ('refiner', getattr(sd, 'refiner_unet', None))]
        modules += _module_attributes('sd', sd)
    modules += _module_attributes('trainer', trainer)
    return parameter_names(modules)


def positional(manifest):
    """(group, position, entry) of parameters no module names: they could be paired only by position."""
    return [(gi, pi, entry) for gi, group in enumerate(manifest) for pi, entry in enumerate(group)
            if entry[0].startswith('?')]


def require_named(manifest):
    """Refuse, before training, optimizer parameters a save could not name (docs/STATE_MACHINES.md, PARAM-ID-01)."""
    unnamed = positional(manifest)
    if unnamed:
        gi, pi, entry = unnamed[0]
        raise ValueError(f'{len(unnamed)} trainable parameters (e.g. group {gi} parameter {pi}, shape {entry[1]}) '
                         f'belong to no module of this trainer or its model, so a checkpoint could pair their '
                         f'optimizer state only by position, and a resume could attach it to other weights. Train '
                         f'them as part of a module (network, adapter, decorator), or report this configuration')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record(trainer, generation, step, content=None):
    """The identity record a save writes into optimizer.pt. The manifest is fixed once the optimizer exists.
    `content`: basename -> sha256 of this save's artifacts whose format has no metadata for the generation
    (embedding files)."""
    manifest = getattr(trainer, '_resume_manifest', None)
    if manifest is None:
        manifest = parameter_manifest(trainer.optimizer, trainer_parameter_names(trainer))
    return {'version': VERSION, 'generation': generation, 'step': int(step),
            'optimizer': optimizer_class(trainer.optimizer), 'parameters': manifest,
            'base_model': base_recipe(getattr(trainer, 'model_config', None)), 'content': dict(content or {})}


def ema_options(ema):
    return {'decay': ema.decay, 'use_num_updates': ema.num_updates is not None, 'use_feedback': ema.use_feedback,
            'feedback_rate': ema.feedback_rate, 'param_multiplier': ema.param_multiplier}


def ema_record(ema):
    """Raw weights (stored by ema.eval() for the save) and the shadow, both in the optimizer's parameter order."""
    if ema is None or ema.collected_params is None:
        return None
    return {'raw': [p.detach().cpu().clone() for p in ema.collected_params],
            'shadow': [p.detach().cpu().clone() for p in ema.shadow_params],
            'num_updates': ema.num_updates, 'options': ema_options(ema)}


def validate_ema_record(saved, params):
    """Check the whole EMA record against the optimizer's parameters before anything is restored from it."""
    if not isinstance(saved, dict) or not isinstance(saved.get('raw'), list) or not isinstance(saved.get('shadow'), list):
        raise ValueError('The EMA state saved with this checkpoint is malformed')
    for kind in ('raw', 'shadow'):
        values = saved[kind]
        if len(values) != len(params):
            raise ValueError(f'The saved EMA {kind} weights cover {len(values)} parameters, the optimizer has '
                             f'{len(params)}')
        for i, (value, p) in enumerate(zip(values, params)):
            if not torch.is_tensor(value) or value.shape != p.shape or value.dtype != p.dtype:
                raise ValueError(f'The saved EMA {kind} weight {i} does not match its parameter '
                                 f'({getattr(value, "shape", None)}/{getattr(value, "dtype", None)} vs '
                                 f'{tuple(p.shape)}/{p.dtype})')
    num_updates = saved.get('num_updates')
    if num_updates is not None and (type(num_updates) is not int or num_updates < 0):
        raise ValueError('The saved EMA update count is malformed')
    options = saved.get('options')
    if not isinstance(options, dict):
        raise ValueError('The saved EMA options are malformed')


def _first_difference(saved, current):
    if len(saved) != len(current):
        return f'{len(saved)} parameter groups were saved, this run has {len(current)}'
    for gi, (a, b) in enumerate(zip(saved, current)):
        if len(a) != len(b):
            return f'group {gi} saved {len(a)} parameters, this run has {len(b)}'
        for pi, (x, y) in enumerate(zip(a, b)):
            if list(x) != list(y):
                return f'group {gi} parameter {pi} was saved as {tuple(x)}, this run has {tuple(y)}'
    return None


def validate(identity, optimizer_path):
    ok = (isinstance(identity, dict) and identity.get('version') == VERSION
          and isinstance(identity.get('generation'), str) and len(identity['generation']) == 32
          and type(identity.get('step')) is int and isinstance(identity.get('optimizer'), str)
          and isinstance(identity.get('parameters'), list) and isinstance(identity.get('content', {}), dict))
    if not ok:
        version = identity.get('version') if isinstance(identity, dict) else None
        raise ValueError(f'{optimizer_path} carries a save record this version cannot read (version {version}); '
                         f'resume it with the version that wrote it, or move it out of the folder to continue with '
                         f'fresh optimizer state')


def belongs(identity, path, generation):
    """Whether the artifact at path (with its metadata generation, or None) is from the save identity describes:
    the same generation, or, for formats without metadata, the content hash recorded at that save."""
    if generation is not None:
        return generation == identity['generation']
    digest = identity.get('content', {}).get(os.path.basename(path))
    return digest is not None and os.path.isfile(path) and sha256(path) == digest


def check(identity, current, artifacts, step, check_step, path, optimizer_path):
    """Refuse an optimizer state that does not belong to every resumed artifact and this run's parameters.
    artifacts: [(path, metadata generation or None)] of everything this resume loaded."""
    validate(identity, optimizer_path)
    move = (f'Move {optimizer_path} out of the folder to continue {path} with fresh optimizer state, or restore the '
            f'checkpoint it was saved with')
    unnamed = positional(identity['parameters'])
    if unnamed:
        gi, pi, entry = unnamed[0]
        raise ValueError(f'{optimizer_path} records {len(unnamed)} parameters only by position (no module name, e.g. '
                         f'group {gi} parameter {pi}, shape {entry[1]}), so its optimizer state cannot be shown to '
                         f'belong to the same weights. {move}')
    if not artifacts:
        raise ValueError(f'{optimizer_path} has no resumed checkpoint to belong to. {move}')
    for artifact, generation in artifacts:
        if not belongs(identity, artifact, generation):
            why = ('a different save' if generation is not None else
                   'its format carries no save record, and it is not the file saved with this state')
            raise ValueError(f'{optimizer_path} was saved at step {identity["step"]} and does not belong to '
                             f'{artifact} ({why}). {move}')
    if check_step and identity['step'] != step:
        raise ValueError(f'{optimizer_path} was saved at step {identity["step"]}, but {path} resumes at step '
                         f'{step}. {move}')
    if identity.get('optimizer') != current['optimizer']:
        raise ValueError(f'{optimizer_path} holds {identity.get("optimizer")} state, but this run uses '
                         f'{current["optimizer"]}. {move}')
    difference = _first_difference(identity.get('parameters') or [], current['parameters'])
    if difference is not None:
        raise ValueError(f'{optimizer_path} was saved for different trainable parameters ({difference}); its moments '
                         f'would be applied to other weights. Resume with the original network/model settings. '
                         f'{move}')
    saved_recipe, recipe = identity.get('base_model'), current.get('base_model')
    if saved_recipe is not None and recipe is not None and saved_recipe != recipe:
        changed = sorted(k for k in set(saved_recipe) | set(recipe) if saved_recipe.get(k) != recipe.get(k))
        warnings.warn(f'The base model settings changed since {path} was saved ({", ".join(changed)}); the resume '
                      f'cannot be certified as an exact continuation', RuntimeWarning, stacklevel=2)


def read_optimizer_identity(path):
    """optimizer.pt's save record, without reading its tensors; None if absent, legacy, unreadable or malformed."""
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except Exception:
        return None
    try:
        identity = state.get(KEY) if isinstance(state, dict) else None
        try:
            validate(identity, path)
        except ValueError:
            return None
        return {'generation': identity['generation'], 'content': dict(identity.get('content', {}))}
    finally:
        # mapped storages pin the file on Windows (the next save replaces it); drop them now
        del state


def read_optimizer_generation(path):
    identity = read_optimizer_identity(path)
    return None if identity is None else identity['generation']


def read_checkpoint_generation(path):
    """aitk_resume_generation of a safetensors file or a Diffusers folder's aitk_meta.yaml; None if absent."""
    try:
        if os.path.isdir(path):
            meta_path = os.path.join(path, 'aitk_meta.yaml')
            if not os.path.exists(meta_path):
                return None
            import yaml
            with open(meta_path, encoding='utf-8') as f:
                meta = yaml.safe_load(f) or {}
            value = meta.get(META_GENERATION)
        elif path.endswith('.safetensors'):
            from safetensors import safe_open
            with safe_open(path, framework='pt') as f:
                value = (f.metadata() or {}).get(META_GENERATION)
        else:
            return None
    except Exception:
        return None
    if isinstance(value, str) and value.startswith('"'):
        value = json.loads(value)
    return value if isinstance(value, str) else None
