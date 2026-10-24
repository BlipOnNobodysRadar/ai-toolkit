"""RAWR-only checkpoint continuation helpers; no trainer or model imports."""
import copy
import os
import tempfile
import warnings

KEY = '_aitk_rawr_checkpoint'
FIELDS = ('lr', 'warmup_steps', 'current_cycle_max_steps', 'min_lr',
          'current_cycle', 'current_cycle_step', 'initial_lr', 'current_max_lr')


def completed_step(trainer_step, completed_update_step, config):
    """RAWR metadata counts completed updates; other schedulers are unchanged."""
    if config is not None and completed_update_step is not None:
        return int(completed_update_step)
    return int(trainer_step)


def is_rawr_name(name):
    if not isinstance(name, str):
        return False
    name = name.strip().lower().removeprefix('easy:')
    dotted = 'rexannealingwarmrestarts.rexannealingwarmrestarts'
    return name in ('rawr', 'rexannealingwarmrestarts', dotted) or name.endswith('.' + dotted)


def underlying_scheduler(scheduler):
    # The trainer's prepare_accelerator() uses this specific supported wrapper.
    # Do not follow arbitrary objects' scheduler/module attributes.
    from accelerate.scheduler import AcceleratedScheduler
    if isinstance(scheduler, AcceleratedScheduler):
        return scheduler.scheduler
    return scheduler


def is_rawr(scheduler):
    scheduler = underlying_scheduler(scheduler)
    return any(cls.__name__ == 'RexAnnealingWarmRestarts'
               and cls.__module__.rsplit('.', 1)[-1] == 'RexAnnealingWarmRestarts'
               for cls in type(scheduler).__mro__)


def configuration(optimizer, total_steps, scheduler_params):
    # Compare declared settings conservatively. Even equivalent spelling changes
    # may require review; silently ignoring an LR/config change is unsafe.
    params = copy.deepcopy(dict(scheduler_params))
    params.pop('total_iters', None)
    params.pop('total_steps', None)
    return {'total_steps': int(total_steps), 'scheduler_params': params,
            'configured_groups': [{k: copy.deepcopy(g[k]) for k in FIELDS if k in g}
                                  for g in optimizer.param_groups]}


def capture(optimizer_state):
    # Capture BEFORE the trainer resets LR and the constructor advances groups.
    if KEY in optimizer_state and not isinstance(optimizer_state[KEY], dict):
        raise ValueError('Malformed RAWR checkpoint payload')
    if not isinstance(optimizer_state.get('param_groups'), list) or not all(
            isinstance(g, dict) for g in optimizer_state['param_groups']):
        raise ValueError('Malformed RAWR optimizer parameter groups')
    return {'groups': [{k: copy.deepcopy(g[k]) for k in FIELDS if k in g}
                       for g in optimizer_state['param_groups']],
            'payload': copy.deepcopy(optimizer_state.get(KEY))}


def add_state(optimizer_state, scheduler, config, trainer_step, training_state=None):
    scheduler = underlying_scheduler(scheduler)
    if not is_rawr(scheduler):
        if config is not None:
            raise TypeError('Configured RAWR has an unsupported scheduler wrapper; refusing to omit checkpoint metadata')
        return optimizer_state
    result = dict(optimizer_state)
    result[KEY] = {'version': 2 if training_state is not None else 1, 'config': copy.deepcopy(config),
                   'trainer_step': int(trainer_step),
                   'scheduler': copy.deepcopy(scheduler.state_dict()),
                   'groups': capture(optimizer_state)['groups']}
    if training_state is not None:
        result[KEY]['training_state'] = training_state
    return result


def restore(scheduler, captured, config, trainer_step):
    """Undo construction and restore RAWR; return whether startup must be skipped.

    Legacy optimizer-only state can continue its retained group trajectory under
    the CURRENT scheduler settings, but cannot certify the original settings,
    scheduler bookkeeping, or pairing with the selected model checkpoint.
    """
    scheduler = underlying_scheduler(scheduler)
    if captured is None or not is_rawr(scheduler):
        return False
    groups = captured['groups']
    if len(groups) != len(scheduler.optimizer.param_groups):
        raise ValueError('RAWR checkpoint parameter-group count changed')
    for group in groups:
        if any(k not in group for k in FIELDS):
            raise ValueError('RAWR resume requires complete optimizer cycle fields; start a new run')
        if not (0 <= group['current_cycle_step'] < group['current_cycle_max_steps']
                and 0 <= group['warmup_steps'] < group['current_cycle_max_steps']
                and group['current_cycle'] >= 0):
            raise ValueError('Invalid RAWR checkpoint cycle position')
    payload = captured['payload']
    if payload is not None:
        if payload.get('version') not in (1, 2):
            raise ValueError('Unsupported RAWR checkpoint version')
        if payload['version'] == 2 and not isinstance(payload.get('training_state'), dict):
            raise ValueError('Missing RAWR training checkpoint state')
        if payload.get('config') != config:
            raise ValueError('RAWR resume configuration changed; use the saved configuration or start a new run')
        if payload.get('trainer_step') != int(trainer_step):
            raise ValueError('RAWR optimizer checkpoint step does not match selected training metadata')
        if payload.get('groups') != groups:
            raise ValueError('RAWR optimizer groups do not match scheduler checkpoint metadata')
        state = payload.get('scheduler')
        if not isinstance(state, dict):
            raise ValueError('Malformed RAWR scheduler checkpoint state')
        if state.get('base_lrs') != [g['initial_lr'] for g in groups] or state.get('_last_lr') != [g['lr'] for g in groups]:
            raise ValueError('RAWR scheduler LR metadata does not match optimizer groups')
        for name in ('d', 'gamma', 'cycle_multiplier'):
            if state.get(name) != getattr(scheduler, name):
                raise ValueError('RAWR scheduler settings do not match current configuration')
        scheduler.load_state_dict(copy.deepcopy(state))
    else:
        configured = config['configured_groups']
        if len(configured) != len(groups) or any(
                current['lr'] != saved['initial_lr']
                for current, saved in zip(configured, groups)):
            raise ValueError('Legacy RAWR configured LR differs from saved initial LR; start a new run')
        warnings.warn(
            'Legacy RAWR optimizer-only checkpoint: retaining saved LR/cycle fields '
            'under current d/gamma/cycle_multiplier settings. Original scheduler '
            'settings, bookkeeping, and model/optimizer step pairing are unavailable; '
            f'exact continuation cannot be certified (trainer step {trainer_step}).', RuntimeWarning, stacklevel=2)
        # These diagnostic labels cannot be recovered exactly from old files.
        # RAWR future arithmetic uses optimizer cycle fields, not last_epoch.
        scheduler.last_epoch = int(trainer_step)
        scheduler._step_count = max(2, int(trainer_step) + 2)
        scheduler.base_lrs = [copy.deepcopy(g['initial_lr']) for g in groups]
        scheduler._last_lr = [copy.deepcopy(g['lr']) for g in groups]
    for target, saved in zip(scheduler.optimizer.param_groups, groups):
        target.update(copy.deepcopy(saved))
    scheduler.validate_optimizer()
    return True


def training_state(trainer, checkpoint_step):
    """Serialize unfinished microbatches, with the counter for the NEXT loop.

    Periodic saves precede the host's counter increment. UI/end-step/final
    saves follow it. Completed loop metadata disambiguates those phases.
    RNG, dataset iterators, and auxiliary optimizers are not captured here.
    """
    return {
        'gradient_accumulation_steps': trainer.train_config.gradient_accumulation_steps,
        'gradient_accumulation': trainer.train_config.gradient_accumulation,
        'next_accumulation_step': trainer.grad_accumulation_step + int(checkpoint_step > trainer.step_num),
        'gradients': [[None if p.grad is None else p.grad.detach().cpu().clone()
                       for p in g['params']] for g in trainer.optimizer.param_groups],
    }


def restore_training_state(optimizer, captured, config, trainer_step):
    """Restore after startup zero_grad/Accelerate preparation, before training.

    Old files have no partial gradients. Version-1 completed-loop labels can
    certify a fixed accumulation boundary; legacy labels cannot certify one.
    """
    import torch
    if captured is None:
        return 1
    payload = captured['payload']
    state = payload.get('training_state') if payload is not None else None
    gas = config.gradient_accumulation_steps
    if gas == -1 and trainer_step > 0:
        raise ValueError('RAWR whole-epoch accumulation resume requires dataset position, which is not saved')
    if state is None:
        if gas > 1 and trainer_step > 0:
            if payload is None or int(trainer_step) % gas:
                raise ValueError('RAWR checkpoint has no partial gradients; resume from a known optimizer boundary')
        return 1
    if (state.get('gradient_accumulation_steps') != gas or
            state.get('gradient_accumulation') != config.gradient_accumulation):
        raise ValueError('RAWR gradient accumulation configuration changed')
    counter = state.get('next_accumulation_step')
    if type(counter) is not int or not (1 <= counter <= max(1, gas)):
        raise ValueError('Invalid RAWR accumulation position')
    gradients = state.get('gradients')
    groups = optimizer.param_groups
    if not isinstance(gradients, list) or len(gradients) != len(groups):
        raise ValueError('Invalid RAWR gradient groups')
    # Validate everything before changing any gradients.
    for group, saved in zip(groups, gradients):
        if not isinstance(saved, list) or len(saved) != len(group['params']):
            raise ValueError('Invalid RAWR gradient parameter count')
        for p, grad in zip(group['params'], saved):
            if grad is not None and (not torch.is_tensor(grad) or grad.shape != p.shape or grad.dtype != p.dtype):
                raise ValueError('Invalid RAWR gradient shape or dtype')
    for group, saved in zip(groups, gradients):
        for p, grad in zip(group['params'], saved):
            p.grad = None if grad is None else grad.to(device=p.device).clone()
    return counter


def atomic_save(optimizer_state, path):
    """Replace a complete optimizer file, preserving the previous one on error.

    This is a single-file transaction, not an atomic model/optimizer pair.
    """
    import torch
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.optimizer.', suffix='.tmp',
                                         dir=os.path.dirname(os.path.abspath(path)), delete=False) as f:
            tmp_path = f.name
            torch.save(optimizer_state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)
