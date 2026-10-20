"""RAWR-only checkpoint continuation helpers; no trainer or model imports."""
import copy
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


def add_state(optimizer_state, scheduler, config, trainer_step):
    scheduler = underlying_scheduler(scheduler)
    if not is_rawr(scheduler):
        if config is not None:
            raise TypeError('Configured RAWR has an unsupported scheduler wrapper; refusing to omit checkpoint metadata')
        return optimizer_state
    result = dict(optimizer_state)
    result[KEY] = {'version': 1, 'config': copy.deepcopy(config),
                   'trainer_step': int(trainer_step),
                   'scheduler': copy.deepcopy(scheduler.state_dict()),
                   'groups': capture(optimizer_state)['groups']}
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
        if payload.get('version') != 1:
            raise ValueError('Unsupported RAWR checkpoint version')
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
