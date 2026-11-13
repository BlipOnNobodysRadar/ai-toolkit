"""Opt-in MOSS clean-half timestep curriculum; no torch dependency.

Original implementation of the published Fizgig 8.8.1 schedule, not copied
from Fizgig. Probabilities describe ideal continuous sampling before float
rounding and finite timestep pools.
"""
import copy
from functools import lru_cache
import math

def _logit(x):
    return math.log(x) - math.log1p(-x)


def _cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def clean_share(mean, std, cutoff=1.0, uniform_mix=0.0):
    """Ideal continuous P(sigma < .5) for a truncated/rescaled mixture.

    Assumes ALL sampled timestep indices remain eligible. Index trimming and
    content/style selection can change the resulting training distribution.
    Does not model torch dtype/rounding, finite pools, or the fork's uniform
    draw clamp_min(1e-7) before inverse CDF and post-rescale clamp(max=1).
    """
    if not all(math.isfinite(x) for x in (mean, std, cutoff, uniform_mix)):
        raise ValueError('parameters must be finite')
    if std <= 0 or not 0 < cutoff <= 1 or not 0 <= uniform_mix <= 1:
        raise ValueError('invalid distribution parameters')
    numerator = _cdf((_logit(0.5 * cutoff) - mean) / std)
    denominator = 1.0 if cutoff == 1 else _cdf((_logit(cutoff) - mean) / std)
    if denominator == 0:
        raise ValueError('normal tail underflow; unsupported numerical range')
    return (1 - uniform_mix) * (numerator / denominator) + uniform_mix * 0.5


def mean_for_share(target, std, cutoff=1.0, uniform_mix=0.0):
    clean_share(0.0, std, cutoff, uniform_mix)  # validate distribution
    if not math.isfinite(target) or not uniform_mix / 2 < target < 1 - uniform_mix / 2:
        raise ValueError('target outside strict attainable mixture range')
    center = 0.0 if cutoff == 1 else _logit(cutoff)
    lo, hi = center - 32 * std, center + 32 * std
    if not clean_share(hi, std, cutoff, uniform_mix) <= target <= clean_share(lo, std, cutoff, uniform_mix):
        raise ValueError('target outside supported numerical bracket')
    for _ in range(100):
        mid = (lo + hi) / 2
        if clean_share(mid, std, cutoff, uniform_mix) > target:
            lo = mid
        else:
            hi = mid
    result = (lo + hi) / 2
    if abs(clean_share(result, std, cutoff, uniform_mix) - target) > 1e-10:
        raise ValueError('solver failed target tolerance')
    return result



class CleanShareSchedule:
    """start_percent:hold_through_epoch:end_percent:ramp_end_epoch:refresh_period."""
    def __init__(self, spec):
        try:
            if not isinstance(spec, str):
                raise ValueError()
            parts = spec.split(':')
            if len(parts) != 5:
                raise ValueError()
            self.start, self.hold, self.end, self.finish, self.refresh = (
                float(parts[0]) / 100, int(parts[1]), float(parts[2]) / 100,
                int(parts[3]), int(parts[4]))
        except (ValueError, TypeError):
            raise ValueError('logit_normal_clean_schedule must be start%:hold_epoch:end%:end_epoch:refresh_period') from None
        if not (0 < self.start < 1 and 0 < self.end < 1 and
                self.hold >= 1 and self.finish > self.hold and self.refresh >= 0):
            raise ValueError('invalid clean-share curriculum percentages or epochs')
        self.spec = [self.start, self.hold, self.end, self.finish, self.refresh]

    def target(self, epoch):
        if type(epoch) is not int or epoch < 1:
            raise ValueError('curriculum epochs are positive, one-based integers')
        if epoch <= self.hold:
            return self.start
        if epoch < self.finish:
            return self.start + (self.end - self.start) * (epoch - self.hold) / (self.finish - self.hold)
        if self.refresh and epoch > self.finish and (epoch - self.finish) % self.refresh == 0:
            return self.start
        return self.end


@lru_cache(maxsize=128)
def scheduled_mean(spec, epoch, std, cutoff, uniform):
    return mean_for_share(CleanShareSchedule(spec).target(epoch), std, cutoff, uniform)


class CurriculumCursor:
    """Track primary batches fetched, independently of optimizer/outer steps.

    Resumes restore the within-epoch offset. This guarantees curriculum phase,
    not identical sample order, RNG, augmentation or bitwise continuation.
    """
    def __init__(self, spec, sampler, batches_per_epoch, data_identity, saved=None, resuming=False):
        if type(batches_per_epoch) is not int or batches_per_epoch <= 0:
            raise ValueError('curriculum needs a nonempty fixed-length primary loader')
        schedule = CleanShareSchedule(spec)
        # Validate both endpoints against the actual mixture before training.
        for target in (schedule.start, schedule.end):
            mean_for_share(target, sampler['logit_std'], sampler['cutoff'], sampler['uniform_mix'])
        self.contract = {'version': 1, 'schedule': schedule.spec,
                         'sampler': copy.deepcopy(sampler),
                         'batches_per_epoch': batches_per_epoch,
                         'data_identity': copy.deepcopy(data_identity)}
        self.pending = False
        self.consumed = 0
        if saved is not None:
            if not isinstance(saved, dict) or saved.get('contract') != self.contract:
                raise ValueError('clean-share curriculum checkpoint/configuration or dataset mismatch')
            count = saved.get('consumed')
            if type(count) is not int or count < 0:
                raise ValueError('invalid clean-share curriculum checkpoint cursor')
            self.consumed = count
        elif resuming:
            raise ValueError('checkpoint has no clean-share curriculum cursor; start a new output folder')

    @property
    def position(self):
        return divmod(self.consumed, self.contract['batches_per_epoch'])

    def begin(self):
        self.pending = True

    def commit(self):
        self.pending = False

    def stamp(self, batch):
        epoch = self.position[0] + 1
        batch._clean_share_epoch = epoch
        self.consumed += 1
        return epoch

    def state_dict(self):
        if self.pending:
            raise RuntimeError('cannot checkpoint an incomplete curriculum batch group')
        return {'contract': copy.deepcopy(self.contract), 'consumed': self.consumed}


def prepare_curriculum(trainer, data_identity):
    cfg = trainer.train_config
    spec = cfg.logit_normal_clean_schedule
    if spec is None:
        return None
    if getattr(cfg, 'gradient_accumulation_steps', 1) != 1:
        raise ValueError('clean-share curriculum requires gradient_accumulation_steps: 1; use gradient_accumulation for microbatches')
    if trainer.data_loader is None or trainer.data_loader_reg is not None:
        raise ValueError('clean-share curriculum requires one primary loader and no regularization loader')
    if getattr(trainer.accelerator, 'num_processes', 1) != 1:
        raise ValueError('clean-share curriculum currently supports single-process training only')
    if getattr(trainer.sd, 'is_multistage', False) or trainer.model_config.refiner_name_or_path is not None:
        raise ValueError('clean-share curriculum does not support multistage/refiner timestep ranges')
    stochastic = getattr(trainer, '_rawr_stochastic', None)
    if stochastic is not None and not stochastic.reason:
        raise ValueError('clean-share curriculum cannot combine with recorded stochastic iterator resume')
    if 'unavailable' in data_identity:
        raise ValueError('clean-share curriculum requires identifiable datasets')
    cursor = CurriculumCursor(spec, cfg.flow_timestep_kwargs('logit_normal'), len(trainer.data_loader),
                              data_identity, getattr(trainer, '_clean_share_saved', None),
                              resuming=trainer.step_num > 0)
    trainer._clean_share_cursor = cursor
    return cursor
