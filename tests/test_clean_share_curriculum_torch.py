"""CPU sampler acceptance for the clean-share probability contract."""
import pytest
import torch

from toolkit.clean_share_curriculum import CleanShareSchedule, scheduled_mean
from toolkit.config_modules import TrainConfig
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


@pytest.mark.parametrize('cutoff', [1.0, 0.95])
@pytest.mark.parametrize('epoch', [1, 11, 15, 17])
def test_actual_sampler_clean_fraction(cutoff, epoch):
    spec = '60:7:30:15:2'
    target = CleanShareSchedule(spec).target(epoch)
    mean = scheduled_mean(spec, epoch, 0.95, cutoff, 0.01)
    scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    # Restore the caller's CPU RNG; no CUDA state or GPU allocation is needed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(epoch)
        timesteps = scheduler.set_train_timesteps(
            200_000, 'cpu', timestep_type='logit_normal', logit_mean=mean,
            logit_std=0.95, cutoff=cutoff, uniform_mix=0.01,
        )
    # >5 standard deviations at the highest-variance target; catches inversion,
    # cutoff rescaling and mixture errors without asserting exact sample counts.
    assert abs((timesteps < 500).float().mean().item() - target) < 0.006
    assert torch.isfinite(timesteps).all()


def test_real_config_constructor_preserves_opt_in_default():
    assert TrainConfig().logit_normal_clean_schedule is None
    cfg = TrainConfig(
        noise_scheduler='flowmatch', timestep_type='logit_normal',
        content_or_style='balanced', logit_normal_clean_schedule='60:7:30:15:2',
    )
    assert cfg.logit_normal_clean_schedule == '60:7:30:15:2'
    with pytest.raises(ValueError, match='full-range'):
        TrainConfig(
            noise_scheduler='flowmatch', timestep_type='logit_normal',
            content_or_style='balanced', logit_normal_clean_schedule='60:7:30:15:2',
            first_timestep_chance=0.1,
        )
