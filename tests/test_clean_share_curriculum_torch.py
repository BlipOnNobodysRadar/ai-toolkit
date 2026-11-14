"""CPU sampler acceptance for the clean-share probability contract."""
import pytest
import torch
from types import SimpleNamespace

from toolkit.clean_share_curriculum import CleanShareSchedule, CurriculumCursor, prepare_curriculum, scheduled_mean
from toolkit.config_modules import TrainConfig
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


@pytest.mark.parametrize('cutoff,uniform', [(1.0, 0.0), (0.95, 0.01)])
@pytest.mark.parametrize('epoch', [1, 11, 15, 17])
def test_actual_sampler_clean_fraction(cutoff, uniform, epoch):
    spec = '60:7:30:15:2'
    target = CleanShareSchedule(spec).target(epoch)
    mean = scheduled_mean(spec, epoch, 0.95, cutoff, uniform)
    scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    # Restore the caller's CPU RNG; no CUDA state or GPU allocation is needed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(epoch)
        timesteps = scheduler.set_train_timesteps(
            200_000, 'cpu', timestep_type='logit_normal', logit_mean=mean,
            logit_std=0.95, cutoff=cutoff, uniform_mix=uniform,
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
    for incompatible in ({'first_timestep_chance': 0.1}, {'loss_type': 'mean_flow'}):
        with pytest.raises(ValueError, match='full-range'):
            TrainConfig(
                noise_scheduler='flowmatch', timestep_type='logit_normal',
                content_or_style='balanced', logit_normal_clean_schedule='60:7:30:15:2',
                **incompatible,
            )


def test_fixed_sigma_scheduler_rejected():
    cfg = TrainConfig(noise_scheduler='flowmatch', timestep_type='logit_normal',
                      content_or_style='balanced', logit_normal_clean_schedule='60:7:30:15:2')
    scheduler = CustomFlowMatchEulerDiscreteScheduler()
    scheduler.fixed_train_sigmas = [1.0, 0.75, 0.5, 0.25]
    trainer = SimpleNamespace(train_config=cfg, data_loader=[1, 2, 3], data_loader_reg=None,
                              accelerator=SimpleNamespace(num_processes=1),
                              sd=SimpleNamespace(is_multistage=False, noise_scheduler=scheduler),
                              model_config=SimpleNamespace(refiner_name_or_path=None), step_num=0)
    with pytest.raises(ValueError, match='fixed_train_sigmas'):
        prepare_curriculum(trainer, [['dataset'], None])


def test_checkpoint_metadata_roundtrip(tmp_path):
    from safetensors.torch import save_file
    from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors
    spec = '60:7:30:15:2'
    sampler = dict(logit_mean=1.09, logit_std=.95, cutoff=.95, uniform_mix=.01)
    identity = [['dataset'], None]
    cursor = CurriculumCursor(spec, sampler, 3, identity)
    cursor.begin()
    for _ in range(5):
        cursor.stamp(SimpleNamespace())
    cursor.commit()
    path = tmp_path / 'checkpoint.safetensors'
    metadata = get_meta_for_safetensors({'training_info': {'clean_share_curriculum': cursor.state_dict()}})
    save_file({'weight': torch.zeros(1)}, str(path), metadata=metadata)
    saved = load_metadata_from_safetensors(str(path))['training_info']['clean_share_curriculum']
    restored = CurriculumCursor(spec, sampler, 3, identity, saved=saved, resuming=True)
    assert restored.position == (1, 2)
    assert restored.stamp(SimpleNamespace()) == 2
    assert restored.stamp(SimpleNamespace()) == 3
    saved['contract']['sampler']['logit_std'] = 1.5
    with pytest.raises(ValueError, match='mismatch'):
        CurriculumCursor(spec, sampler, 3, identity, saved=saved, resuming=True)
