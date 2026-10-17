import random
from types import SimpleNamespace

import pytest
import torch

from toolkit.noise_augmentation import (
    augmentation_seed, begin_noise_augmentation_batch, disable_noise_augmentation,
    input_perturbed_noise, multires_noise_like,
)


def test_disabled_preserves_tensor_and_rng():
    noise = torch.randn(2, 4, 32, 48)
    before = torch.get_rng_state().clone()
    assert multires_noise_like(noise, 0, .1, seed=3) is noise
    assert input_perturbed_noise(noise, 0, seed=4) is noise
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('shape', [(3, 4, 32, 48), (2, 4, 5, 32, 48)])
def test_enabled_preserves_input_and_other_random_streams(shape):
    noise = torch.randn(shape)
    saved = noise.clone()
    before = torch.get_rng_state().clone()
    py_before = random.getstate()
    target = multires_noise_like(noise, 6, .1, seed=11)
    perturbed = input_perturbed_noise(target, .04, seed=17)
    assert target.shape == perturbed.shape == noise.shape
    assert torch.equal(noise, saved)
    assert torch.equal(target, multires_noise_like(noise, 6, .1, seed=11))
    assert torch.equal(before, torch.get_rng_state())
    assert random.getstate() == py_before
    assert torch.isfinite(perturbed).all()
    assert target.std().item() == pytest.approx(1, abs=1e-5)


def test_pyramid_increases_spatial_low_frequency_energy():
    noise = torch.randn((16, 4, 128, 128), generator=torch.Generator().manual_seed(7))
    pyramid = multires_noise_like(noise, 6, .6, seed=11)
    low = lambda x: torch.nn.functional.avg_pool2d(x, 8).square().mean().item()
    assert low(pyramid) > low(noise) * 1.4


def test_input_perturbation_changes_input_not_velocity_target():
    clean = torch.randn(8, 4, 32, 32)
    noise = torch.randn_like(clean)
    target = noise - clean
    ip = input_perturbed_noise(noise, .04, seed=19)
    sigma = torch.tensor([0., .25, .5, 1., 0., .25, .5, 1.]).view(-1, 1, 1, 1)
    ordinary = (1 - sigma) * clean + sigma * noise
    noisy = (1 - sigma) * clean + sigma * ip
    assert torch.equal(target, noise - clean)
    assert torch.equal(noisy[sigma.flatten() == 0], ordinary[sigma.flatten() == 0])
    assert not torch.equal(noisy[-1], ordinary[-1])
    assert (ip - noise).std().item() == pytest.approx(.04, rel=.03)


def test_validation_context_restores_after_exception():
    trainer = SimpleNamespace()
    with pytest.raises(RuntimeError):
        with disable_noise_augmentation(trainer):
            assert trainer._disable_noise_augmentation
            raise RuntimeError('cache failure')
    assert not trainer._disable_noise_augmentation
    trainer._disable_noise_augmentation = True
    with disable_noise_augmentation(trainer):
        pass
    assert trainer._disable_noise_augmentation


def test_native_get_noise_keeps_validation_control_clean():
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
    noise = torch.randn(2, 4, 32, 32)
    trainer = object.__new__(BaseSDTrainProcess)
    trainer.train_config = SimpleNamespace(
        optimal_noise_pairing_samples=1, force_consistent_noise=False,
        noise_offset=0, blended_blur_noise=False, multires_noise_iterations=6,
        multires_noise_discount=.1, noise_augmentation_seed=42,
    )
    trainer.sd = SimpleNamespace(get_latent_noise_from_latents=lambda *a, **k: noise)
    trainer.device_torch = torch.device('cpu')
    trainer.step_num = 0
    trainer.grad_accumulation_step = 1
    augmented = trainer.get_noise(noise, 2)
    assert not torch.equal(augmented, noise)
    with disable_noise_augmentation(trainer):
        assert torch.equal(trainer.get_noise(noise, 2), noise)


def test_seed_separates_updates_and_components():
    seeds = {augmentation_seed(42, step, micro, stream)
             for step in range(100) for micro in range(4) for stream in (1, 2)}
    assert len(seeds) == 800


def test_list_accumulation_has_distinct_augmentation_without_shifting_next_step():
    trainer = SimpleNamespace(step_num=3, grad_accumulation_step=0)
    begin_noise_augmentation_batch(trainer)
    first = augmentation_seed(42, 3, 0, 1, batch_index=trainer._noise_augmentation_batch_index)
    assert first == augmentation_seed(42, 3, 0, 1)
    begin_noise_augmentation_batch(trainer)
    second = augmentation_seed(42, 3, 0, 1, batch_index=trainer._noise_augmentation_batch_index)
    assert first != second
    trainer.step_num = 4
    begin_noise_augmentation_batch(trainer)
    assert trainer._noise_augmentation_batch_index == 0


def test_random_strength_is_bounded_and_repeatable():
    noise = torch.zeros(8, 4, 32, 32)
    delta = input_perturbed_noise(noise, .1, seed=123, random_strength=True)
    assert 0 < delta.std().item() < .103
    assert torch.equal(delta, input_perturbed_noise(noise, .1, seed=123, random_strength=True))
