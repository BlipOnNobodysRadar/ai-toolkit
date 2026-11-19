"""Loss weights reach the parameter gradients exactly once (docs/STATE_MACHINES.md, OPUS-MF-01), and empty masks give
defined, finite weights (INV-MASK-01). The real SDTrainer.get_mean_flow_loss and the shared backward_weighted_loss
run unbound with a real CPU Accelerator; a one-parameter predictor stands in for the model (as in the audit's
loss_probes.py). CPU only."""
import random
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer, inverted_prior_mask_multiplier, latent_mask_multiplier
from test_rawr_trainer_continuity import cpu_only  # noqa: F401  (autouse: no CUDA, fresh Accelerate state)


def fixture(batch_size=1, weights=None, fail=None, nan=False):
    p = torch.nn.Parameter(torch.tensor(.5))
    accelerator = Accelerator(cpu=True)
    calls = []
    real_backward = accelerator.backward

    def backward(loss, **kwargs):
        calls.append(float(loss.detach()))
        real_backward(loss, **kwargs)
    accelerator.backward = backward
    trainer = SimpleNamespace(
        train_config=SimpleNamespace(dtype='float32'), accelerator=accelerator, timer=lambda _: nullcontext(),
        sd=SimpleNamespace(noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000),
                                                          timesteps=torch.arange(1000, 0, -1))))
    count = []

    def predict_noise(noisy_latents, **kwargs):
        count.append(1)
        if fail is not None and len(count) == fail:
            raise RuntimeError('prediction failed')
        out = p * noisy_latents
        return out * float('nan') if nan else out
    trainer.predict_noise = predict_noise
    latents = torch.arange(1., 1. + batch_size * 4).reshape(batch_size, 1, 2, 2)
    batch = SimpleNamespace(latents=latents, loss_multiplier_list=weights or [1.] * batch_size)
    return trainer, p, batch, calls


def microbatch(trainer, batch, reg_weight=1., accum_scale=1.):
    random.seed(43)
    noise = torch.zeros_like(batch.latents)
    loss = SDTrainer.get_mean_flow_loss(trainer, noisy_latents=noise, conditional_embeds=None,
                                        match_adapter_assist=False, network_weight_list=[1], timesteps=None,
                                        pred_kwargs={}, batch=batch, noise=noise)
    multiplier = torch.full((batch.latents.shape[0], 1, 1, 1), reg_weight)
    return SDTrainer.backward_weighted_loss(trainer, loss, multiplier, accum_scale)


def baseline():
    trainer, p, batch, calls = fixture()
    loss = microbatch(trainer, batch)
    return float(p.grad), float(loss), calls


def test_single_step_baseline_is_one_backward_of_the_real_loss():
    grad, loss, calls = baseline()
    assert grad != 0 and calls == [loss]  # one backward, of the loss the caller reports
    # the same expression differentiated directly
    trainer, p, batch, _ = fixture()
    random.seed(43)
    noise = torch.zeros_like(batch.latents)
    direct = SDTrainer.get_mean_flow_loss(trainer, noisy_latents=noise, conditional_embeds=None,
                                          match_adapter_assist=False, network_weight_list=[1], timesteps=None,
                                          pred_kwargs={}, batch=batch, noise=noise)
    assert direct.requires_grad and float(torch.autograd.grad(direct, p)[0]) == grad


def test_a_zero_dataset_weight_trains_nothing():
    trainer, p, batch, _ = fixture(weights=[0.])
    loss = microbatch(trainer, batch)
    assert float(loss) == 0 and float(p.grad) == 0


def test_the_reg_weight_scales_the_gradient():
    grad, _, _ = baseline()
    trainer, p, batch, _ = fixture()
    microbatch(trainer, batch, reg_weight=.25)
    assert float(p.grad) == pytest.approx(.25 * grad, rel=1e-6)


def test_two_half_scaled_microbatches_equal_one_full_batch():
    grad, _, _ = baseline()
    trainer, p, batch, calls = fixture()
    for _ in range(2):
        microbatch(trainer, batch, accum_scale=.5)
    assert float(p.grad) == pytest.approx(grad, rel=1e-6) and len(calls) == 2  # was 2x before


def test_per_sample_dataset_weights_apply_like_calculate_loss():
    trainer, p, batch, _ = fixture(batch_size=2, weights=[1., 0.])
    microbatch(trainer, batch)
    two = float(p.grad)
    first, q, single, _ = fixture(batch_size=1)
    microbatch(first, single)
    # the batch mean of (sample 0 at weight 1, sample 1 at weight 0) is half of sample 0 alone: sample 0 has the same
    # latents and the same timestep draws (the first two after the seed) in both
    assert two == pytest.approx(float(q.grad) / 2, rel=1e-5)


def test_a_nonfinite_loss_never_reaches_the_parameters():
    trainer, p, batch, calls = fixture(nan=True)
    loss = microbatch(trainer, batch)
    assert float(loss) == 0 and (p.grad is None or float(p.grad) == 0) and len(calls) == 1


@pytest.mark.parametrize('fail', [1, 2])
def test_a_failed_prediction_leaves_no_gradient(fail):
    trainer, p, batch, calls = fixture(fail=fail)
    with pytest.raises(RuntimeError, match='prediction failed'):
        microbatch(trainer, batch)
    assert p.grad is None and calls == []


def test_accelerate_divides_by_its_own_accumulation_once():
    # with Accelerate's gradient_accumulation_steps 2 its backward halves the loss; the trainer's accum_scale is a
    # separate policy and is not applied by Accelerate a second time
    trainer, p, batch, _ = fixture()
    trainer.accelerator.gradient_accumulation_steps = 2
    microbatch(trainer, batch)
    grad, _, _ = baseline()
    assert float(p.grad) == pytest.approx(grad / 2, rel=1e-6)


# ---- masks ----

def no_host_sync(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('host sync in the mask helper')
    monkeypatch.setattr(torch.Tensor, 'item', forbidden)
    monkeypatch.setattr(torch.Tensor, '__bool__', forbidden)


@pytest.mark.parametrize('mode', ['nearest', 'area'])
def test_an_empty_foreground_mask_contributes_nothing(monkeypatch, mode):
    no_host_sync(monkeypatch)
    weight = latent_mask_multiplier(torch.zeros(2, 1, 16, 16), (2, 4, 2, 2), mode)
    monkeypatch.undo()
    assert torch.isfinite(weight).all() and torch.equal(weight, torch.zeros(2, 4, 2, 2))


@pytest.mark.parametrize('mode', ['nearest', 'area'])
def test_full_and_ordinary_foreground_masks_are_unchanged(monkeypatch, mode):
    torch.manual_seed(0)
    for mask in (torch.ones(2, 1, 16, 16), torch.rand(2, 1, 16, 16), (torch.rand(2, 1, 16, 16) > .7).float()):
        size = torch.nn.functional.interpolate(mask, size=(4, 4), mode=mode) if mode == 'area' else \
            torch.nn.functional.interpolate(mask, size=(4, 4))
        legacy = size.expand(-1, 3, -1, -1)
        legacy = legacy / legacy.mean()
        no_host_sync(monkeypatch)
        weight = latent_mask_multiplier(mask, (2, 3, 4, 4), mode)
        monkeypatch.undo()
        assert torch.equal(weight, legacy)


@pytest.mark.parametrize('shape,latent', [((1, 1, 64, 64), (1, 4, 8, 8)), ((1, 1, 100, 73), (1, 4, 13, 9))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_an_all_white_mask_leaves_no_prior_region(monkeypatch, shape, latent, dtype):
    # 100x73 -> 13x9 is a non-integer resize: bicubic leaves residue (up to ~4e-3 in bf16) instead of exact zeros
    no_host_sync(monkeypatch)
    weight = inverted_prior_mask_multiplier(torch.ones(shape, dtype=dtype), latent, latent)
    monkeypatch.undo()
    assert torch.isfinite(weight).all() and not weight.any()


def test_ordinary_inverted_prior_masks_are_unchanged(monkeypatch):
    torch.manual_seed(0)
    mask = torch.rand(2, 1, 32, 32)
    prior = torch.nn.functional.interpolate(mask, size=(4, 4), mode='bicubic')[:, :1].expand(-1, 3, -1, -1)
    legacy = (1.0 - prior) / (1.0 - prior).mean()
    no_host_sync(monkeypatch)
    weight = inverted_prior_mask_multiplier(mask, (2, 3, 4, 4), (2, 3, 4, 4))
    monkeypatch.undo()
    assert torch.equal(weight, legacy)
