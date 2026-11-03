"""timestep_type 'shift_once': the static shift applied once to the raw 1 .. 1/N training grid.

The legacy 'shift' types start from the scheduler's sigma_min, which diffusers has already shifted for static
shifting, so their endpoint is shifted twice. Contracts adapted from Astra's flowmatch review (models-6, 2 Nov 2026).
"""
import math

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler, calculate_shift
from extensions_built_in.diffusion_models.minimax_h3.src.packing import build_sigma_schedule

N = 1000


def _once(shift, n):
    raw = torch.linspace(1.0, 1.0 / N, n, dtype=torch.float64)
    return shift * raw / (1 + (shift - 1) * raw)


@pytest.mark.parametrize("n", [1000, 30])
@pytest.mark.parametrize("shift", [1.0, 3.0, 5.0, 12.0])
def test_shift_once_is_one_shift_of_the_raw_grid(shift, n):
    scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, shift=shift)
    t = scheduler.set_train_timesteps(n, "cpu", timestep_type="shift_once")
    expected = _once(shift, n)
    torch.testing.assert_close(t.double() / N, expected, atol=2e-7, rtol=2e-6)
    # sigmas and timesteps stay consistent, with the trailing zero the sampler expects
    torch.testing.assert_close(scheduler.sigmas[:-1].double(), expected, atol=2e-7, rtol=2e-6)
    assert scheduler.sigmas[-1] == 0
    torch.testing.assert_close(scheduler.timesteps, scheduler.sigmas[:-1] * N)


def test_shift_once_endpoint_at_shift_12():
    scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, shift=12)
    t = scheduler.set_train_timesteps(N, "cpu", timestep_type="shift_once")
    assert (t[-1] / N).item() == pytest.approx(0.011869436, abs=1e-7)
    assert (t[0] / N).item() == pytest.approx(1.0, abs=1e-7)


def test_shift_once_matches_h3_dense_sampling_grid():
    scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, shift=12)
    actual = scheduler.set_train_timesteps(N, "cpu", timestep_type="shift_once") / N
    expected = build_sigma_schedule(N, shift=12)[:-1]
    assert expected.shape == actual.shape
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("shift,legacy_min", [(3.0, 0.008928572), (5.0, 0.024414066), (12.0, 0.125984253)])
def test_legacy_shift_is_unchanged(shift, legacy_min):
    parent = FlowMatchEulerDiscreteScheduler(num_train_timesteps=N, shift=shift)
    parent.set_timesteps(N, device="cpu")
    custom = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, shift=shift)
    t = custom.set_train_timesteps(N, "cpu", timestep_type="shift")
    torch.testing.assert_close(custom.sigmas, parent.sigmas, atol=0, rtol=0)
    assert (t[-1] / N).item() == pytest.approx(legacy_min, abs=1e-7)


def test_shift_once_equals_shift_at_shift_1():
    a = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, shift=1.0)
    b = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, shift=1.0)
    torch.testing.assert_close(
        a.set_train_timesteps(N, "cpu", timestep_type="shift_once"),
        b.set_train_timesteps(N, "cpu", timestep_type="shift"),
        atol=1e-4, rtol=0,
    )


@pytest.mark.parametrize("legacy_type", ["flux_shift", "shift"])
def test_shift_once_equals_legacy_when_dynamic(legacy_type):
    # dynamic shifting already starts from the unshifted endpoint, so the two types agree up to float rounding
    latents = torch.zeros(1, 4, 16, 16)
    a = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, use_dynamic_shifting=True)
    b = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, use_dynamic_shifting=True)
    t_once = a.set_train_timesteps(N, "cpu", timestep_type="shift_once", latents=latents)
    t_legacy = b.set_train_timesteps(N, "cpu", timestep_type=legacy_type, latents=latents)
    torch.testing.assert_close(t_once, t_legacy, atol=1e-4, rtol=0)
    shift = math.exp(calculate_shift(256))
    torch.testing.assert_close(t_once.double() / N, _once(shift, N), atol=2e-7, rtol=2e-6)


# lookups the trainer does on the grid (Astra's review nit, 2 Nov): get_sigmas finds a drawn timestep by exact value,
# index_for_timestep (diffusers) likewise; both need unique timesteps and sigmas that agree with them

@pytest.mark.parametrize("n", [1000, 30])
def test_shift_once_lookups(n):
    scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, shift=12)
    t = scheduler.set_train_timesteps(n, "cpu", timestep_type="shift_once")
    assert torch.unique(t).numel() == n
    idx = [0, n // 2, n - 2, n - 1]
    for i in idx:
        assert scheduler.index_for_timestep(t[i]) == i
    sigma = scheduler.get_sigmas(t[idx], n_dim=4, dtype=torch.bfloat16, device="cpu")
    assert sigma.shape == (len(idx), 1, 1, 1) and sigma.dtype == torch.bfloat16
    torch.testing.assert_close(sigma.flatten().float(), (t[idx] / N).float(), atol=0, rtol=8e-3)


def test_shift_once_dynamic_non_base_sequence_length():
    latents = torch.zeros(1, 4, 48, 40)  # 1920 tokens, between the base (256) and max (4096) lengths
    a = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, use_dynamic_shifting=True)
    b = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=N, use_dynamic_shifting=True)
    t_once = a.set_train_timesteps(N, "cpu", timestep_type="shift_once", latents=latents)
    t_legacy = b.set_train_timesteps(N, "cpu", timestep_type="flux_shift", latents=latents)
    torch.testing.assert_close(t_once, t_legacy, atol=1e-4, rtol=0)
    # the scheduler takes the shift range from its config (diffusers: max_shift 1.15, not calculate_shift's 1.16)
    c = a.config
    mu = calculate_shift(48 * 40, c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.16))
    torch.testing.assert_close(t_once.double() / N, _once(math.exp(mu), N), atol=2e-7, rtol=2e-6)
    sigma = a.get_sigmas(t_once[[0, 999]], n_dim=2, dtype=torch.bfloat16, device="cpu")
    torch.testing.assert_close(sigma.flatten().float(), (t_once[[0, 999]] / N).float(), atol=0, rtol=8e-3)
