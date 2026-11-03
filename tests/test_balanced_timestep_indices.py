"""content_or_style 'balanced': flowmatch timestep indices are drawn from [min, max] inclusive (3 Nov 2026).

torch.randint's exclusive upper bound used to leave out the grid's last index, which holds its lowest sigma: with
timestep_type shift_once at shift 12, training never saw sigma 0.0119, only down to the next step (0.0235).
"""
import pytest
import torch

from toolkit.timestep_indices import balanced_timestep_indices, last_train_index
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


def _draw(lo, hi, flowmatch, n=2_000_000):
    torch.manual_seed(0)
    return balanced_timestep_indices(lo, hi, n, flowmatch=flowmatch)


@pytest.mark.parametrize("lo,hi", [(0, 999), (0, 29), (250, 750)])
def test_flowmatch_covers_both_ends(lo, hi):
    idx = _draw(lo, hi, True)
    assert idx.dtype == torch.long
    assert idx.min().item() == lo and idx.max().item() == hi
    # uniform over the hi - lo + 1 indices (>= 2000 draws per index, so a 4.5-sigma band is under 10%)
    counts = torch.bincount(idx - lo, minlength=hi - lo + 1).float()
    assert counts.numel() == hi - lo + 1
    assert (counts / counts.mean() - 1).abs().max().item() < 0.1


def test_legacy_open_interval_unchanged_for_other_schedulers():
    idx = _draw(0, 999, False)
    assert idx.min().item() == 1 and idx.max().item() == 997


@pytest.mark.parametrize("flowmatch", [True, False])
def test_equal_bounds(flowmatch):
    idx = balanced_timestep_indices(500, 500, 7, flowmatch=flowmatch)
    assert idx.dtype == torch.long and idx.tolist() == [500] * 7


def test_lowest_sigma_reachable_with_shift_once():
    scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=12)
    timesteps = scheduler.set_train_timesteps(1000, "cpu", timestep_type="shift_once")
    idx = _draw(0, 999, True)  # the trainer clips max_denoising_steps (999) to len(grid) - 1
    sigmas = timesteps[idx] / 1000
    assert sigmas.min().item() == pytest.approx(0.011869436, abs=1e-6)
    assert sigmas.max().item() == pytest.approx(1.0, abs=1e-6)


# Astra's review (3 Nov): the inclusive bound must hold against the grid the scheduler actually built, and against the
# fixed 1000-entry weighting tables.

@pytest.mark.parametrize("timestep_type,n,last", [("lognorm_blend", 30, 28), ("lognorm_blend", 1001, 999),
                                                  ("shift_once", 1000, 999), ("weighted", 1001, 1000), ("linear", 30, 29)])
def test_draws_stay_on_the_built_grid(timestep_type, n, last):
    scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3)
    timesteps = scheduler.set_train_timesteps(n, "cpu", timestep_type=timestep_type)
    # the trainer's clip: max_denoising_steps (default 999, or set higher) against the built grid
    top = min(10_000, last_train_index(n, scheduler))
    assert top == last == len(timesteps) - 1
    idx = _draw(0, top, True, n=20_000)
    assert idx.max().item() == last
    timesteps[idx]  # in range for every draw


def test_last_train_index_without_a_grid():
    class NoGrid:
        timesteps = None
    assert last_train_index(1000, NoGrid()) == 999


@pytest.mark.parametrize("kwargs", [{"timestep_type": "weighted"}, {"timestep_type": "weighted", "x0_pred": True},
                                    {"timestep_type": "linear"}, {"timestep_type": "linear", "v2": True}])
def test_weights_for_a_grid_longer_than_the_tables(kwargs):
    scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    timesteps = scheduler.set_train_timesteps(1001, "cpu", timestep_type="weighted")
    w = scheduler.get_weights_for_timesteps(timesteps[[0, 998, 999, 1000]], **kwargs)
    assert w.shape == (4,) and torch.isfinite(w).all()
    assert w[3].item() == w[2].item()  # the tail past the 1000-entry table reuses its last weight
