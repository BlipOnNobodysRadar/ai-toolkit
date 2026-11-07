"""Parameter swapping selection (docs/STATE_MACHINES.md M13, SM-13).

The selection added a parameter's size before deciding, and stopped once the running total reached the target, so
the parameter that reached it never trained: with one parameter nothing trained at any factor, and factor 1 always
left one parameter out. Swapping also clears every gradient at the top of each loop step, so it cannot accumulate.
Real Adafactor/Automagic optimizers and the real trainer check; CPU only.
"""
import importlib
from types import SimpleNamespace

import pytest
import torch

from toolkit.optimizer import get_optimizer


@pytest.mark.parametrize('name', ['adafactor', 'automagic'])
@pytest.mark.parametrize('factor', [.1, .5, 1.])
def test_a_single_parameter_always_trains(name, factor):
    p = torch.nn.Parameter(torch.ones(8, 8))
    optimizer = get_optimizer([p], name, 1e-3, {})
    optimizer.enable_paramiter_swapping(factor)
    assert p.requires_grad


@pytest.mark.parametrize('name', ['adafactor', 'automagic'])
def test_factor_one_trains_every_parameter(name):
    params = [torch.nn.Parameter(torch.ones(n)) for n in (3, 5, 7, 11)]
    optimizer = get_optimizer(params, name, 1e-3, {})
    for _ in range(10):
        optimizer.enable_paramiter_swapping(1.)
        assert all(p.requires_grad for p in params)


@pytest.mark.parametrize('name', ['adafactor', 'automagic'])
def test_selection_stays_within_the_target(name):
    params = [torch.nn.Parameter(torch.ones(10)) for _ in range(10)]
    optimizer = get_optimizer(params, name, 1e-3, {})
    for _ in range(10):
        optimizer.enable_paramiter_swapping(.35)
        assert sum(p.numel() for p in params if p.requires_grad) == 30


@pytest.mark.parametrize('name', ['adafactor', 'automagic'])
@pytest.mark.parametrize('factor', [0, -.1, 1.5])
def test_out_of_range_factor_is_rejected(name, factor):
    optimizer = get_optimizer([torch.nn.Parameter(torch.ones(4))], name, 1e-3, {})
    with pytest.raises(ValueError, match='paramiter_swapping_factor'):
        optimizer.enable_paramiter_swapping(factor)


@pytest.mark.parametrize('gas,ok', [(1, True), (2, False), (-1, False)])
def test_swapping_refuses_gradient_accumulation_steps(gas, ok):
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    f = SimpleNamespace(train_config=SimpleNamespace(gradient_accumulation_steps=gas, gradient_accumulation=2,
                                                     do_paramiter_swapping=True, optimizer='adafactor'))
    optimizer = get_optimizer([torch.nn.Parameter(torch.ones(4))], 'adafactor', 1e-3, {})
    if ok:
        base._check_accumulation_support(f, optimizer)
    else:
        with pytest.raises(ValueError, match='do_paramiter_swapping'):
            base._check_accumulation_support(f, optimizer)
