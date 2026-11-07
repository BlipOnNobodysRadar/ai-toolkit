"""Adafactor with stochastic_rounding off must still write its update into bf16/fp16 parameters (docs/STATE_MACHINES.md
P-03; Codex adafactor-disabled-rounding-probe). Real toolkit Adafactor on CPU; the fp32 run of the same step is the
reference, rounded to nearest. (The stochastic-rounding path asserts a CUDA device and is not exercised here.)
"""
import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor

SETTINGS = dict(lr=.1, relative_step=False, scale_parameter=False, warmup_init=False)


def one_step(dtype, stochastic_rounding):
    p = torch.nn.Parameter(torch.tensor([[1., -2.], [.5, 3.]], dtype=dtype))
    opt = Adafactor([p], stochastic_rounding=stochastic_rounding, **SETTINGS)
    p.grad = torch.tensor([[1., -1.], [.5, 2.]], dtype=dtype)  # exactly representable in bf16/fp16
    opt.step()
    return p, opt


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_low_precision_parameters_receive_the_update_without_stochastic_rounding(dtype):
    p, opt = one_step(dtype, stochastic_rounding=False)
    reference, ref_opt = one_step(torch.float32, stochastic_rounding=False)
    start = torch.tensor([[1., -2.], [.5, 3.]], dtype=dtype)
    assert not torch.equal(p.detach(), start)  # the original left them untouched
    assert torch.equal(p.detach(), reference.detach().to(dtype))  # round to nearest of the fp32 result
    assert opt.state[p]['step'] == ref_opt.state[reference]['step'] == 1
    assert torch.equal(opt.state[p]['exp_avg_sq_row'], ref_opt.state[reference]['exp_avg_sq_row'])


def test_fp32_parameters_are_unchanged_by_the_fix():
    p, _ = one_step(torch.float32, stochastic_rounding=False)
    q, _ = one_step(torch.float32, stochastic_rounding=True)
    assert torch.equal(p, q) and not torch.equal(p.detach(), torch.tensor([[1., -2.], [.5, 3.]]))
