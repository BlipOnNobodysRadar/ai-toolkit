"""Automagic4's calibration -> updating transition (docs/STATE_MACHINES.md M3a). The first step only measures (second
moments and the sensitivity reference); later steps move weights. A resume saved after calibration must continue the
uninterrupted run exactly: load_state_dict rebuilds the group reference from the saved per-parameter sens_ref.
Real toolkit Automagic4 (fused=False) on CPU fp32, through the toolkit factory.
"""
import copy

import torch

from toolkit.optimizer import get_optimizer

TARGET = torch.tensor([[.3, -.7], [1.1, .2]])


def build(start):
    p = torch.nn.Parameter(start.clone())
    return p, get_optimizer([p], 'automagic4', .01, {'fused': False})


def step(p, opt):
    opt.zero_grad(set_to_none=True)
    (p - TARGET).square().sum().backward()
    opt.step()


def test_the_first_step_calibrates_and_later_steps_update():
    start = torch.tensor([[1., 2.], [-.5, .25]])
    p, opt = build(start)
    step(p, opt)
    assert torch.equal(p.detach(), start)  # calibration: measured, not moved
    assert opt._group_ref[0] is not None
    step(p, opt)
    assert not torch.equal(p.detach(), start)


def test_resume_after_calibration_continues_exactly():
    start = torch.tensor([[1., 2.], [-.5, .25]])
    a, opt_a = build(start)
    for _ in range(3):
        step(a, opt_a)
    weights, payload = a.detach().clone(), copy.deepcopy(opt_a.state_dict())
    for _ in range(4):
        step(a, opt_a)

    b, opt_b = build(weights)
    opt_b.load_state_dict(payload)
    assert opt_b._group_ref[0] is not None  # rebuilt from the saved sens_ref, not a second calibration
    for _ in range(4):
        step(b, opt_b)
    assert torch.equal(a, b)


def test_a_resume_that_lost_the_reference_would_calibrate_again():
    # negative control: without sens_ref the resumed optimizer measures again and the trajectories part
    start = torch.tensor([[1., 2.], [-.5, .25]])
    a, opt_a = build(start)
    for _ in range(3):
        step(a, opt_a)
    weights, payload = a.detach().clone(), copy.deepcopy(opt_a.state_dict())
    for _ in range(2):
        step(a, opt_a)
    for st in payload['state'].values():
        st.pop('sens_ref', None)
    b, opt_b = build(weights)
    opt_b.load_state_dict(payload)
    for _ in range(2):
        step(b, opt_b)
    assert not torch.equal(a, b)
