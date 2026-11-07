"""Schedule-free eval()/train() must convert every parameter group (docs/STATE_MACHINES.md M15, Codex C16 /
schedulefree-groups-probe). The vendored optimizer-attribute classes set self.train_mode inside the first group's
iteration, so the second group was skipped. Real optimizers on CPU fp32, two groups (separate LRs, as LoRA+ or
per-module learning rates produce). Uses only the vendored classes' own API, so it runs on the original code too.
"""
import pytest
import torch

from training_extras.optim import create_optimizer

TARGETS = [torch.tensor([.3, -.7]), torch.tensor([1.1, .2])]
# optimizer-attribute layout, each with its own beta for the x <-> y interpolation; the AO and EMAMix/MARS/Nesterov
# variants share the corrected loop but are not run here
CASES = [('easy:ADOPTScheduleFree', lambda opt, g: g['betas'][0]),
         ('easy:FADOPTScheduleFree', lambda opt, g: g['betas'][0])]


def two_groups(name):
    a = torch.nn.Parameter(torch.tensor([1., 2.]))
    b = torch.nn.Parameter(torch.tensor([3., 4.]))
    opt = create_optimizer([{'params': [a], 'lr': .05}, {'params': [b], 'lr': .02}], name, .05)
    opt.train()  # fresh: a flag flip (no z yet)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        sum((p - t).square().sum() for p, t in zip((a, b), TARGETS)).backward()
        opt.step()
    return opt, (a, b)


@pytest.mark.parametrize('name,beta', CASES)
def test_eval_and_train_convert_every_group(name, beta):
    opt, params = two_groups(name)
    y = [p.detach().clone() for p in params]
    z = [opt.state[p]['z'].detach().clone() for p in params]
    expected_x = [torch.lerp(y_i, z_i, 1 - 1 / beta(opt, g)) for y_i, z_i, g in zip(y, z, opt.param_groups)]
    opt.eval()
    for p, x in zip(params, expected_x):
        assert torch.allclose(p, x, rtol=0, atol=1e-6)  # the original converted only the first group
    opt.train()
    for p, y_i in zip(params, y):
        assert torch.allclose(p, y_i, rtol=0, atol=1e-5)
    assert opt.train_mode is True
