"""ExponentialMovingAverage.train() releases the store()d clone (deep dive, 1 Nov 2026).

eval() clones every trained param into collected_params before copying the EMA weights in; train() copied them back
but kept the clone, so after the first save or sample one extra full copy of the trained params stayed allocated.
"""
import torch

from toolkit.ema import ExponentialMovingAverage


def test_train_restores_raw_weights_and_frees_the_clone():
    params = [torch.nn.Parameter(torch.tensor([1., 2.])), torch.nn.Parameter(torch.tensor([[3.]]))]
    ema = ExponentialMovingAverage(params, decay=0.5)
    with torch.no_grad():
        for p in params:
            p.add_(1.)
    ema.update()
    raw = [p.detach().clone() for p in params]
    ema.eval()
    assert ema.collected_params is not None
    assert not torch.equal(params[0], raw[0])  # EMA weights swapped in
    ema.train()
    assert ema.collected_params is None
    assert all(torch.equal(p, r) for p, r in zip(params, raw))
    # a second eval/train round still works
    ema.eval()
    ema.train()
    assert ema.collected_params is None and all(torch.equal(p, r) for p, r in zip(params, raw))
