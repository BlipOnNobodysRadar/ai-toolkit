"""CPU-only tests for SimplifiedAdEMAMixExM ("SAEM") transfer and compile paths.

All tests force torch.cuda.is_available() to False so the optimizer computes on
the CPU even on a machine with a GPU.
"""
import pytest
import torch

from training_extras.vendor.LoraEasyCustomOptimizer import ademamix
from training_extras.vendor.LoraEasyCustomOptimizer.ademamix import SimplifiedAdEMAMixExM


SHAPES = [(8, 4), (4, 8), (16,), (3, 3, 2, 2)]


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _params(dtype=torch.float32, shapes=SHAPES, seed=0):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s).to(dtype)) for s in shapes]


def _set_grads(params, step):
    g = torch.Generator().manual_seed(100 + step)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g).to(p.dtype)


def test_cpu_param_write_back_is_non_blocking_when_pinned(monkeypatch):
    # Under network layer offload the fp32 LoRA params are pinned CPU tensors.
    # Their write-back must be an async D2H (the group-end synchronize orders
    # it); a blocking copy_ syncs the CUDA stream once per parameter.
    params = _params()
    opt = SimplifiedAdEMAMixExM(params, lr=1e-3, torch_compile=False)
    _set_grads(params, 0)
    opt.step()  # step 1 does not update params; step 2 does

    ptrs = {p.data_ptr() for p in params}
    writes = {}
    real_copy = torch.Tensor.copy_

    def recording_copy(self, src, non_blocking=False):
        if self.data_ptr() in ptrs:
            writes[self.data_ptr()] = non_blocking
        return real_copy(self, src, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self, *a, **k: True)
    monkeypatch.setattr(torch.Tensor, "copy_", recording_copy)
    _set_grads(params, 1)
    opt.step()

    assert set(writes) == ptrs
    assert all(writes.values()), writes
