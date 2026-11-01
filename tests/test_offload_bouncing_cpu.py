"""CPU-only checks of the offload autograd Functions (_BouncingLinearFn, _BouncingConv2dFn), 1 Nov 2026.

The CUDA branches are exercised on CPU by a fake device whose .type is "cuda": the ring staging, streams and events
are stubbed out, so only the math and what each Function saves for backward run. The real device="cpu" branch is
checked as well.

offload-2: under autocast the forward computes in bf16 but backward runs without autocast, so the backward mixed a
bf16 grad_out with fp32 weights/inputs and raised "expected m1 and m2 to have the same dtype" (the 31 Oct lorabf16
profile crash); the conv path failed in convolution_backward, which autocast does not cover either.
"""
import contextlib

import pytest
import torch
import torch.nn.functional as F

from toolkit.memory_management import manager_modules as mm


class FakeCudaDevice:
    """Takes the CUDA branch (type == "cuda") while compute stays on the CPU (compares equal to the cpu device)."""
    type = "cuda"

    # equal to the activations' torch.device (no x.to(device)), but not to the string "cpu", which the conv
    # backward tests to pick its CPU branch
    def __eq__(self, other):
        return isinstance(other, (torch.device, FakeCudaDevice))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 0


class _Noop:
    def wait_event(self, *a):
        pass

    def record(self, *a):
        pass

    def synchronize(self):
        pass


@pytest.fixture
def fake_cuda(monkeypatch):
    state = {"grad_xfer_done": [_Noop()] * 4, "w_grad_buffers": [None] * 4, "b_grad_buffers": [None] * 4}
    monkeypatch.setattr(mm, "_get_device_state", lambda device: state)
    monkeypatch.setattr(mm, "_stage_forward_weight",
                        lambda st, dev, mat, w, b: (0, mat(w, torch.device("cpu")), b))
    monkeypatch.setattr(mm, "_release_forward_slot", lambda st, idx: None)
    # the float path stages the weight uncast, as the real ring does
    def stage_bwd(st, dev, mat, w):
        calls.append("bwd")
        return 0, w.detach()

    calls = []
    monkeypatch.setattr(mm, "_stage_backward_weight", stage_bwd)
    monkeypatch.setattr(mm, "_release_backward_weight_slot", lambda st, idx: None)
    monkeypatch.setattr(mm, "_stage_grads_to_cpu", lambda st, idx, gw, gb, wp, bp: (
        mm._match_param_layout(gw, wp), mm._match_param_layout(gb, bp)))
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _Noop())
    dev = FakeCudaDevice()
    dev.calls = calls
    return dev


def _device(kind, fake_cuda):
    return fake_cuda if kind == "cuda-branch" else torch.device("cpu")


def _check_branch(kind, fake_cuda):
    # the CUDA-branch backward must have run (it is the only one that stages a backward weight)
    assert bool(fake_cuda.calls) == (kind == "cuda-branch"), fake_cuda.calls


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _lin(x, w, dev):
    return mm._BouncingLinearFn.apply(x, w, None, dev)


def _conv(x, w, dev, padding):
    return mm._BouncingConv2dFn.apply(x, w, None, dev, (1, 1), padding, (1, 1), 1)


@pytest.mark.parametrize("kind", ["cpu", "cuda-branch"])
def test_linear_lora_pair_backward_under_autocast(kind, fake_cuda):
    dev = _device(kind, fake_cuda)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 6, 32, generator=g)
    w_down = torch.randn(8, 32, generator=g).mul_(0.2).requires_grad_(True)
    w_up = torch.randn(32, 8, generator=g).mul_(0.2).requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    xg = x.clone().requires_grad_(True)

    F.linear(F.linear(xr, w_down), w_up).pow(2).sum().backward()
    ref = (xr.grad, w_down.grad.clone(), w_up.grad.clone())
    w_down.grad = w_up.grad = None

    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = _lin(_lin(xg, w_down, dev), w_up, dev)
    assert y.dtype == torch.bfloat16
    y.float().pow(2).sum().backward()  # backward outside autocast, as in training
    assert w_down.grad.dtype == torch.float32 and w_up.grad.dtype == torch.float32
    for got, want in zip((xg.grad, w_down.grad, w_up.grad), ref):
        assert _rel(got, want) < 2e-2
    _check_branch(kind, fake_cuda)


@pytest.mark.parametrize("kind", ["cpu", "cuda-branch"])
def test_conv_lora_pair_backward_under_autocast(kind, fake_cuda):
    dev = _device(kind, fake_cuda)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 8, 6, 6, generator=g)
    w_down = torch.randn(4, 8, 3, 3, generator=g).mul_(0.2).requires_grad_(True)
    w_up = torch.randn(8, 4, 1, 1, generator=g).mul_(0.2).requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    xg = x.clone().requires_grad_(True)

    F.conv2d(F.conv2d(xr, w_down, padding=1), w_up).pow(2).sum().backward()
    ref = (xr.grad, w_down.grad.clone(), w_up.grad.clone())
    w_down.grad = w_up.grad = None

    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = _conv(_conv(xg, w_down, dev, (1, 1)), w_up, dev, (0, 0))
    assert y.dtype == torch.bfloat16
    y.float().pow(2).sum().backward()
    assert w_down.grad.dtype == torch.float32 and w_up.grad.dtype == torch.float32
    for got, want in zip((xg.grad, w_down.grad, w_up.grad), ref):
        assert _rel(got, want) < 2e-2
    _check_branch(kind, fake_cuda)


@pytest.mark.parametrize("kind", ["cpu", "cuda-branch"])
def test_fp32_without_autocast_is_exact(kind, fake_cuda):
    # the casts must be no-ops when nothing is autocast (both current training configs)
    dev = _device(kind, fake_cuda)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(3, 16, generator=g)
    w = torch.randn(4, 16, generator=g).requires_grad_(True)
    xr, xg = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    F.linear(xr, w).pow(2).sum().backward()
    ref_w = w.grad.clone()
    w.grad = None
    _lin(xg, w, dev).pow(2).sum().backward()
    assert torch.allclose(xg.grad, xr.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(w.grad, ref_w, rtol=1e-6, atol=1e-6)
    _check_branch(kind, fake_cuda)


# offload-3: a frozen (or quantized) base never reads its input in backward, so the CUDA branch must not save it
# (native linear/conv don't either); under non-reentrant checkpointing it otherwise stayed alive from the recompute
# until that block's backward. A trainable offloaded weight (H3's LoRA) still saves x for its grad_weight.

def _packed_ids(fn):
    ids = []

    def pack(t):
        ids.append(id(t))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, ids


@pytest.mark.parametrize("kind", ["cuda-branch", "cpu"])  # the cpu branch too (DeepSeek review B5)
@pytest.mark.parametrize("op", ["linear", "conv"])
@pytest.mark.parametrize("trainable", [False, True])
def test_frozen_base_does_not_save_input(op, trainable, kind, fake_cuda):
    g = torch.Generator().manual_seed(4)
    dev = _device(kind, fake_cuda)
    if op == "linear":
        x = torch.randn(2, 5, 16, generator=g).requires_grad_(True)
        w = torch.randn(8, 16, generator=g).requires_grad_(trainable)
        fn = lambda: _lin(x, w, dev)
    else:
        x = torch.randn(1, 4, 5, 5, generator=g).requires_grad_(True)
        w = torch.randn(6, 4, 3, 3, generator=g).requires_grad_(trainable)
        fn = lambda: _conv(x, w, dev, (1, 1))
    out, ids = _packed_ids(fn)
    assert (id(x) in ids) == trainable
    out.pow(2).sum().backward()  # and the backward still works without it
    assert x.grad is not None and (w.grad is not None) == trainable


@pytest.mark.parametrize("kind", ["cuda-branch", "cpu"])
@pytest.mark.parametrize("op", ["linear", "conv"])
def test_frozen_base_grads_under_nonreentrant_checkpoint(op, kind, fake_cuda):
    from torch.utils.checkpoint import checkpoint

    fake_cuda = _device(kind, fake_cuda)
    g = torch.Generator().manual_seed(5)
    if op == "linear":
        x0 = torch.randn(2, 5, 16, generator=g)
        w1, w2 = torch.randn(16, 16, generator=g) * 0.3, torch.randn(16, 16, generator=g) * 0.3
        lora = torch.randn(16, 16, generator=g).mul_(0.1).requires_grad_(True)
        ref_blk = lambda x: F.linear(F.gelu(F.linear(x, w1) + F.linear(x, lora)), w2)
        blk = lambda x: _lin(F.gelu(_lin(x, w1, fake_cuda) + _lin(x, lora, fake_cuda)), w2, fake_cuda)
    else:
        x0 = torch.randn(1, 4, 5, 5, generator=g)
        w1, w2 = torch.randn(4, 4, 3, 3, generator=g) * 0.3, torch.randn(4, 4, 1, 1, generator=g) * 0.3
        lora = torch.randn(4, 4, 3, 3, generator=g).mul_(0.1).requires_grad_(True)
        ref_blk = lambda x: F.conv2d(F.gelu(F.conv2d(x, w1, padding=1) + F.conv2d(x, lora, padding=1)), w2)
        blk = lambda x: _conv(F.gelu(_conv(x, w1, fake_cuda, (1, 1)) + _conv(x, lora, fake_cuda, (1, 1))), w2,
                              fake_cuda, (0, 0))
    xr, xg = x0.clone().requires_grad_(True), x0.clone().requires_grad_(True)
    ref_blk(xr).pow(2).sum().backward()
    ref = (xr.grad.clone(), lora.grad.clone())
    lora.grad = None
    checkpoint(blk, xg, use_reentrant=False).pow(2).sum().backward()
    assert torch.allclose(xg.grad, ref[0], rtol=1e-5, atol=1e-6)
    assert torch.allclose(lora.grad, ref[1], rtol=1e-5, atol=1e-6)
