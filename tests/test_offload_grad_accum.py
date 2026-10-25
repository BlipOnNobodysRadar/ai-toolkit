"""Gradient accumulation through layer_offloading (MemoryManager) must match the same model without offloading.

A toy network built like a toolkit LoRA: frozen bf16 base linears, each with fp32 lora_down/lora_up whose input is
cast to the LoRA dtype (network_mixins.LoRAModule). The base and the LoRA are both offloaded, as with
model.layer_offloading and network.layer_offloading, optionally under gradient checkpointing.

Regression (found 25 Oct 2026 on MOSS): with gradient checkpointing, ctx.saved_tensors hands backward an alias of
the weight rather than the Parameter, so `weight.grad is not None` never held and the sync before
AccumulateGrad's `grad += staged` was skipped. Every micro-batch after the first added a pinned buffer whose async
D2H copy had not landed. GA 1 was unaffected (no `+=`), as was GA without offloading.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from toolkit.memory_management import MemoryManager, sync_grad_transfers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="layer offloading needs CUDA")
DEV = torch.device("cuda")


class ToyLoRALayer(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.base = nn.Linear(d, d).to(torch.bfloat16).requires_grad_(False)
        self.lora_down = nn.Linear(d, r, bias=False)
        self.lora_up = nn.Linear(r, d, bias=False)
        nn.init.normal_(self.lora_up.weight, std=0.05)  # non-zero so lora_down gets a gradient too

    def forward(self, x):
        org = self.base(x)
        lora = self.lora_up(self.lora_down(x.to(self.lora_down.weight.dtype)))
        return org + lora.to(org.dtype)


class ToyNet(nn.Module):
    def __init__(self, d=512, r=64, n_layers=3, ckpt=False):
        super().__init__()
        self.layers = nn.ModuleList(ToyLoRALayer(d, r) for _ in range(n_layers))
        self.ckpt = ckpt

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(checkpoint(layer, x, use_reentrant=False) if self.ckpt else layer(x))
        return x


def build(offload, **kw):
    torch.manual_seed(0)
    net = ToyNet(**kw)
    if offload:
        MemoryManager.attach(nn.ModuleList(l.base for l in net.layers), DEV)
        MemoryManager.attach(nn.ModuleList(m for l in net.layers for m in (l.lora_down, l.lora_up)), DEV)
    else:
        net.to(DEV)
    return net


def data(d=512, batch=8):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 16, d, generator=g).to(DEV, torch.bfloat16)
    y = torch.randn(batch, 16, d, generator=g).to(DEV)
    return x, y


def accumulate(net, x, y, micro_batches):
    """One optimizer step's worth of backward passes, scaled like SDTrainer (loss / n per micro-batch)."""
    chunks = torch.arange(x.shape[0]).chunk(micro_batches)
    for idx in chunks:
        (F.mse_loss(net(x[idx]).float(), y[idx]) / len(chunks)).backward()
    sync_grad_transfers()
    return {n: p.grad.detach().float().cpu().clone() for n, p in net.named_parameters() if p.requires_grad}


def assert_close(a, b, what, tol):
    """Relative error per tensor, ||a - b|| / ||b||. Elementwise is too strict here: the bf16 base makes batch 8
    and 4 x 2 differ by ~2% on single elements even without offloading."""
    for k in a:
        err = ((a[k] - b[k]).norm() / b[k].norm().clamp_min(1e-12)).item()
        assert err < tol, f"{what}: {k}: relative error {err:.3g} (max {tol}); |a|={a[k].norm():.4g} |b|={b[k].norm():.4g}"


@pytest.mark.parametrize("ckpt", [False, True], ids=["no-ckpt", "ckpt"])
@pytest.mark.parametrize("micro_batches", [1, 2, 8])
def test_offloaded_accumulated_grads_match(micro_batches, ckpt):
    x, y = data()
    plain = accumulate(build(offload=False, ckpt=ckpt), x, y, micro_batches)
    off = accumulate(build(offload=True, ckpt=ckpt), x, y, micro_batches)
    # same micro-batching, same math: offloading may only add rounding noise
    assert_close(off, plain, f"offload vs none, {micro_batches} micro-batches, ckpt={ckpt}", tol=1e-3)
    # and accumulation itself reproduces one plain batch up to bf16 noise
    ref = accumulate(build(offload=False, ckpt=ckpt), x, y, micro_batches=1)
    assert_close(plain, ref, f"no offload, {micro_batches} micro-batches vs batch 8", tol=3e-2)


def test_offloaded_accumulation_r512_sized():
    """Multi-MB D2H copies (r512 on a 2048-wide layer), checkpointed, 8 micro-batches, like the MOSS run."""
    x, y = data(d=2048)
    kw = dict(d=2048, r=512, ckpt=True)
    plain = accumulate(build(offload=False, **kw), x, y, 8)
    off = accumulate(build(offload=True, **kw), x, y, 8)
    assert_close(off, plain, "r512-sized, checkpointed, 8 micro-batches: offload vs none", tol=1e-3)


@pytest.mark.parametrize("ckpt", [False, True], ids=["no-ckpt", "ckpt"])
def test_offloaded_accumulation_over_optimizer_steps(ckpt):
    """Two SGD steps with zero_grad(set_to_none=True) between them, like the trainer: weights must match."""
    x, y = data()
    finals = {}
    for offload in (False, True):
        net = build(offload, ckpt=ckpt)
        opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.5)
        for _ in range(2):
            accumulate(net, x, y, micro_batches=4)
            opt.step()
            opt.zero_grad(set_to_none=True)
        finals[offload] = {n: p.detach().float().cpu().clone() for n, p in net.named_parameters() if p.requires_grad}
    assert_close(finals[True], finals[False], f"weights after 2 steps, ckpt={ckpt}, offload vs none", tol=1e-3)
