"""Offload audit (26 Oct 2026): the toolkit's real LoRA network under layer_offloading must give the same gradients as
without offloading, across what H3 / MOSS / Krea2 training uses.

Built like BaseSDTrainProcess builds it for H3 (network type locon, full_locon, use_scalar, conv LoCon): a toy model of
transformer-style blocks (LayerNorm + bf16 linears) and a conv block, the base offloaded first (fully or partially),
then a LoRASpecialNetwork made on CPU in fp32, applied, and offloaded. Every trainable parameter's gradient
(LoCon factors, conv factors, scalars, norm deltas) is compared to the same build without offloading, for no
checkpointing, non-reentrant and reentrant checkpointing, and 1 or 4 accumulated micro-batches.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.memory_management import MemoryManager, sync_grad_transfers
from toolkit.memory_management import manager_modules

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="layer offloading needs CUDA")
DEV = torch.device("cuda")
D, R, CR = 128, 16, 4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(3 * D, D)

    def forward(self, x):
        return x + self.out(F.gelu(self.qkv(self.norm(x))))


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(D, D, 3, padding=1)

    def forward(self, x):
        b, t, d = x.shape
        side = int(t ** 0.5)
        y = self.conv(x.transpose(1, 2).reshape(b, d, side, side))
        return x + y.reshape(b, d, t).transpose(1, 2)


class Toy(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), ConvBlock(), Block()])
        self.ckpt = ckpt

    def forward(self, x):
        for blk in self.blocks:
            if self.ckpt is None:
                x = blk(x)
            else:
                x = checkpoint(blk, x, use_reentrant=self.ckpt == "reentrant")
        return x


def build(base_offload, net_offload, ckpt):
    torch.manual_seed(0)
    model = Toy(ckpt).to(torch.bfloat16).requires_grad_(False)
    if base_offload:
        MemoryManager.attach(model, DEV, offload_percent=base_offload)
    else:
        model.to(DEV)
    cfg = NetworkConfig(type="locon", linear=R, linear_alpha=2 * R, conv=CR, conv_alpha=2 * CR,
                        network_kwargs={"full_locon": True, "use_scalar": True, "scalar_init_value": 0.1})
    net = LoRASpecialNetwork(None, model, multiplier=1.0, lora_dim=R, alpha=2 * R, conv_lora_dim=CR, conv_alpha=2 * CR,
                             train_text_encoder=False, train_unet=True, target_lin_modules=["Block", "ConvBlock"],
                             target_conv_modules=["ConvBlock"], network_type="locon", is_transformer=True,
                             network_config=cfg, full_locon=True, use_scalar=True, scalar_init_value=0.1)
    net.force_to(torch.device("cpu") if net_offload else DEV, dtype=torch.float32)
    net._update_torch_multiplier()
    net.apply_to(None, model, False, True)
    net.prepare_grad_etc(None, model)  # as BaseSDTrainProcess does before attaching the offload manager
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in sorted(net.named_parameters()):
            if "lora_up" in n:  # zero-init would leave lora_down without a gradient
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    if net_offload:
        MemoryManager.attach(net, DEV)
        net._update_torch_multiplier()
    return model, net


def grads(base_offload, net_offload, ckpt, micro_batches):
    model, net = build(base_offload, net_offload, ckpt)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 16, D, generator=g).to(DEV, torch.bfloat16)
    y = torch.randn(4, 16, D, generator=g).to(DEV)
    for idx in torch.arange(4).chunk(micro_batches):
        xi = x[idx].clone().requires_grad_(ckpt == "reentrant")  # reentrant checkpointing needs a grad-requiring input
        with net:  # the trainer keeps forward and backward inside the network context (LoRA active)
            (F.mse_loss(model(xi).float(), y[idx]) / micro_batches).backward()
    sync_grad_transfers()
    out = {n: p.grad.detach().float().cpu().clone() for n, p in net.named_parameters() if p.requires_grad and p.grad is not None}
    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    return out, missing


def rel_errors(a, b):
    return {k: ((a[k] - b[k]).norm() / b[k].norm().clamp_min(1e-12)).item() for k in b}


@pytest.mark.parametrize("micro_batches", [1, 4])
@pytest.mark.parametrize("ckpt", [None, "nonreentrant", "reentrant"])
@pytest.mark.parametrize("base_offload,net_offload", [(None, True), (1.0, False), (1.0, True), (0.5, True)],
                         ids=["net", "base", "base+net", "halfbase+net"])
def test_real_network_offload_grads_match(base_offload, net_offload, ckpt, micro_batches):
    ref, ref_missing = grads(None, False, ckpt, micro_batches)
    got, got_missing = grads(base_offload, net_offload, ckpt, micro_batches)
    assert ref, "no trainable gradients in the reference"
    assert set(got) == set(ref) and got_missing == ref_missing, (sorted(set(ref) ^ set(got)), got_missing, ref_missing)
    assert any("scalar" in k for k in ref), "scalar not exercised"
    bad = {k: e for k, e in rel_errors(got, ref).items() if e > 2e-3}
    assert not bad, f"{len(bad)}/{len(ref)} params off: " + ", ".join(f"{k}={e:.3g}" for k, e in sorted(bad.items())[:6])


@pytest.mark.parametrize("base_offload", [None, 1.0], ids=["net", "base+net"])
def test_staged_grads_survive_a_slow_copy(monkeypatch, base_offload):
    """Regression (26 Oct 2026): conv2d_weight can return a channels_last grad. Returned as is, autograd can't adopt it
    as .grad (its strides differ from the param's) and copies it on the host at once, before the async D2H has landed,
    so a conv LoCon factor's first-micro-batch grad came out as zeros or garbage, now and then (3 in 12 audit runs).
    Holding the grad stream back before each copy makes that race deterministic."""
    stage = manager_modules._stage_grads_to_cpu

    def slow_copy(state, *args):
        with torch.cuda.stream(state["transfer_grad_stream"]):
            torch.cuda._sleep(50_000_000)
        return stage(state, *args)

    monkeypatch.setattr(manager_modules, "_stage_grads_to_cpu", slow_copy)
    ref, _ = grads(None, False, None, 1)
    got, _ = grads(base_offload, True, None, 1)
    bad = {k: e for k, e in rel_errors(got, ref).items() if e > 2e-3}
    assert not bad, f"{len(bad)}/{len(ref)} params off: " + ", ".join(f"{k}={e:.3g}" for k, e in sorted(bad.items())[:6])
