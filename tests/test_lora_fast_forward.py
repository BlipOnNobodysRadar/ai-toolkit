"""LoRA forward fast path (deep dive 1 Nov 2026, network-1): with a uniform multiplier and a power of two
alpha/rank, LoRAModule applies the scale on the rank side (up(down(x) * s)) instead of multiplying the full-size
output twice. Outputs and every gradient must be bit-identical to the general path, and anything the fast path
cannot reproduce exactly (per-sample multipliers, lora_up bias, non power of two scales, dropout) must fall back.
CPU only.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit import network_mixins

D, R, CR = 32, 8, 4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(3 * D, D)

    def forward(self, x):
        return x + self.out(F.gelu(self.qkv(x)))


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(D, D, 3, padding=1)

    def forward(self, x):
        b, t, d = x.shape
        img = x.transpose(1, 2).reshape(b, d, 4, t // 4)
        return x + self.conv(img).reshape(b, d, t).transpose(1, 2)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), ConvBlock(), Block()])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def build(alpha_mult=2.0, use_scalar=False, use_bias=False, multiplier=1.0, base_dtype=torch.bfloat16):
    torch.manual_seed(0)
    model = Toy().to(base_dtype).requires_grad_(False)
    kwargs = {"full_locon": True}
    if use_scalar:
        kwargs.update(use_scalar=True, scalar_init_value=0.1)
    cfg = NetworkConfig(type="locon", linear=R, linear_alpha=alpha_mult * R, conv=CR, conv_alpha=alpha_mult * CR,
                        respect_alpha=True, network_kwargs=dict(kwargs))  # both of Ryan's configs respect alpha
    net = LoRASpecialNetwork(None, model, multiplier=1.0, lora_dim=R, alpha=alpha_mult * R,
                             conv_lora_dim=CR, conv_alpha=alpha_mult * CR, train_text_encoder=False, train_unet=True,
                             target_lin_modules=["Block", "ConvBlock"], target_conv_modules=["ConvBlock"],
                             network_type="locon", is_transformer=True, network_config=cfg, use_bias=use_bias,
                             **kwargs)
    net.force_to(torch.device("cpu"), dtype=torch.float32)
    net._update_torch_multiplier()
    net.multiplier = multiplier  # after the modules exist, as the trainer sets it
    net.apply_to(None, model, False, True)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in sorted(net.named_parameters()):
            if "lora_up" in n:  # zero-init would leave lora_down without a gradient
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    assert all(m.scale == alpha_mult for m in net.get_all_modules())
    return model, net


def run(model, net, force_fallback, batch=4):
    for p in net.parameters():
        p.grad = None
    saved = net._uniform_multiplier
    if force_fallback:
        net._uniform_multiplier = None
    try:
        g = torch.Generator().manual_seed(1)
        x = torch.randn(batch, 16, D, generator=g).to(next(model.parameters()).dtype).requires_grad_(True)
        y = torch.randn(batch, 16, D, generator=g)
        with net:
            out = model(x)
            F.mse_loss(out.float(), y).backward()
    finally:
        net._uniform_multiplier = saved
    grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    return out.detach().clone(), x.grad.clone(), grads


def spy_general_path(monkeypatch):
    calls = {"general": 0}
    orig = network_mixins.ToolkitModuleMixin._call_forward

    def spy(self, x):
        calls["general"] += 1
        return orig(self, x)

    monkeypatch.setattr(network_mixins.ToolkitModuleMixin, "_call_forward", spy)
    return calls


def assert_same(a, b):
    out_a, xg_a, g_a = a
    out_b, xg_b, g_b = b
    assert torch.equal(out_a, out_b)
    assert torch.equal(xg_a, xg_b)
    assert g_a.keys() == g_b.keys() and len(g_a) > 0
    for k in g_a:
        assert torch.equal(g_a[k], g_b[k]), k


def n_lora_modules(net):
    return sum(1 for m in net.get_all_modules() if m.__class__.__name__ == "LoRAModule")


@pytest.mark.parametrize("use_scalar", [False, True])
@pytest.mark.parametrize("multiplier", [1.0, [1.0] * 4, 0.5, -1.0])
@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_fast_path_bit_identical(monkeypatch, use_scalar, multiplier, base_dtype):
    model, net = build(use_scalar=use_scalar, multiplier=multiplier, base_dtype=base_dtype)
    assert net._uniform_multiplier is not None
    slow = run(model, net, force_fallback=True)
    calls = spy_general_path(monkeypatch)
    fast = run(model, net, force_fallback=False)
    assert calls["general"] == 0, "fast path not taken"
    assert_same(fast, slow)
    if use_scalar:
        assert any(k.endswith("scalar") for k in fast[2])


def test_trainer_style_list_multiplier(monkeypatch):
    # SDTrainer sets network.multiplier = batch.get_network_weight_list() -> [1.0] * B
    model, net = build(use_scalar=True)
    net.multiplier = [1.0] * 4
    assert net._uniform_multiplier == 1.0
    slow = run(model, net, force_fallback=True)
    calls = spy_general_path(monkeypatch)
    fast = run(model, net, force_fallback=False)
    assert calls["general"] == 0
    assert_same(fast, slow)


@pytest.mark.parametrize("case", ["per_sample", "bias", "non_pow2_scale", "non_pow2_mult", "direct_assign"])
def test_fallbacks(monkeypatch, case):
    kw = {}
    if case == "per_sample":
        kw["multiplier"] = [1.0, 0.5, 1.0, 0.5]
    elif case == "bias":
        kw["use_bias"] = True
    elif case == "non_pow2_scale":
        kw["alpha_mult"] = 1.5
    elif case == "non_pow2_mult":
        kw["multiplier"] = 0.7
    model, net = build(**kw)
    if case == "bias":
        assert any(getattr(m.lora_up, "bias", None) is not None for m in net.get_all_modules())
    if case == "direct_assign":
        net.torch_multiplier = torch.ones(1)  # as testing/test_lora_compile_scalars.py does
    calls = spy_general_path(monkeypatch)
    run(model, net, force_fallback=False)
    assert calls["general"] == n_lora_modules(net), "must take the general path"


def test_rank_dropout_falls_back_in_training(monkeypatch):
    model, net = build()
    for m in net.get_all_modules():
        m.rank_dropout = 0.1
    calls = spy_general_path(monkeypatch)
    run(model, net, force_fallback=False)
    assert calls["general"] == n_lora_modules(net)
