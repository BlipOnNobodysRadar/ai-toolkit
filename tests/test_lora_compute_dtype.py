"""Opt-in LoRA compute dtype (deep dive 1 Nov 2026, network-3): network.compute_dtype=bf16 runs plain linear LoRA
GEMMs in bf16 with fp32 masters and fp32 grads, skipping the per-module bf16->fp32 input copy and fp32->bf16 output
cast. Default (None) must be bit-identical to the fp32 path; offloaded networks, conv and non-LoRAModule classes
stay fp32. CPU only.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork

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


def build(compute_dtype=None, multiplier=1.0, use_scalar=False, alpha_mult=2.0, layer_offloading=False):
    torch.manual_seed(0)
    model = Toy().to(torch.bfloat16).requires_grad_(False)
    kwargs = {"full_locon": True}
    if use_scalar:
        kwargs.update(use_scalar=True, scalar_init_value=0.1)
    cfg = NetworkConfig(type="locon", linear=R, linear_alpha=alpha_mult * R, conv=CR, conv_alpha=alpha_mult * CR,
                        respect_alpha=True, compute_dtype=compute_dtype, layer_offloading=layer_offloading,
                        network_kwargs=dict(kwargs))
    net = LoRASpecialNetwork(None, model, multiplier=1.0, lora_dim=R, alpha=alpha_mult * R, conv_lora_dim=CR,
                             conv_alpha=alpha_mult * CR, train_text_encoder=False, train_unet=True,
                             target_lin_modules=["Block", "ConvBlock"], target_conv_modules=["ConvBlock"],
                             network_type="locon", is_transformer=True, network_config=cfg, **kwargs)
    net.force_to(torch.device("cpu"), dtype=torch.float32)
    net._update_torch_multiplier()
    net.multiplier = multiplier
    net.apply_to(None, model, False, True)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in sorted(net.named_parameters()):
            if "lora_up" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model, net


def run(model, net, seed=1):
    for p in net.parameters():
        p.grad = None
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 16, D, generator=g).to(torch.bfloat16)
    y = torch.randn(4, 16, D, generator=g)
    with net:
        out = model(x)
        F.mse_loss(out.float(), y).backward()
    return out.detach().float(), {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}


def lora_linears(net):
    return [m for m in net.get_all_modules() if isinstance(m.lora_down, nn.Linear)]


def manual_reference(model, net, x, mult):
    # the fp32 path written out by hand: base(x) + ((up(down(x.float())) * scale) * mult).to(bf16)
    outs = {}
    for m in lora_linears(net):
        base = m.org_module[0]
        hook_in = {}

        def grab(mod, inp, out, hook_in=hook_in):
            hook_in["x"] = inp[0]
            hook_in["out"] = out
        h = base.register_forward_hook(grab)
        outs[m.lora_name] = (m, base, hook_in, h)
    with torch.no_grad(), net:
        model(x)
    for m, base, hook_in, h in outs.values():
        h.remove()
        xi = hook_in["x"]
        org = F.linear(xi, base.weight, base.bias)
        lx = m.lora_up(m.lora_down(xi.float()))
        expected = org + broadcast(lx * m._runtime_scale, mult).to(org.dtype)
        assert torch.equal(hook_in["out"], expected), m.lora_name


def broadcast(t, mult):
    mt = torch.tensor(mult if isinstance(mult, list) else [mult], dtype=torch.float32)
    while mt.dim() < t.dim():
        mt = mt.unsqueeze(-1)
    return t * mt


@pytest.mark.parametrize("multiplier", [1.0, [1.0, 0.5, 1.0, 0.5]])
def test_default_is_fp32_path_bit_identical(multiplier):
    model, net = build(multiplier=multiplier)
    assert net.network_config.compute_dtype is None
    x = torch.randn(4, 16, D, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    manual_reference(model, net, x, multiplier)


class RecordLinear(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.dtypes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.linear:
            self.dtypes.append((args[0].dtype, args[1].dtype))
        return func(*args, **kwargs)


@pytest.mark.parametrize("use_scalar", [False, True])
@pytest.mark.parametrize("multiplier", [1.0, [1.0] * 4, [1.0, 0.5, 1.0, 0.5], 0.7])
def test_bf16_close_to_fp32_with_fp32_grads(use_scalar, multiplier):
    ref_model, ref_net = build(None, multiplier, use_scalar)
    ref_out, ref_g = run(ref_model, ref_net)
    model, net = build("bf16", multiplier, use_scalar)
    rec = RecordLinear()
    with rec:
        out, grads = run(model, net)
    assert all(a == torch.bfloat16 and w == torch.bfloat16 for a, w in rec.dtypes), set(rec.dtypes)
    assert torch.allclose(out, ref_out, rtol=2e-2, atol=2e-2)
    lin_names = {m.lora_name for m in lora_linears(net)}
    assert ref_g.keys() == grads.keys()
    for k in grads:
        assert grads[k].dtype == torch.float32, k
        rel = (grads[k] - ref_g[k]).norm() / ref_g[k].norm().clamp_min(1e-12)
        # a 0-d scalar's grad is one bf16 reduction over the whole output: looser than the factor grads
        assert rel < (6e-2 if k.endswith("scalar") else 3e-2), (k, float(rel))
    assert lin_names


def test_conv_and_offloaded_stay_fp32():
    _, net = build("bf16")
    x = torch.zeros(1, 4, D, dtype=torch.bfloat16)
    convs = [m for m in net.get_all_modules() if isinstance(m.lora_down, nn.Conv2d)]
    assert convs and all(m._lora_compute_dtype(net, x) is None for m in convs)
    assert all(m._lora_compute_dtype(net, x) is torch.bfloat16 for m in lora_linears(net))
    _, off = build("bf16", layer_offloading=True)
    assert all(m._lora_compute_dtype(off, x) is None for m in off.get_all_modules())


def test_configured_dtype_wins_over_activation_dtype():
    # DeepSeek review B3: compute_dtype fp16 used to follow bf16 activations and run as bf16.
    _, net = build("fp16")
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        x = torch.zeros(1, 4, D, dtype=dtype)
        assert all(m._lora_compute_dtype(net, x) is torch.float16 for m in lora_linears(net))
    _, net = build("bf16")
    x = torch.zeros(1, 4, D, dtype=torch.float16)
    assert all(m._lora_compute_dtype(net, x) is torch.bfloat16 for m in lora_linears(net))


def test_rank_dropout_still_applies():
    model, net = build("bf16")
    base_out, _ = run(model, net)
    for m in net.get_all_modules():
        m.rank_dropout = 0.5
    torch.manual_seed(5)
    drop_out, grads = run(model, net)
    assert not torch.equal(base_out, drop_out)
    assert all(g.dtype == torch.float32 for g in grads.values())


def test_config_validation():
    assert NetworkConfig(type="lora", compute_dtype="bfloat16").compute_dtype == "bf16"
    assert NetworkConfig(type="lora").compute_dtype is None
    with pytest.raises(ValueError):
        NetworkConfig(type="lora", compute_dtype="int8")
