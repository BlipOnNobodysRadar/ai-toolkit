"""CPU-master copies (deep dive 1 Nov 2026, network-2): under network layer offloading the LoCon `scalar` and the
full-module norm deltas (`diff`, `diff_b`) are pinned CPU masters that every forward copies to the device. A
blocking copy forces a stream sync each time (832 per H3 step), so those copies must pass non_blocking=True.
On CPU non_blocking is a no-op: outputs and grads stay bit-identical, and the test checks the flag on the calls.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork

D, R = 32, 8


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(3 * D, D)

    def forward(self, x):
        return x + self.out(F.gelu(self.qkv(self.norm(x))))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def build():
    torch.manual_seed(0)
    model = Toy().to(torch.bfloat16).requires_grad_(False)
    kwargs = {"full_locon": True, "use_scalar": True, "scalar_init_value": 0.1}
    cfg = NetworkConfig(type="locon", linear=R, linear_alpha=2 * R, respect_alpha=True, all_layers=True,
                        network_kwargs=dict(kwargs))
    net = LoRASpecialNetwork(None, model, multiplier=1.0, lora_dim=R, alpha=2 * R, train_text_encoder=False,
                             train_unet=True, target_lin_modules=["Block"], network_type="locon", is_transformer=True,
                             network_config=cfg, **kwargs)
    net.force_to(torch.device("cpu"), dtype=torch.float32)
    net._update_torch_multiplier()
    net.apply_to(None, model, False, True)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in sorted(net.named_parameters()):
            if "lora_up" in n or n.endswith("diff") or n.endswith("diff_b"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model, net


class RecordTo(TorchFunctionMode):
    def __init__(self, watched):
        super().__init__()
        self.watched = watched
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.to and args and id(args[0]) in self.watched:
            self.calls.append((self.watched[id(args[0])], bool(kwargs.get("non_blocking", False))))
        return func(*args, **kwargs)


def test_cpu_master_copies_are_non_blocking():
    model, net = build()
    masters = {}
    for name, p in net.named_parameters():
        if name.split(".")[-1] in ("scalar", "diff", "diff_b"):
            masters[id(p)] = name
    kinds = {n.split(".")[-1] for n in masters.values()}
    assert {"scalar", "diff", "diff_b"} <= kinds
    x = torch.randn(2, 8, D).to(torch.bfloat16)
    rec = RecordTo(masters)
    with net, rec:
        out = model(x)
    out.float().pow(2).mean().backward()
    seen = {name for name, _ in rec.calls}
    assert {n.split(".")[-1] for n in seen} == {"scalar", "diff", "diff_b"}
    blocking = sorted(name for name, nb in rec.calls if not nb)
    assert not blocking, f"blocking CPU-master copies: {blocking}"
    for name, p in net.named_parameters():
        if name in seen:
            assert p.grad is not None and p.grad.dtype == torch.float32, name


def test_outputs_and_grads_unchanged_by_flag():
    # non_blocking must not change numbers: compare against copies made without it
    model, net = build()
    x = torch.randn(2, 8, D).to(torch.bfloat16)

    def run():
        for p in net.parameters():
            p.grad = None
        with net:
            out = model(x)
        out.float().pow(2).mean().backward()
        return out.detach().clone(), {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    a_out, a_g = run()

    class ForceBlocking(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func is torch.Tensor.to:
                kwargs.pop("non_blocking", None)
            return func(*args, **kwargs)

    with ForceBlocking():
        b_out, b_g = run()
    assert torch.equal(a_out, b_out)
    assert a_g.keys() == b_g.keys() and len(a_g) > 0
    for k in a_g:
        assert torch.equal(a_g[k], b_g[k]), k
