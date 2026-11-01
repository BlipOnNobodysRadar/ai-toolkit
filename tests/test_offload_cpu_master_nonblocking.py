"""offload-1 (1 Nov 2026): the CPU-master scalar (use_scalar) and the FullModule norm deltas (all_layers) are pinned
CPU tensors under network offload, and the forward copied them to the device with a blocking .to(), which on CUDA is a
memcpy followed by a stream sync (832 syncs per H3 step, each idling the GPU before the next offloaded weight copy).
They are now copied with non_blocking=True. CPU-only check: the forward asks for non_blocking copies of exactly those
masters, and the gradients are unchanged.
"""
import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import FullModule, LoRASpecialNetwork

D, R = 16, 4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(3 * D, D)

    def forward(self, x):
        return x + self.out(torch.nn.functional.gelu(self.qkv(self.norm(x))))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def build():
    torch.manual_seed(0)
    model = Toy().requires_grad_(False)
    cfg = NetworkConfig(type="locon", linear=R, linear_alpha=2 * R, all_layers=True,
                        network_kwargs={"use_scalar": True, "scalar_init_value": 0.1})
    net = LoRASpecialNetwork(None, model, multiplier=1.0, lora_dim=R, alpha=2 * R, train_text_encoder=False,
                             train_unet=True, target_lin_modules=["Block"], network_type="locon", is_transformer=True,
                             network_config=cfg, use_scalar=True, scalar_init_value=0.1)
    net.force_to(torch.device("cpu"), dtype=torch.float32)
    net._update_torch_multiplier()
    net.apply_to(None, model, False, True)
    net.prepare_grad_etc(None, model)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in sorted(net.named_parameters()):
            if "diff" in n or "lora_up" in n:  # nonzero so every grad path is exercised
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model, net


class RecordTo(TorchFunctionMode):
    def __init__(self, watch):
        super().__init__()
        self.watch = {id(t): n for n, t in watch.items()}
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.to and args and id(args[0]) in self.watch:
            self.calls.append((self.watch[id(args[0])], bool(kwargs.get("non_blocking", False))))
        return func(*args, **kwargs)


def test_cpu_masters_copied_non_blocking_and_grads_match():
    model, net = build()
    assert any(isinstance(m, FullModule) for m in net.modules()), "all_layers made no FullModule"
    masters = {n: p for n, p in net.named_parameters() if n.endswith(("scalar", "diff", "diff_b"))}
    assert any(n.endswith("scalar") for n in masters) and any(n.endswith("diff") for n in masters)

    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 5, D, generator=g)
    y = torch.randn(2, 5, D, generator=g)
    rec = RecordTo(masters)
    with net, rec:
        torch.nn.functional.mse_loss(model(x), y).backward()
    seen = {n for n, _ in rec.calls}
    assert seen == set(masters), sorted(set(masters) ^ seen)
    blocking = sorted({n for n, nb in rec.calls if not nb})
    assert not blocking, f"blocking copies of CPU masters: {blocking[:4]}"
    got = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    # reference: the same build with plain autograd and no recording; non_blocking is a no-op on CPU, so the
    # gradients must be bit-identical
    model2, net2 = build()
    with net2:
        torch.nn.functional.mse_loss(model2(x), y).backward()
    ref = {n: p.grad.clone() for n, p in net2.named_parameters() if p.grad is not None}
    assert set(got) == set(ref) and any(n.endswith("scalar") for n in ref)
    for n in ref:
        assert torch.equal(got[n], ref[n]), n
