"""MOSS pointwise fusion (deep dive models-3, 2 Nov 2026): model_kwargs.fused_pointwise compiles the blocks' pure
pointwise helpers (RMSNorm, norm + modulation, gate + residual, RoPE, situ) with torch.compile. CPU: with compile
swapped for identity, the fused code path is bit-identical to eager (both modulation forms, with and without
checkpointing), and a norm whose forward a network wraps (FullModule's trained delta) keeps that forward. CUDA: the
compiled model matches eager within bf16 rounding and does not recompile when the text length changes."""
import importlib.util
import inspect

import pytest
import torch

from extensions_built_in.diffusion_models.moss_image.src import transformer as moss
from extensions_built_in.diffusion_models.moss_image.src.transformer import MossImageParams, MossImageTransformer


def _model(checkpointing=False, fused=False, dim=64, heads=4, device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    p = MossImageParams(dim=dim, ffn_dim=2 * dim, num_heads=heads, num_layers=4, text_dim=32, freq_dim=16,
                        latent_channels=8)
    m = MossImageTransformer(p).to(device=device, dtype=dtype).train()
    with torch.no_grad():
        for prm in m.parameters():
            prm.normal_(0, 0.05)
    m.gradient_checkpointing = checkpointing
    m.set_fused_pointwise(fused)
    return m


def _batch(lens, device="cpu", dtype=torch.float32):
    torch.manual_seed(1)
    ctx = torch.zeros(len(lens), max(lens), 32)
    mask = torch.zeros(len(lens), max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        ctx[i, :n] = torch.randn(n, 32)
        mask[i, :n] = True
    x = torch.randn(len(lens), 8, 4, 6)
    t = torch.tensor([900.0, 400.0, 650.0, 120.0][:len(lens)])
    return x.to(device, dtype), t.to(device), ctx.to(device, dtype), mask.to(device)


def _step(m, lens, device="cpu", dtype=torch.float32):
    x, t, ctx, mask = _batch(lens, device, dtype)
    out = m(x, t, ctx, mask)
    (out.float() * torch.linspace(-1, 1, out.numel(), device=device).view_as(out)).sum().backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    return out.detach(), grads


@pytest.fixture
def identity_compile(monkeypatch):
    """torch.compile -> the function itself: runs the fused code path in eager mode."""
    monkeypatch.setattr(moss, "_COMPILED", {})
    monkeypatch.setattr(moss.torch, "compile", lambda fn, **kw: fn)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_fused_path_is_bit_identical_to_eager(identity_compile, checkpointing):
    eager_out, eager_grads = _step(_model(checkpointing), (9, 3, 3, 6))
    fused_out, fused_grads = _step(_model(checkpointing, fused=True), (9, 3, 3, 6))
    torch.testing.assert_close(fused_out, eager_out, atol=0, rtol=0)
    assert fused_grads.keys() == eager_grads.keys()
    for name, g in eager_grads.items():
        torch.testing.assert_close(fused_grads[name], g, atol=0, rtol=0, msg=name)


def _block_inputs(dim=64, heads=4, length=10, batch=2):
    torch.manual_seed(2)
    x = torch.randn(batch, length, dim)
    rope = torch.randn(batch, length, dim // heads)
    mod, ref_mod = torch.randn(batch, 6, dim) * 0.1, torch.randn(batch, 6, dim) * 0.1
    return x, rope, mod, ref_mod


def _block(fused):
    torch.manual_seed(3)
    block = moss._Block(64, 128, 4)
    with torch.no_grad():
        for prm in block.parameters():
            prm.normal_(0, 0.05)
    for module in block.modules():
        if isinstance(module, (moss._Block, moss._RMSNorm)):
            module.fused_pointwise = fused
    return block


def test_reference_modulation_path_is_bit_identical(identity_compile):
    x, rope, mod, ref_mod = _block_inputs()
    eager = _block(False)(x, rope, (mod, ref_mod, 6), None)
    fused = _block(True)(x, rope, (mod, ref_mod, 6), None)
    torch.testing.assert_close(fused, eager, atol=0, rtol=0)


def _wrap_like_full_module(norm, delta):
    """What toolkit.lora_special.FullModule.apply_to does: an instance forward that swaps weight + delta in."""
    org_forward = norm.forward

    def forward(x):
        weight = norm._parameters["weight"]
        norm._parameters["weight"] = weight + delta
        try:
            return org_forward(x)
        finally:
            norm._parameters["weight"] = weight

    norm.forward = forward


def test_wrapped_norm_keeps_its_forward(identity_compile):
    x, rope, mod, _ = _block_inputs()
    plain = _block(True)(x, rope, mod, None)
    outs = []
    for fused in (False, True):
        block = _block(fused)
        for norm in (block.norm1, block.norm2, block.q_norm):
            _wrap_like_full_module(norm, torch.full_like(norm.weight, 0.5))
        outs.append(block(x, rope, mod, None))
    torch.testing.assert_close(outs[1], outs[0], atol=0, rtol=0)
    assert not torch.allclose(outs[1], plain)


def test_set_fused_pointwise_reaches_every_block_and_norm():
    m = _model()
    targets = [mod for mod in m.modules() if isinstance(mod, (moss._TextBlock, moss._Block, moss._RMSNorm))]
    assert targets and not any(t.fused_pointwise for t in targets)
    m.set_fused_pointwise(True)
    assert all(t.fused_pointwise for t in targets)
    other = _model()
    assert not any(t.fused_pointwise for t in other.modules() if isinstance(t, moss._Block))


def test_model_kwarg_wiring():
    from extensions_built_in.diffusion_models.moss_image import moss_image
    assert 'kw.get("fused_pointwise", False)' in inspect.getsource(moss_image.MossImageModel.__init__)
    assert "transformer.set_fused_pointwise(self.fused_pointwise)" in inspect.getsource(
        moss_image.MossImageModel.load_model)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                    reason="needs CUDA and Triton")
def test_cuda_compiled_matches_eager_and_text_length_does_not_recompile():
    from torch._dynamo.utils import counters
    kw = dict(dim=256, heads=4, device="cuda", dtype=torch.bfloat16)
    eager_out, eager_grads = _step(_model(True, **kw), (9, 3, 3, 6), "cuda", torch.bfloat16)
    fused = _model(True, fused=True, **kw)
    fused_out, fused_grads = _step(fused, (9, 3, 3, 6), "cuda", torch.bfloat16)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

    assert rel(fused_out, eager_out) < 2e-2
    worst = max(rel(fused_grads[n], g) for n, g in eager_grads.items())
    assert worst < 5e-2, worst
    graphs = counters["stats"]["unique_graphs"]
    _step(fused, (5, 5, 2, 5), "cuda", torch.bfloat16)
    _step(fused, (12, 4, 7, 7), "cuda", torch.bfloat16)
    assert counters["stats"]["unique_graphs"] == graphs
