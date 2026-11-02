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


# Module-behaviour contracts below are Astra's (internal review of 0cfcde1, 2 Nov 2026, task/fusion-review/
# test_fusion_independent.py), adopted: the first fused norm + affine read norm.weight and skipped every one of these.
def _astra_block(fused):
    torch.manual_seed(814)
    b = moss._Block(32, 80, 4)
    b.sdpa_priority = None
    for part in b.modules():
        if isinstance(part, (moss._Block, moss._RMSNorm)):
            part.fused_pointwise = fused
    return b


def _run_block(b, reference=False, dtype=torch.float32):
    torch.manual_seed(919)
    x = torch.randn(2, 7, 32, dtype=dtype, requires_grad=True)
    rope = torch.randn(2, 7, 8)
    mod = torch.randn(2, 6, 32, dtype=dtype, requires_grad=True)
    ref = torch.randn(2, 6, 32, dtype=dtype, requires_grad=True)
    out = b(x, rope, (mod, ref, 4) if reference else mod, None)
    out.float().square().mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in b.named_parameters() if p.grad is not None}
    grads.update(x=x.grad, mod=mod.grad)
    if reference:
        grads["reference_modulation"] = ref.grad
    return out.detach(), grads


def _exact_pair(a, b):
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    assert a[1].keys() == b[1].keys()
    for name in a[1]:
        torch.testing.assert_close(a[1][name], b[1][name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("kind", ["pre", "post", "backward", "subclass", "subclass_call"])
def test_norm_module_customizations_are_preserved(identity_compile, kind):
    results, events = [], []
    for fused in (False, True):
        b = _astra_block(fused)
        seen = []

        def pre(norm, args):
            seen.append("pre")
            return (args[0] + 0.3,)

        def post(norm, args, out):
            seen.append("post")
            return out * 1.3

        def backward(norm, gin, gout):
            seen.append("backward")
            return tuple(g * 1.3 if g is not None else None for g in gin)

        if kind == "pre":
            b.norm1.register_forward_pre_hook(pre)
        elif kind == "post":
            b.norm1.register_forward_hook(post)
        elif kind == "backward":
            b.norm1.register_full_backward_hook(backward)
        else:
            class ScaledNorm(moss._RMSNorm):
                def forward(self, x):
                    seen.append("subclass")
                    return super().forward(x) * 1.3

            class CustomCallNorm(moss._RMSNorm):  # Astra's probe_subclass_call.py: forward inherited
                def __call__(self, x):
                    seen.append("call")
                    return super().__call__(x) * 1.3

            replacement = (ScaledNorm if kind == "subclass" else CustomCallNorm)(32)
            replacement.load_state_dict(b.norm1.state_dict())
            replacement.fused_pointwise = fused
            b.norm1 = replacement
        results.append(_run_block(b))
        events.append(seen)
    assert events[0] == events[1], "fused norm bypassed module behavior"
    _exact_pair(*results)


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_outputs_inputs_modulation_and_parameter_grads(identity_compile, reference, dtype):
    _exact_pair(_run_block(_astra_block(False).to(dtype), reference, dtype),
                _run_block(_astra_block(True).to(dtype), reference, dtype))


def test_wrapped_delta_keeps_gradient_and_restores_weight(identity_compile):
    """What toolkit.lora_special.FullModule.apply_to does: an instance forward that swaps weight + delta in."""
    results = []
    for fused in (False, True):
        b = _astra_block(fused)
        base = b.norm1.weight
        delta = torch.nn.Parameter(torch.full_like(base, 0.2))
        original = b.norm1.forward

        def wrapped(x):
            b.norm1._parameters["weight"] = base + delta
            try:
                return original(x)
            finally:
                b.norm1._parameters["weight"] = base

        b.norm1.forward = wrapped
        result = _run_block(b)
        assert b.norm1.weight is base
        assert delta.grad is not None and delta.grad.abs().sum() > 0
        result[1]["trained_delta"] = delta.grad
        results.append(result)
    _exact_pair(*results)


def test_existing_per_op_casting_hook_preserved(identity_compile):
    from toolkit.util.mixed_precision import attach_per_op_casting
    results = []
    for fused in (False, True):
        b = _astra_block(fused).to(torch.bfloat16)
        b.norm1.float()
        b.norm2.float()
        assert attach_per_op_casting(b) > 0
        results.append(_run_block(b, dtype=torch.bfloat16))
    _exact_pair(*results)


def test_global_module_hook_preserved(identity_compile):
    seen = []
    handle = torch.nn.modules.module.register_module_forward_hook(
        lambda mod, args, out: seen.append(type(mod).__name__) or None)
    try:
        counts = []
        for fused in (False, True):
            seen.clear()
            _run_block(_astra_block(fused))
            counts.append(seen.count("_RMSNorm"))
    finally:
        handle.remove()
    assert counts[0] == counts[1] > 0


def test_aot_capture_block_forward_backward(monkeypatch):
    original_compile = torch.compile
    torch._dynamo.reset()
    monkeypatch.setattr(moss, "_COMPILED", {})
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: original_compile(fn, backend="aot_eager", **kw))
    eager, compiled = _run_block(_astra_block(False)), _run_block(_astra_block(True))
    torch.testing.assert_close(eager[0], compiled[0], rtol=1e-6, atol=1e-6)
    assert eager[1].keys() == compiled[1].keys()
    for name in eager[1]:
        torch.testing.assert_close(eager[1][name], compiled[1][name], rtol=1e-5, atol=1e-6, msg=name)
    torch._dynamo.reset()


def test_dynamo_previews_stay_eager_and_recompile_limit_falls_back(monkeypatch):
    """Astra's probe (task/fusion-review/probe_dynamo.py): two checkpointed training batches made 5 _rms_norm graphs
    and a no_grad batch-1 preview then hit the recompile limit, a hard error under fullgraph=True. Now previews run
    eager, and going past the limit (forced low here) falls back to eager instead of raising."""
    original_compile = torch.compile
    counts = {}

    def counting(fn, **kw):
        def backend(gm, inputs):
            counts[fn.__name__] = counts.get(fn.__name__, 0) + 1
            return gm.forward
        return original_compile(fn, backend=backend, **kw)

    torch._dynamo.reset()
    monkeypatch.setattr(moss, "_COMPILED", {})
    monkeypatch.setattr(torch, "compile", counting)
    from torch._dynamo.utils import counters
    counters.clear()
    model = _model(checkpointing=True, fused=True)
    _step(model, (9, 3, 3, 6))
    _step(model, (5, 13, 2, 5))
    trained = dict(counts)
    assert trained and not counters["graph_break"]
    with torch.no_grad():
        model(*_batch((7,)))
    assert counts == trained
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 1)
    _step(model, (8,))  # batch 1 specializes: new variants, over the forced limit
    _step(model, (9, 3, 3, 6))
    torch._dynamo.reset()


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
