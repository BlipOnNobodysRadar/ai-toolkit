"""MOSS attention runs with cuDNN first in SDPA's backend order (1-2 Nov 2026): Windows torch has no flash
attention, so the padding mask of a batch-4 training step ran on the mem-efficient kernel. The order is set per
model on its blocks (model_kwargs.sdpa_cudnn: false = torch's order), only reorders backends that are enabled, and a
padded batch still matches every sample run alone. CPU fp32 tiny DiT, plus CUDA checks on real MOSS strides:
cuDNN is the selected kernel for the masked calls, and outputs and grads agree with the mem-efficient kernel."""
import inspect

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from extensions_built_in.diffusion_models.moss_image.src import transformer as moss
from extensions_built_in.diffusion_models.moss_image.src.transformer import (
    CUDNN_FIRST,
    MossImageParams,
    MossImageTransformer,
)


def _model(checkpointing, dim=64, heads=4, device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    p = MossImageParams(dim=dim, ffn_dim=2 * dim, num_heads=heads, num_layers=4, text_dim=32, freq_dim=16,
                        latent_channels=8)
    m = MossImageTransformer(p).to(device=device, dtype=dtype).train()
    with torch.no_grad():
        for prm in m.parameters():
            prm.normal_(0, 0.05)
    m.gradient_checkpointing = checkpointing
    return m


def _padded_batch(lens, device="cpu", dtype=torch.float32):
    torch.manual_seed(1)
    ctx = torch.zeros(len(lens), max(lens), 32)
    mask = torch.zeros(len(lens), max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        ctx[i, :n] = torch.randn(n, 32)
        mask[i, :n] = True
    x = torch.randn(len(lens), 8, 4, 6)
    t = torch.tensor([900.0, 400.0, 650.0, 120.0][:len(lens)])
    return x.to(device, dtype), t.to(device), ctx.to(device, dtype), mask.to(device)


@pytest.fixture
def sdpa_calls(monkeypatch):
    """Records (priority order, cuDNN enabled) as each MOSS attention call runs."""
    calls = []
    real = moss.F.scaled_dot_product_attention

    def spy(*a, **kw):
        calls.append((torch._C._get_sdp_priority_order(), torch.backends.cuda.cudnn_sdp_enabled()))
        return real(*a, **kw)

    monkeypatch.setattr(moss.F, "scaled_dot_product_attention", spy)
    return calls


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("lens", [(9, 3, 3, 6), (5, 5, 2, 5)])
def test_padded_batch_matches_each_sample_alone(lens, checkpointing):
    m = _model(checkpointing)
    x, t, ctx, mask = _padded_batch(lens)
    weights = torch.randn(len(lens), 8, 4, 6)
    out = m(x, t, ctx, mask)
    (out * weights).sum().backward()
    batched = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    for i, n in enumerate(lens):
        alone = m(x[i:i + 1], t[i:i + 1], ctx[i:i + 1, :n], mask[i:i + 1, :n])
        torch.testing.assert_close(out[i:i + 1], alone, atol=1e-5, rtol=1e-4)
        (alone * weights[i:i + 1]).sum().backward()
    for name, p in m.named_parameters():
        if name in batched:
            torch.testing.assert_close(batched[name], p.grad, atol=1e-5, rtol=1e-4, msg=name)


def test_order_is_per_model_and_restored(sdpa_calls):
    default, opted_out = _model(True), _model(True)
    opted_out.set_sdpa_priority(None)
    x, t, ctx, mask = _padded_batch((9, 3, 3, 6))
    before = torch._C._get_sdp_priority_order()
    default(x, t, ctx, mask).sum().backward()  # checkpoint recompute included
    n = len(sdpa_calls)
    assert n and all(order[:4] == [int(b) for b in CUDNN_FIRST] for order, _ in sdpa_calls)
    opted_out(x, t, ctx, mask)
    assert len(sdpa_calls) > n and all(order == before for order, _ in sdpa_calls[n:])
    assert torch._C._get_sdp_priority_order() == before


def test_disabled_backend_stays_disabled(sdpa_calls):
    m = _model(False)
    x, t, ctx, mask = _padded_batch((9, 3, 3, 6))
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        m(x, t, ctx, mask)
    finally:
        torch.backends.cuda.enable_cudnn_sdp(True)
    assert sdpa_calls and not any(enabled for _, enabled in sdpa_calls)


def test_model_kwarg_wiring():
    from extensions_built_in.diffusion_models.moss_image import moss_image
    assert 'kw.get("sdpa_cudnn", True)' in inspect.getsource(moss_image.MossImageModel.__init__)
    assert "transformer.set_sdpa_priority(CUDNN_FIRST if self.sdpa_cudnn else None)" in inspect.getsource(
        moss_image.MossImageModel.load_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_padded_training_step_selects_cudnn_and_matches_mem_efficient(monkeypatch):
    # real MOSS layouts: q/k/v come out of a fused qkv projection, reshape, RoPE and transpose (non-contiguous)
    chosen = []
    real = moss.F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kw):
        chosen.append((attn_mask is not None, torch._fused_sdp_choice(q, k, v, attn_mask)))
        return real(q, k, v, attn_mask=attn_mask, **kw)

    monkeypatch.setattr(moss.F, "scaled_dot_product_attention", spy)
    results = {}
    for name, priority in (("cudnn", CUDNN_FIRST), ("efficient", (SDPBackend.EFFICIENT_ATTENTION,))):
        m = _model(True, dim=256, heads=2, device="cuda", dtype=torch.bfloat16)  # head dim 128 like MOSS
        m.set_sdpa_priority(priority)
        x, t, ctx, mask = _padded_batch((9, 3, 3, 6), "cuda", torch.bfloat16)
        x.requires_grad_(True)
        out = m(x, t, ctx, mask)
        out.float().square().mean().backward()
        results[name] = (out.float(), x.grad.float(), {n: p.grad.float() for n, p in m.named_parameters()
                                                       if p.grad is not None})
        if name == "cudnn":
            masked = [b for has_mask, b in chosen if has_mask]
            assert masked and all(b == int(SDPBackend.CUDNN_ATTENTION) for b in masked)
        chosen.clear()
    (o1, dx1, g1), (o2, dx2, g2) = results["cudnn"], results["efficient"]

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()

    assert rel(o1, o2) < 2e-2 and rel(dx1, dx2) < 5e-2
    worst = max(rel(g1[n], g2[n]) for n in g2 if g2[n].norm() > 0)
    assert worst < 5e-2 and all(torch.isfinite(g).all() for g in g1.values())
