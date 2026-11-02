"""MOSS attention runs with cuDNN first in SDPA's backend order (1-2 Nov 2026): Windows torch has no flash
attention, so the padding mask of a batch-4 training step ran on the mem-efficient kernel. A padded batch still
matches every sample run alone at its true text length, outputs and gradients. CPU fp32 tiny DiT, plus a CUDA
check that the padded call really lands on cuDNN."""
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from extensions_built_in.diffusion_models.moss_image.src import transformer as moss
from extensions_built_in.diffusion_models.moss_image.src.transformer import MossImageParams, MossImageTransformer


def _model(checkpointing):
    torch.manual_seed(0)
    p = MossImageParams(dim=64, ffn_dim=128, num_heads=4, num_layers=4, text_dim=32, freq_dim=16, latent_channels=8)
    m = MossImageTransformer(p).float().train()
    with torch.no_grad():
        for prm in m.parameters():
            prm.normal_(0, 0.05)
    m.gradient_checkpointing = checkpointing
    return m


def _padded_batch(lens):
    torch.manual_seed(1)
    ctx = torch.zeros(len(lens), max(lens), 32)
    mask = torch.zeros(len(lens), max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        ctx[i, :n] = torch.randn(n, 32)
        mask[i, :n] = True
    return torch.randn(len(lens), 8, 4, 6), torch.tensor([900.0, 400.0, 650.0, 120.0][:len(lens)]), ctx, mask


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


def test_attention_calls_run_cudnn_first(monkeypatch):
    orders = []
    real = moss.F.scaled_dot_product_attention

    def spy(*a, **kw):
        orders.append(torch._C._get_sdp_priority_order())
        return real(*a, **kw)

    monkeypatch.setattr(moss.F, "scaled_dot_product_attention", spy)
    m = _model(True)
    x, t, ctx, mask = _padded_batch((9, 3, 3, 6))
    before = torch._C._get_sdp_priority_order()
    m(x, t, ctx, mask).sum().backward()
    cudnn_first = [int(b) for b in moss.CUDNN_FIRST]
    assert orders and all(o[:4] == cudnn_first for o in orders)
    assert torch._C._get_sdp_priority_order() == before  # restored after each call
    monkeypatch.setattr(moss, "SDPA_PRIORITY", None)
    orders.clear()
    m(x, t, ctx, mask)
    assert orders and all(o == before for o in orders)


def test_model_kwarg_opts_out(monkeypatch):
    from extensions_built_in.diffusion_models.moss_image import moss_image
    source = __import__("inspect").getsource(moss_image)
    assert 'kw.get("sdpa_cudnn", True)' in source and "moss_transformer.SDPA_PRIORITY" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_padded_bf16_call_matches_forced_cudnn():
    torch.manual_seed(2)
    q, k, v = (torch.randn(2, 4, 300, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    keep = torch.arange(300, device="cuda")[None] < torch.tensor([[300], [211]], device="cuda")
    mask = keep[:, None, None, :]
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        forced = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.equal(moss._sdpa(q, k, v, attn_mask=mask), forced)
