# From Astra (Codex) review branch codex/moss-sdpa-review-20261102 d629ba7, adapted to the per-model priority (e025d4a).
"""Independent CPU contracts for backend opt-outs and MOSS attention math."""
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from extensions_built_in.diffusion_models.moss_image.src import transformer as moss


def _enabled():
    return tuple(getattr(torch.backends.cuda, f"{name}_sdp_enabled")()
                 for name in ("cudnn", "flash", "mem_efficient", "math"))


@pytest.fixture(autouse=True)
def restore_policy():
    # Installed torch restores only the ordering of enabled backends. Preserve
    # the full priority list so a failed regression cannot taint later tests.
    order = torch._C._get_sdp_priority_order()
    yield
    torch._C._set_sdp_priority_order(order)


def test_explicit_math_only_context_is_respected(monkeypatch):
    real = F.scaled_dot_product_attention
    observed = []

    def spy(*args, **kwargs):
        observed.append(_enabled())
        return real(*args, **kwargs)

    monkeypatch.setattr(moss.F, "scaled_dot_product_attention", spy)
    q = torch.ones(1, 2, 3, 8)
    with sdpa_kernel(SDPBackend.MATH):
        before = _enabled()
        moss._sdpa(q, q, q, priority=moss.CUDNN_FIRST)
        assert _enabled() == before
    assert observed == [(False, False, False, True)]


def test_disabled_cudnn_stays_disabled_inside_attention(monkeypatch):
    observed = []
    real = F.scaled_dot_product_attention

    def spy(*args, **kwargs):
        observed.append(torch.backends.cuda.cudnn_sdp_enabled())
        return real(*args, **kwargs)

    monkeypatch.setattr(moss.F, "scaled_dot_product_attention", spy)
    q = torch.ones(1, 2, 3, 8)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                      SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        moss._sdpa(q, q, q, priority=moss.CUDNN_FIRST)
    assert observed == [False]


@pytest.mark.parametrize("layout", ["transposed", "fused_qkv", "sliced"])
@pytest.mark.parametrize("mask_kind", ["padding", "isolation"])
def test_noncontiguous_masked_attention_matches_explicit_reference(layout, mask_kind):
    torch.manual_seed(147)
    batch, heads, seq, dim = 2, 3, 7, 8
    if layout == "fused_qkv":
        packed = torch.randn(batch, seq, 3 * heads * dim, dtype=torch.float64)
        parts = [t.reshape(batch, seq, heads, dim).transpose(1, 2)
                 for t in packed.chunk(3, -1)]
    else:
        parts = [torch.randn(batch, seq * (2 if layout == "sliced" else 1),
                             heads, dim, dtype=torch.float64).transpose(1, 2)
                 for _ in range(3)]
        if layout == "sliced":
            parts = [t[:, :, ::2] for t in parts]
    assert all(not t.is_contiguous() for t in parts)
    q, k, v = [t.detach().requires_grad_() for t in parts]
    if mask_kind == "padding":
        mask = (torch.arange(seq)[None] < torch.tensor([[7], [4]]))[:, None, None]
    else:
        refs = torch.arange(seq) >= 4
        mask = (~refs[:, None] | refs[None, :])[None, None]
    weight = torch.randn(batch, heads, seq, dim, dtype=torch.float64)
    actual = moss._sdpa(q, k, v, mask, priority=moss.CUDNN_FIRST)
    grads = torch.autograd.grad((actual * weight).sum(), (q, k, v))
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    logits = (qr @ kr.transpose(-2, -1)) / dim ** 0.5
    expected = logits.masked_fill(~mask, -torch.inf).softmax(-1) @ vr
    expected_grads = torch.autograd.grad((expected * weight).sum(), (qr, kr, vr))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for actual_grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)
    if mask_kind == "padding":
        assert torch.count_nonzero(grads[1][1, :, 4:]) == 0
        assert torch.count_nonzero(grads[2][1, :, 4:]) == 0
