"""MOSS _sdpa under restricting SDPA scopes (Astra internal review P2/P3, 2 Nov 2026). Adapted from Astra's
test_partial_bypass.py (internal-review), now against the production _sdpa. torch's sdpa_kernel(..., set_priority=
True) restores only the enabled backends' order on exit, so _sdpa reorders only when every listed backend is enabled:
under any restriction the caller's flags and full priority order come back unchanged, on normal and error exits.
P3 scope: the four standard backends; OVERRIDEABLE (PrivateUse1 devices only) is off during a reordered call."""
import itertools

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from extensions_built_in.diffusion_models.moss_image.src import transformer as moss

BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
            SDPBackend.CUDNN_ATTENTION, SDPBackend.OVERRIDEABLE]


def flags():
    c = torch._C
    return (c._get_flash_sdp_enabled(), c._get_mem_efficient_sdp_enabled(), c._get_math_sdp_enabled(),
            c._get_cudnn_sdp_enabled(), c._get_overrideable_sdp_enabled())


@pytest.fixture(autouse=True)
def restore_global_sdpa_state():
    order, saved = torch._C._get_sdp_priority_order(), flags()
    yield
    torch._C._set_sdp_priority_order(order)
    for setter, value in zip((torch._C._set_sdp_use_flash, torch._C._set_sdp_use_mem_efficient,
                              torch._C._set_sdp_use_math, torch._C._set_sdp_use_cudnn,
                              torch._C._set_sdp_use_overrideable), saved):
        setter(value)


@pytest.mark.parametrize("raise_in_call", [False, True])
def test_restricted_scopes_keep_flags_and_full_order(monkeypatch, raise_in_call):
    observed = []

    def fake(q, k, v, **kwargs):
        observed.append((flags(), torch._C._get_sdp_priority_order()))
        if raise_in_call:
            raise RuntimeError("injected")
        return q

    monkeypatch.setattr(moss.F, "scaled_dot_product_attention", fake)
    q = torch.ones(1, 2, 3, 8)
    initial = [1, 2, 0, 3, 4]
    for enabled in itertools.product((False, True), repeat=4):
        torch._C._set_sdp_priority_order(initial)
        before = flags()
        with sdpa_kernel([b for b, e in zip(BACKENDS[:4], enabled) if e]):
            inner = flags()
            if raise_in_call:
                with pytest.raises(RuntimeError, match="injected"):
                    moss._sdpa(q, q, q, priority=moss.CUDNN_FIRST)
            else:
                moss._sdpa(q, q, q, priority=moss.CUDNN_FIRST)
            if not all(enabled):  # restricted: the caller's policy, untouched
                assert observed[-1] == (inner, initial)
            assert flags() == inner and torch._C._get_sdp_priority_order() == initial
        assert flags() == before and torch._C._get_sdp_priority_order() == initial


def test_reordered_call_scope_is_the_four_standard_backends(monkeypatch):
    seen = []
    real = moss.F.scaled_dot_product_attention
    monkeypatch.setattr(moss.F, "scaled_dot_product_attention",
                        lambda *a, **kw: seen.append((flags(), torch._C._get_sdp_priority_order())) or real(*a, **kw))
    before = flags()
    assert before == (True,) * 5  # torch's defaults
    q = torch.randn(1, 2, 3, 8)
    moss._sdpa(q, q, q, priority=moss.CUDNN_FIRST)
    inner_flags, inner_order = seen[-1]
    assert inner_flags == (True, True, True, True, False)  # OVERRIDEABLE off only for the call
    assert inner_order[:4] == [int(b) for b in moss.CUDNN_FIRST]
    assert flags() == before
