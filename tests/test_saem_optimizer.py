"""CPU-only tests for SimplifiedAdEMAMixExM ("SAEM") transfer and compile paths.

All tests force torch.cuda.is_available() to False so the optimizer computes on
the CPU even on a machine with a GPU.
"""
import pytest
import torch

from training_extras.vendor.LoraEasyCustomOptimizer import ademamix
from training_extras.vendor.LoraEasyCustomOptimizer.ademamix import SimplifiedAdEMAMixExM


SHAPES = [(8, 4), (4, 8), (16,), (3, 3, 2, 2)]


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _params(dtype=torch.float32, shapes=SHAPES, seed=0):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s).to(dtype)) for s in shapes]


def _set_grads(params, step):
    g = torch.Generator().manual_seed(100 + step)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g).to(p.dtype)


def test_cpu_param_write_back_is_non_blocking_when_pinned(monkeypatch):
    # Under network layer offload the fp32 LoRA params are pinned CPU tensors.
    # Their write-back must be an async D2H (the group-end synchronize orders
    # it); a blocking copy_ syncs the CUDA stream once per parameter.
    params = _params()
    opt = SimplifiedAdEMAMixExM(params, lr=1e-3, torch_compile=False)
    _set_grads(params, 0)
    opt.step()  # step 1 does not update params; step 2 does

    ptrs = {p.data_ptr() for p in params}
    writes = {}
    real_copy = torch.Tensor.copy_

    def recording_copy(self, src, non_blocking=False):
        if self.data_ptr() in ptrs:
            writes[self.data_ptr()] = non_blocking
        return real_copy(self, src, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self, *a, **k: True)
    monkeypatch.setattr(torch.Tensor, "copy_", recording_copy)
    _set_grads(params, 1)
    opt.step()

    assert set(writes) == ptrs
    assert all(writes.values()), writes


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_h2d_moves_before_casting(monkeypatch, dtype):
    # A single .to(device, dtype=...) from CPU converts on the CPU into a
    # pageable temp (host-synchronous H2D). SAEM must move, then cast.
    params = _params(dtype)
    opt = SimplifiedAdEMAMixExM(params, lr=1e-3, torch_compile=False)
    combined = []
    real_to = torch.Tensor.to

    def recording_to(self, *args, **kwargs):
        has_device = any(isinstance(a, (torch.device, str)) for a in args) or "device" in kwargs
        has_dtype = any(isinstance(a, torch.dtype) for a in args) or "dtype" in kwargs
        if has_device and has_dtype:
            combined.append((tuple(self.shape), args, kwargs))
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)
    for step in range(2):
        _set_grads(params, step)
        opt.step()
    assert combined == []



def test_zero_weight_decay_skips_kernel(monkeypatch):
    from training_extras.vendor.LoraEasyCustomOptimizer.utils import apply_weight_decay

    calls = []
    real_mul = torch.Tensor.mul_
    monkeypatch.setattr(torch.Tensor, "mul_", lambda self, *a, **k: calls.append(a) or real_mul(self, *a, **k))
    p = torch.randn(4)
    before = p.clone()
    apply_weight_decay(p, None, lr=1e-3, weight_decay=0.0, weight_decouple=True, fixed_decay=False)
    assert calls == [] and torch.equal(p, before)
    apply_weight_decay(p, None, lr=1e-3, weight_decay=0.1, weight_decouple=True, fixed_decay=False)
    assert len(calls) == 1 and not torch.equal(p, before)


def test_compiled_weight_decay_scalars_built_once_per_step(monkeypatch):
    from training_extras.vendor.LoraEasyCustomOptimizer import utils

    seen = []

    def fake_compiled(p, grad, lr_t, wd_t, *rest):
        seen.append((lr_t, wd_t))
        return utils._apply_weight_decay_compile(p, grad, lr_t, wd_t, *rest)

    monkeypatch.setattr(utils, "_get_compiled_apply_weight_decay", lambda: fake_compiled)

    def run(torch_compile, lrs):
        params = _params()
        opt = SimplifiedAdEMAMixExM(
            params, lr=lrs[0], weight_decay=0.05, torch_compile=torch_compile,
            use_stable_spam_clipping=False, use_newton_schulz=False,
        )
        for step, lr in enumerate(lrs):
            opt.param_groups[0]["lr"] = lr
            _set_grads(params, step)
            opt.step()
        return params

    lrs = [1e-3, 1e-3, 5e-4, 2e-4]
    compiled = run(True, lrs)
    eager = run(False, lrs)
    for a, b in zip(compiled, eager):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    # Step 1 skips the update; steps 2-4 each decay every param.
    assert len(seen) == 3 * len(SHAPES)
    per_step = [seen[i * len(SHAPES):(i + 1) * len(SHAPES)] for i in range(3)]
    for chunk, lr in zip(per_step, lrs[1:]):
        assert all(t[0] is chunk[0][0] and t[1] is chunk[0][1] for t in chunk)
        assert chunk[0][0].item() == pytest.approx(lr) and chunk[0][1].item() == pytest.approx(0.05)


MANY_SHAPES = [(4, 2 + i) for i in range(10)]


@pytest.fixture
def counting_compile(monkeypatch):
    """Route torch.compile through dynamo with a counting eager backend (no
    inductor/C++ toolchain needed) and start from a clean dynamo cache."""
    import torch._dynamo
    from torch._dynamo.testing import CompileCounter
    from training_extras.vendor.LoraEasyCustomOptimizer import utils

    counter = CompileCounter()
    real_compile = torch.compile

    def compile_with_counter(fn=None, **kwargs):
        kwargs["backend"] = counter
        return real_compile(fn, **kwargs)

    torch._dynamo.reset()
    monkeypatch.setattr(torch, "compile", compile_with_counter)
    monkeypatch.setattr(utils, "_compiled_fns", {})
    # torch's own config.patch, not monkeypatch.setattr: pytest 9.2 undoes an instance attribute it did not find in
    # __dict__ by deleting it, and torch's config module then hides the setting for the rest of the process
    with torch._dynamo.config.patch(recompile_limit=8):
        yield counter
    torch._dynamo.reset()


def _spam_run(params, torch_compile, steps, opt=None, start=0):
    torch.manual_seed(1234 + start)  # stochastic rounding of bf16 moments draws RNG
    if opt is None:
        opt = SimplifiedAdEMAMixExM(
            params, lr=1e-3, torch_compile=torch_compile, use_newton_schulz=False,
        )
    for step in range(start, start + steps):
        _set_grads(params, step)
        opt.step()
    return opt


def test_compiled_stable_spam_compiles_once_per_shape(counting_compile):
    params = _params(shapes=MANY_SHAPES)
    _spam_run(params, torch_compile=True, steps=12)
    assert counting_compile.frame_count == len(MANY_SHAPES)


def test_compiled_stable_spam_clip_matches_eager(counting_compile):
    from training_extras.vendor.LoraEasyCustomOptimizer.utils import (
        _stable_spam_clipping_impl, stable_spam_bias_terms, stable_spam_clipping_compiled,
    )

    eager_state, compiled_state = {}, {}
    for step in range(1, 13):
        g = torch.Generator().manual_seed(step)
        grad = torch.randn(6, 5, generator=g) * (1 + step)
        out_e = _stable_spam_clipping_impl(eager_state, grad.clone(), step=step, eps=1e-12)
        out_c = stable_spam_clipping_compiled(compiled_state, grad.clone(), step, eps=1e-12)
        torch.testing.assert_close(out_c, out_e, rtol=1e-6, atol=0)
    assert counting_compile.frame_count == 1


def test_compiled_stable_spam_clip_matches_eager_at_scale_2(counting_compile):
    # DeepSeek review B1: the compiled path took caller-built bias terms at the default scale=1,
    # so any scale != 1 debiased m_norm with 1 - 0.85**step instead of 1 - (0.85*scale)**step.
    from training_extras.vendor.LoraEasyCustomOptimizer.utils import (
        _stable_spam_clipping_impl, stable_spam_clipping_compiled,
    )

    eager_state, compiled_state, cache = {}, {}, {}
    for step in range(1, 9):
        g = torch.Generator().manual_seed(step)
        grad = torch.randn(6, 5, generator=g) * (1 + step)
        out_e = _stable_spam_clipping_impl(eager_state, grad.clone(), step=step, scale=1.1, eps=1e-12)
        out_c = stable_spam_clipping_compiled(compiled_state, grad.clone(), step, scale=1.1, eps=1e-12,
                                              bias_cache=cache)
        torch.testing.assert_close(out_c, out_e, rtol=1e-5, atol=0)
    assert len(cache) == 8  # one entry per (step, device, scale, gammas)


def test_compiled_stable_spam_matches_eager(counting_compile):
    compiled = _params(shapes=MANY_SHAPES)
    eager = _params(shapes=MANY_SHAPES)
    opt_c = _spam_run(compiled, torch_compile=True, steps=12)
    opt_e = _spam_run(eager, torch_compile=False, steps=12)
    for a, b in zip(compiled, eager):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for a, b in zip(compiled, eager):
        for key in ("ssc_m_max_t", "ssc_m_norm_t", "ssc_v_norm_t"):
            sa, sb = opt_c.state[a][key], opt_e.state[b][key]
            assert sa.dtype == torch.float32 and sa.ndim == 0
            torch.testing.assert_close(sa, sb, rtol=1e-5, atol=0)


def test_stable_spam_bias_terms_use_double_precision():
    from training_extras.vendor.LoraEasyCustomOptimizer.utils import stable_spam_bias_terms

    for step in (1, 10, 1200):
        bc3, bc1, bc2 = stable_spam_bias_terms(step, torch.device("cpu"))
        assert bc2.item() == pytest.approx(1.0 - 0.99999 ** step, rel=1e-6)
        assert bc3.item() == pytest.approx(1.0 - 0.999 ** step, rel=1e-6)
        assert bc1.item() == pytest.approx(1.0 - 0.85 ** step, rel=1e-6)


def test_compiled_state_dict_round_trip(counting_compile):
    params = _params(shapes=SHAPES)
    opt = _spam_run(params, torch_compile=True, steps=1)
    saved = {k: v for k, v in opt.state_dict().items()}
    resumed_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed = SimplifiedAdEMAMixExM(
        resumed_params, lr=1e-3, torch_compile=True, use_newton_schulz=False,
    )
    resumed.load_state_dict(saved)
    _spam_run(params, torch_compile=True, steps=2, opt=opt, start=1)
    _spam_run(resumed_params, torch_compile=True, steps=2, opt=resumed, start=1)
    for a, b in zip(params, resumed_params):
        assert torch.equal(a, b)
    assert counting_compile.frame_count == len(SHAPES)
