"""Unit tests for Ideogram 4 guidance-aware training helpers.

Run with:   python testing/test_ideogram4_guidance_aware.py
or:         pytest testing/test_ideogram4_guidance_aware.py

Only torch is required. The heavy pipeline imports (transformers / diffusers /
PIL) are stubbed when missing, and the ideogram4 ``src`` modules are loaded by
file path so the package ``__init__`` (which pulls in the whole toolkit) is
never executed. The tests instantiate *tiny* Ideogram4 transformers on CPU and
verify, against a by-hand reconstruction of the official negative branch, that:

  1. ``predict_velocity_unconditional`` packs the image-only / zeroed-text
     sequence exactly like the reference pipeline's asymmetric pass.
  2. ``combine_guided_velocity`` implements u + g (c - u), per-sample g works,
     and ``grad_rescale`` leaves the forward value unchanged while dividing the
     gradient by g.
  3. ``guidance_aware_prediction`` routes gradients to the conditional model
     only, and reduces exactly to the conditional prediction at g = 1.
"""

import importlib.util
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(
    ROOT, "extensions_built_in", "diffusion_models", "ideogram4", "src"
)


# ---------------------------------------------------------------------------
# Import machinery: stub heavy deps, load src modules by path.
# ---------------------------------------------------------------------------


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _ensure_stubs():
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        pil = _stub_module("PIL")
        pil.Image = _stub_module("PIL.Image")

    try:
        from diffusers.utils.torch_utils import randn_tensor  # noqa: F401
    except ImportError:

        def randn_tensor(shape, generator=None, device=None, dtype=None):
            return torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )

        d = _stub_module("diffusers")
        du = _stub_module("diffusers.utils")
        dut = _stub_module("diffusers.utils.torch_utils", randn_tensor=randn_tensor)
        d.utils = du
        du.torch_utils = dut

    try:
        from transformers.masking_utils import create_causal_mask  # noqa: F401
    except ImportError:
        t = _stub_module("transformers")
        tm = _stub_module(
            "transformers.masking_utils",
            create_causal_mask=lambda **kwargs: None,
        )
        t.masking_utils = tm


def _load_src():
    _ensure_stubs()
    pkg = types.ModuleType("ideogram4_src_for_test")
    pkg.__path__ = [SRC]
    sys.modules[pkg.__name__] = pkg
    loaded = {}
    for name in ("transformer", "pipeline"):
        spec = importlib.util.spec_from_file_location(
            f"{pkg.__name__}.{name}", os.path.join(SRC, f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        loaded[name] = mod
    return loaded["transformer"], loaded["pipeline"]


T, P = _load_src()


# ---------------------------------------------------------------------------
# Tiny model + inputs.
# ---------------------------------------------------------------------------

IN_CH = 16
LLM_DIM = 48


def tiny_transformer(seed: int) -> "T.Ideogram4Transformer2DModel":
    torch.manual_seed(seed)
    cfg = T.Ideogram4Config(
        emb_dim=64,
        num_layers=2,
        num_heads=2,
        intermediate_size=128,
        adanln_dim=32,
        in_channels=IN_CH,
        llm_features_dim=LLM_DIM,
        rope_theta=10_000,
        mrope_section=(6, 5, 5),  # axis sections * 3 must fit in head_dim/2=16
    )
    model = T.Ideogram4Transformer2DModel(cfg)
    model.eval()
    return model


def make_inputs(b=2, gh=2, gw=3, lt=5, seed=0):
    torch.manual_seed(seed)
    latents = torch.randn(b, IN_CH, gh, gw)
    t = torch.rand(b)
    feats = torch.randn(b, lt, LLM_DIM)
    mask = torch.ones(b, lt, dtype=torch.long)
    # make the batch ragged: second sample has 2 padding positions
    mask[1, -2:] = 0
    feats[1, -2:] = 0
    return latents, t, feats, mask


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_unconditional_matches_reference_packing():
    """Helper output == a by-hand build of the official asymmetric neg branch."""
    tr = tiny_transformer(1)
    latents, t, _, _ = make_inputs(seed=11)

    with torch.no_grad():
        out_helper = P.predict_velocity_unconditional(tr, latents, t)

        # Reference construction, mirroring the official pipeline's negative
        # pass: image tokens only, zeroed llm features over image positions,
        # OUTPUT_IMAGE indicators, a single segment, image-only MRoPE positions
        # (+offset), model time = 1 - t, output negated back to toolkit
        # velocity convention.
        b, c, gh, gw = latents.shape
        li = gh * gw
        x = latents.permute(0, 2, 3, 1).reshape(b, li, c)
        llm = torch.zeros(b, li, tr.config.llm_features_dim)
        indicator = torch.full((b, li), T.OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
        segment_ids = torch.ones(b, li, dtype=torch.long)
        h_idx = torch.arange(gh).view(-1, 1).expand(gh, gw).reshape(-1)
        w_idx = torch.arange(gw).view(1, -1).expand(gh, gw).reshape(-1)
        pos = (
            torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1)
            + T.IMAGE_POSITION_OFFSET
        )
        pos = pos.unsqueeze(0).expand(b, -1, -1)

        out_ref = tr(
            llm_features=llm,
            x=x,
            t=1.0 - t,
            position_ids=pos,
            segment_ids=segment_ids,
            indicator=indicator,
        )
        out_ref = -out_ref.reshape(b, gh, gw, c).permute(0, 3, 1, 2)

    assert out_helper.shape == latents.shape
    assert torch.isfinite(out_helper).all()
    assert torch.allclose(out_helper, out_ref, atol=1e-5), (
        f"max diff {(out_helper - out_ref).abs().max().item()}"
    )


def test_combine_matches_cfg_formula():
    torch.manual_seed(2)
    v_c = torch.randn(2, 4, 3, 3)
    v_u = torch.randn(2, 4, 3, 3)

    # per-sample tensor guidance, no rescale
    g = torch.tensor([1.0, 7.0])
    out = P.combine_guided_velocity(v_c, v_u, g, grad_rescale=False)
    expected = v_u + g.view(-1, 1, 1, 1) * (v_c - v_u)
    assert torch.allclose(out, expected, atol=1e-6)

    # scalar guidance, rescale on: forward value must be unchanged
    out2 = P.combine_guided_velocity(v_c, v_u, 3.5, grad_rescale=True)
    expected2 = v_u + 3.5 * (v_c - v_u)
    assert torch.allclose(out2, expected2, atol=1e-5)


def test_grad_rescale_divides_gradient_by_g():
    torch.manual_seed(3)
    g = 7.0
    target = torch.randn(2, 4, 2, 2)
    v_u = torch.randn(2, 4, 2, 2)
    v_c0 = torch.randn(2, 4, 2, 2)

    grads = {}
    for rescale in (False, True):
        v_c = v_c0.clone().requires_grad_(True)
        out = P.combine_guided_velocity(v_c, v_u, g, grad_rescale=rescale)
        loss = torch.nn.functional.mse_loss(out.float(), target)
        loss.backward()
        grads[rescale] = v_c.grad.detach().clone()

    # same forward value -> same residual; gradient differs by exactly g
    assert torch.allclose(grads[False], grads[True] * g, rtol=1e-4, atol=1e-6)
    # and the unconditional side never carries gradient
    assert not v_u.requires_grad


def test_guidance_aware_prediction_grads_and_identity():
    cond = tiny_transformer(4)
    uncond = tiny_transformer(5)  # deliberately different weights
    latents, t, feats, mask = make_inputs(seed=12)

    # g = 1 reduces exactly to the conditional prediction
    with torch.no_grad():
        out_g1 = P.guidance_aware_prediction(
            cond, uncond, latents, t, feats, mask, 1.0
        )
        out_c = P.predict_velocity(cond, latents, t, feats, mask)
    assert torch.allclose(out_g1, out_c, atol=1e-5)

    # g = 7 must differ from the plain conditional prediction
    with torch.no_grad():
        out_g7 = P.guidance_aware_prediction(
            cond, uncond, latents, t, feats, mask, 7.0
        )
    assert not torch.allclose(out_g7, out_c, atol=1e-3)

    # gradients reach the conditional model only
    cond.zero_grad(set_to_none=True)
    uncond.zero_grad(set_to_none=True)
    out = P.guidance_aware_prediction(cond, uncond, latents, t, feats, mask, 7.0)
    out.float().pow(2).mean().backward()

    cond_grad = sum(
        p.grad.abs().sum().item() for p in cond.parameters() if p.grad is not None
    )
    assert cond_grad > 0, "conditional model received no gradient"
    assert all(p.grad is None for p in uncond.parameters()), (
        "unconditional model must stay no-grad"
    )


def test_per_sample_guidance_in_guided_prediction():
    cond = tiny_transformer(6)
    uncond = tiny_transformer(7)
    latents, t, feats, mask = make_inputs(seed=13)
    g = torch.tensor([1.0, 6.0])

    with torch.no_grad():
        out = P.guidance_aware_prediction(cond, uncond, latents, t, feats, mask, g)
        v_c = P.predict_velocity(cond, latents, t, feats, mask)
        v_u = P.predict_velocity_unconditional(uncond, latents, t)
        expected = v_u + g.view(-1, 1, 1, 1) * (v_c - v_u)

    # sample 0 (g=1) is exactly the conditional prediction
    assert torch.allclose(out[0], v_c[0], atol=1e-5)
    assert torch.allclose(out, expected, atol=1e-5)


TESTS = [
    test_unconditional_matches_reference_packing,
    test_combine_matches_cfg_formula,
    test_grad_rescale_divides_gradient_by_g,
    test_guidance_aware_prediction_grads_and_identity,
    test_per_sample_guidance_in_guided_prediction,
]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {e!r}")
    if failed:
        sys.exit(1)
    print(f"\nAll {len(TESTS)} tests passed.")
