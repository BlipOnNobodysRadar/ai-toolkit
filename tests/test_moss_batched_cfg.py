"""models-4: MOSS preview CFG as one B=2 forward (uncond skips its blocks via
skip_layers_items) must match the two B=1 forwards. CPU, fp32, tiny DiT."""
import types

import pytest
import torch

from extensions_built_in.diffusion_models.moss_image.src.pipeline import (
    MossImagePipeline,
    stack_text_embeds,
)
from extensions_built_in.diffusion_models.moss_image.src.transformer import (
    MossImageParams,
    MossImageTransformer,
)

SKIP = 3


def _model():
    torch.manual_seed(0)
    p = MossImageParams(dim=64, ffn_dim=128, num_heads=4, num_layers=6, text_dim=32, freq_dim=16, latent_channels=8)
    m = MossImageTransformer(p).float().eval()
    with torch.no_grad():
        for prm in m.parameters():
            prm.normal_(0, 0.05)
    return m


def _one(m, x, t, ctx, skip):
    return m(x, t, ctx[None], torch.ones(1, ctx.shape[0], dtype=torch.bool), skip_layers=skip)


@pytest.mark.parametrize("cond_len,uncond_len", [(11, 4), (5, 5), (3, 9)])
@pytest.mark.parametrize("skip", [[SKIP], []])
@torch.no_grad()
def test_batched_forward_matches_two_calls(cond_len, uncond_len, skip):
    m = _model()
    x, t = torch.randn(1, 8, 6, 6), torch.tensor([700.0])
    cond, uncond = torch.randn(cond_len, 32), torch.randn(uncond_len, 32)
    a = _one(m, x, t, cond, None)
    b = _one(m, x, t, uncond, skip)
    ctx, mask = stack_text_embeds([cond, uncond], torch.device("cpu"), torch.float32)
    out = m(x.expand(2, -1, -1, -1), t, ctx, mask, skip_layers=skip, skip_layers_items=[1])
    torch.testing.assert_close(out[0], a[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out[1], b[0], atol=1e-6, rtol=1e-5)
    if skip:  # the skip is really exercised
        assert (_one(m, x, t, uncond, None) - b).abs().max() > 1e-5


def test_skip_layers_items_guards():
    m = _model()
    x, t = torch.randn(1, 8, 4, 4), torch.tensor([500.0])
    ctx = torch.randn(2, 5, 32)
    with pytest.raises(ValueError, match="inference only"):
        m(x.expand(2, -1, -1, -1), t, ctx, None, skip_layers=[SKIP], skip_layers_items=[1])
    with torch.no_grad(), pytest.raises(ValueError, match="references"):
        m(x.expand(2, -1, -1, -1), t, ctx, None, skip_layers=[SKIP], skip_layers_items=[1],
          ref_latents=[[torch.randn(8, 2, 2)], [torch.randn(8, 2, 2)]])


class _StubModel:
    device_torch = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self, transformer):
        self.transformer = transformer
        self.decoded = None
        self.calls = 0
        orig = transformer.forward

        def counted(*a, **k):
            self.calls += 1
            return orig(*a, **k)

        transformer.forward = counted

    def decode_latents(self, latents, device=None, dtype=None):
        self.decoded = latents.clone()
        return torch.zeros(1, 3, 2, 2)


@pytest.mark.parametrize("cond_len,uncond_len", [(11, 4), (6, 6)])
@pytest.mark.parametrize("sigmas", [None, [1.0, 0.7, 0.2]])
def test_pipeline_batched_cfg_matches_two_call_path(cond_len, uncond_len, sigmas):
    m = _model()
    torch.manual_seed(1)
    cond = types.SimpleNamespace(text_embeds=[torch.randn(cond_len, 32)])
    uncond = types.SimpleNamespace(text_embeds=[torch.randn(uncond_len, 32)])
    latents = torch.randn(1, 8, 6, 6)
    results = {}
    for batched in (False, True):
        model = _StubModel(m)
        MossImagePipeline(model)(cond, uncond, height=96, width=96, num_inference_steps=4,
                                 latents=latents, uncond_skip_layers=(SKIP,), batched_cfg=batched,
                                 sigmas=sigmas)
        results[batched] = (model.decoded, model.calls)
        del m.forward
    (two, two_calls), (one, one_calls) = results[False], results[True]
    assert one_calls * 2 == two_calls
    assert one_calls == (4 if sigmas is None else len(sigmas))
    torch.testing.assert_close(one, two, atol=1e-5, rtol=1e-5)


def test_batched_cfg_is_off_by_default():
    # DeepSeek review C3: previews keep the two-call path (and its memory) unless model_kwargs.batched_cfg opts in
    import inspect
    from extensions_built_in.diffusion_models.moss_image import moss_image
    assert inspect.signature(MossImagePipeline.__call__).parameters["batched_cfg"].default is False
    assert 'kw.get("batched_cfg", False)' in inspect.getsource(moss_image.MossImageModel.__init__)
    m = _model()
    cond = types.SimpleNamespace(text_embeds=[torch.randn(5, 32)])
    uncond = types.SimpleNamespace(text_embeds=[torch.randn(3, 32)])
    model = _StubModel(m)
    MossImagePipeline(model)(cond, uncond, height=96, width=96, num_inference_steps=3,
                             latents=torch.randn(1, 8, 6, 6), uncond_skip_layers=(SKIP,))
    assert model.calls == 6  # cond + uncond per step
