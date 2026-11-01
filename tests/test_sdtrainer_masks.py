"""CPU checks for the masked-loss and inverted-mask-prior multipliers built in SDTrainer."""
import pytest
import torch

from extensions_built_in.sd_trainer.SDTrainer import latent_mask_multiplier, inverted_prior_mask_multiplier


def _stripe_mask(h=64, w=96, c0=20, c1=28, low=0.3):
    # subject stripe 8 px wide starting mid-cell (cells are 16 px at latent 4x6)
    m = torch.full((2, 1, h, w), low)
    m[:, :, :, c0:c1] = 1.0
    return m


def _legacy(mask, latent_shape, dtype=torch.bfloat16):
    # the pre-fix code path: fp16 nearest resize, cast to train dtype, normalise in that dtype
    m = torch.nn.functional.interpolate(mask.to(torch.float16), size=latent_shape[-2:])
    m = m.expand(-1, latent_shape[1], -1, -1).to(dtype)
    return m / m.mean()


def test_nearest_matches_legacy_within_bf16_rounding():
    mask = _stripe_mask()
    out = latent_mask_multiplier(mask, (2, 16, 4, 6), 'nearest')
    assert out.dtype == torch.float32 and out.shape == (2, 16, 4, 6)
    assert torch.allclose(out, _legacy(mask, (2, 16, 4, 6)).float(), rtol=1e-2)
    assert out.mean().item() == pytest.approx(1.0, abs=1e-6)


def test_area_averages_each_cell():
    mask = _stripe_mask()  # cols 20-27: half of cell 1 (16-31) is subject
    near = latent_mask_multiplier(mask, (2, 4, 4, 6), 'nearest')
    area = latent_mask_multiplier(mask, (2, 4, 4, 6), 'area')
    raw_near = torch.nn.functional.interpolate(mask, size=(4, 6))
    raw_area = torch.nn.functional.interpolate(mask, size=(4, 6), mode='area')
    # nearest reads pixel 16 (background) and drops the stripe entirely; area keeps half its weight
    assert raw_near[0, 0, 0, 1].item() == pytest.approx(0.3)
    assert raw_area[0, 0, 0, 1].item() == pytest.approx(0.65, abs=1e-5)
    assert torch.allclose(area, (raw_area / raw_area.mean()).expand(-1, 4, -1, -1))
    assert not torch.allclose(near, area)


def test_video_latents_use_last_two_dims():
    mask = _stripe_mask()
    out = latent_mask_multiplier(mask, (2, 8, 3, 4, 6), 'area')
    assert out.shape == (2, 8, 4, 6)


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        latent_mask_multiplier(_stripe_mask(), (2, 4, 4, 6), 'bicubic')


def test_config_default_is_nearest():
    from toolkit.config_modules import TrainConfig
    assert TrainConfig().mask_resize_mode == 'nearest'
    assert TrainConfig(mask_resize_mode='area').mask_resize_mode == 'area'


def _legacy_prior(mask, latent_shape, pred_shape):
    m = torch.nn.functional.interpolate(mask, size=latent_shape[-2:], mode='bicubic')
    m = torch.cat([m[:1]] * pred_shape[1], dim=1)
    if len(pred_shape) == 5:
        m = m.unsqueeze(2).repeat(1, 1, pred_shape[2], 1, 1)
    m = 1.0 - m
    return m / m.mean()


def _distinct_masks(b=3, h=32, w=32):
    m = torch.zeros(b, 1, h, w)
    for i in range(b):
        m[i, :, :, i * 8:(i + 1) * 8 + 4] = 1.0  # a different subject region per sample
    return m


def test_inverted_prior_mask_is_per_sample():
    mask = _distinct_masks()
    out = inverted_prior_mask_multiplier(mask, (3, 16, 8, 8), (3, 16, 8, 8))
    assert out.shape == (3, 16, 8, 8)
    resized = torch.nn.functional.interpolate(mask, size=(8, 8), mode='bicubic')
    expected = (1.0 - resized) / (1.0 - resized).mean()
    for i in range(3):
        assert torch.allclose(out[i], expected[i].expand(16, -1, -1))
    assert not torch.allclose(out[1], out[0])
    # the legacy code broadcast sample 0's mask to every sample
    assert _legacy_prior(mask, (3, 16, 8, 8), (3, 16, 8, 8)).shape[0] == 1


def test_inverted_prior_mask_batch_one_unchanged_and_video():
    mask = _distinct_masks(b=1)
    assert torch.allclose(inverted_prior_mask_multiplier(mask, (1, 16, 8, 8), (1, 16, 8, 8)),
                          _legacy_prior(mask, (1, 16, 8, 8), (1, 16, 8, 8)))
    mask = _distinct_masks()
    out = inverted_prior_mask_multiplier(mask, (3, 16, 4, 8, 8), (3, 16, 4, 8, 8))
    assert out.shape == (3, 16, 4, 8, 8)
    assert torch.equal(out[:, :, 0], out[:, :, 3]) and not torch.allclose(out[1], out[0])
