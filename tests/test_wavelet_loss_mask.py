"""Regression tests for spatial mask application with loss_type='wavelet' (CPU)."""
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import TrainConfig


def _make_trainer(loss_type="wavelet"):
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.train_config = TrainConfig(loss_type=loss_type, dtype="fp32")
    trainer.device_torch = torch.device("cpu")
    trainer.dfe = None
    trainer.additional_logs = {}
    trainer.network = None
    trainer.adapter = None
    trainer.training_extras = None
    trainer.sd = SimpleNamespace(
        is_flow_matching=True,
        prediction_type="flow",
        x0_pred=False,
        scale_loss=lambda loss: loss,
        noise_scheduler=SimpleNamespace(),
    )
    return trainer


def _make_batch(latents):
    return SimpleNamespace(
        latents=latents,
        mask_tensor=None,
        loss_multiplier_list=[1.0] * latents.shape[0],
        get_is_reg_list=lambda: [False] * latents.shape[0],
        sigmas=None,
    )


def test_wavelet_loss_applies_spatial_mask_without_dropping(capsys):
    pytest.importorskip("pytorch_wavelets")
    trainer = _make_trainer("wavelet")
    latents = torch.zeros(1, 4, 8, 8)
    noise = torch.zeros(1, 4, 8, 8)
    noisy_latents = torch.zeros(1, 4, 8, 8)
    timesteps = torch.tensor([500.0])
    batch = _make_batch(latents)

    # Error strictly in right half (cols 4..7); mask zeros right half and keeps left half.
    noise_pred = torch.zeros(1, 4, 8, 8)
    noise_pred[:, :, :, 4:] = 2.0

    # Full-channel mask (B, C, H, W) matching what SDTrainer passes from latent masks.
    mask_keep_left = torch.zeros(1, 4, 8, 8)
    mask_keep_left[:, :, :, :4] = 1.0

    masked_loss = trainer.calculate_loss(
        noise_pred=noise_pred,
        noise=noise,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        batch=batch,
        mask_multiplier=mask_keep_left,
    )
    out = capsys.readouterr().out
    assert "Could not apply mask multiplier" not in out
    assert masked_loss.item() == pytest.approx(0.0, abs=1e-6)

    # Flipping the mask to keep the right half (where the error lives) gives non-zero loss.
    mask_keep_right = 1.0 - mask_keep_left
    right_loss = trainer.calculate_loss(
        noise_pred=noise_pred,
        noise=noise,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        batch=batch,
        mask_multiplier=mask_keep_right,
    )
    assert right_loss.item() > 0.1

    # Single-channel mask (B, 1, H, W) and scalar float mask (1.0) also work cleanly.
    single_ch_loss = trainer.calculate_loss(
        noise_pred=noise_pred,
        noise=noise,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        batch=batch,
        mask_multiplier=mask_keep_left[:, :1, :, :],
    )
    assert single_ch_loss.item() == pytest.approx(0.0, abs=1e-6)

    scalar_loss = trainer.calculate_loss(
        noise_pred=noise_pred,
        noise=noise,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        batch=batch,
        mask_multiplier=1.0,
    )
    out2 = capsys.readouterr().out
    assert "Could not apply mask multiplier" not in out2
    assert scalar_loss.item() == pytest.approx(right_loss.item(), rel=1e-5)
