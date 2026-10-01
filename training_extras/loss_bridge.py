"""Compose the vendored pure-torch losses behind one bridge module.

This file owns no backprop, optimiser or checkpoint state -- it only wires the
already-vendored sources under ``.vendor.sd_scripts`` into the training loop.
"""
from __future__ import annotations

import torch
import math
from types import SimpleNamespace

def _namespace(value):
    return SimpleNamespace(**{k: _namespace(v) for k,v in value.items()}) if isinstance(value, dict) else value

from .vendor.sd_scripts import (
    anchor_loss,
    base_losses,
    focal_frequency_loss,
    hf_token_loss,
    patch_topology_loss,
)

_SCALAR_LOSSES = (
    "standard_deviation_loss",
    "psnr_loss",
    "geman_mcclure_loss",
)
_ALIASES = {"mse": "l2", "mae": "l1"}
_VALID = (
    "l2", "l1", "standard_pseudo_huber", "standard_huber",
    "standard_smooth_l1", "huber", "smooth_l1", "x_sigmoid", "log_cosh",
    "squared_logarithmic", "soft_welsch", "scaled_quadratic",
    "standard_deviation_loss", "psnr_loss", "geman_mcclure_loss",
    "smooth_l2_log", "mse_pyramid_2d",
)


def _scale(section):
    value = float(getattr(section, "scale", 0.0)) if section is not None else 0.0
    if value < 0:
        raise ValueError("aux loss scales must be non-negative")
    return value


def _field(section, name, default):
    return getattr(section, name, default) if section is not None else default


class LossBridge(torch.nn.Module):
    """Adapter around the vendored losses; never owns optimiser state."""

    def __init__(self, cfg, scheduler, is_flow, prediction_type="epsilon"):
        super().__init__()
        cfg = _namespace(cfg)
        self.scheduler = scheduler
        self.is_flow = bool(is_flow)
        self.prediction_type = prediction_type

        name = getattr(cfg, "loss", "l2") or "l2"
        name = _ALIASES.get(name, name)
        if name not in _VALID:
            raise ValueError(f"unknown loss {name!r}")
        self.loss_type = name

        delta = float(getattr(cfg, "delta", 0.01))
        if delta <= 0:
            raise ValueError("cfg.delta must be positive")
        self.delta = delta
        self.delta_schedule = getattr(cfg, 'delta_schedule', 'constant')
        if self.delta_schedule not in ('constant','exponential','snr'):
            raise ValueError('delta_schedule must be constant, exponential or snr')
        if self.delta_schedule == 'snr' and self.is_flow:
            raise ValueError('The source SNR Huber schedule requires a DDPM scheduler')
        self.scale = float(getattr(cfg, "scale", 1.0))
        self.loss_float64 = bool(getattr(cfg, "loss_float64", False))
        self.x0_direct = bool(getattr(cfg, "x0_direct", False))

        acfg = getattr(cfg, "anchor", None)
        hcfg = getattr(cfg, "hf", None)
        fcfg = getattr(cfg, "frequency", None)
        tcfg = getattr(cfg, "topology", None)

        self.anchor_scale = _scale(acfg)
        self.hf_scale = _scale(hcfg)
        self.frequency_scale = _scale(fcfg)
        self.topology_scale = _scale(tcfg)

        self.anchor_levels = int(_field(acfg, "levels", 3))
        if self.anchor_levels <= 0:
            raise ValueError("cfg.anchor.levels must be positive")
        self.anchor_snr_weight = bool(_field(acfg, "snr_weight", False))

        self.hf_patch = int(_field(hcfg, "patch", 4))
        if self.hf_patch <= 0:
            raise ValueError("cfg.hf.patch must be positive")
        self.hf_exponent = float(_field(hcfg, "exponent", 1.0))

        self.frequency = None
        if self.frequency_scale > 0:
            self.frequency = focal_frequency_loss.FocalFrequencyLoss(
                loss_weight=1.0,
                alpha=float(_field(fcfg, "alpha", 1.0)),
                patch_factor=int(_field(fcfg, "patch_factor", 1)),
                ave_spectrum=bool(_field(fcfg, "ave_spectrum", False)),
                log_matrix=bool(_field(fcfg, "log_matrix", False)),
                batch_matrix=bool(_field(fcfg, "batch_matrix", False)),
            )

        self.topology = None
        if self.topology_scale > 0:
            self.topology = patch_topology_loss.PatchTopologyLoss(
                loss_weight=1.0,
                tau_latent=float(_field(tcfg, "tau_latent", 0.1)),
                tau_target=float(_field(tcfg, "tau_target", 0.1)),
                scale_levels=int(_field(tcfg, "scale_levels", 2)),
                loss_type=_field(tcfg, "loss_type", "kl"),
                apply_channel_norm=bool(_field(tcfg, "apply_channel_norm", True)),
                apply_timestep_weight=False,
                chunk_size=int(_field(tcfg, "chunk_size", 512)),
            )

    @property
    def x0_mode(self):
        if self.x0_direct:
            return "x0_direct"
        if self.is_flow:
            return "flow"
        if self.prediction_type == "v_prediction":
            return "vpred_ddpm"
        return "eps_ddpm"

    def element_loss(self, pred, target, timesteps=None):
        """Unreduced (per-element / per-sample) baseline loss on pred vs target."""
        out_dtype = torch.float64 if self.loss_float64 else torch.float32
        if self.loss_type == 'squared_logarithmic' and (bool((pred <= -1).any()) or bool((target <= -1).any())):
            raise ValueError('squared_logarithmic requires predictions and targets greater than -1; it is unsuitable for signed diffusion velocities')
        delta = torch.as_tensor(self.delta, device=pred.device, dtype=torch.float32)
        if self.delta_schedule != 'constant':
            if timesteps is None: raise ValueError('Scheduled Huber thresholds require timesteps')
            ts = timesteps.to(pred.device)
            if self.delta_schedule == 'exponential':
                delta = torch.exp(math.log(self.delta) * ts / self.scheduler.config.num_train_timesteps)
            else:
                alpha = self.scheduler.alphas_cumprod.to(pred.device)[ts.long()]
                sigma = ((1-alpha)/alpha).sqrt()
                delta = (1-self.delta)/(1+sigma).square()+self.delta

        if self.loss_type == "mse_pyramid_2d" and pred.dim() == 5:
            return self._video_element_loss(pred, target, delta, out_dtype)

        if self.loss_type in _SCALAR_LOSSES:
            pieces = []
            for i in range(pred.shape[0]):
                value = base_losses.conditional_loss(
                    pred[i:i + 1], target[i:i + 1], self.loss_type, "none",
                    huber_c=delta, scale=self.scale,
                )
                pieces.append(torch.ones_like(pred[i:i + 1], dtype=out_dtype) * value.to(out_dtype))
            return torch.cat(pieces, dim=0)

        loss = base_losses.conditional_loss(
            pred, target, self.loss_type, "none", huber_c=delta, scale=self.scale,
        )
        return loss.to(out_dtype)

    def _video_element_loss(self, pred, target, delta, out_dtype):
        flat = []
        for tensor in (pred, target):
            b, c, t = tensor.shape[0], tensor.shape[1], tensor.shape[2]
            flat.append(tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, *tensor.shape[3:]))
        loss = base_losses.conditional_loss(
            flat[0], flat[1], "mse_pyramid_2d", "none", huber_c=delta, scale=self.scale,
        )
        b, c, t = pred.shape[0], pred.shape[1], pred.shape[2]
        return loss.reshape(b, t, c, *pred.shape[3:]).permute(0, 2, 1, 3, 4).to(out_dtype)

    def _aux_enabled(self):
        return max(self.anchor_scale, self.hf_scale,
                   self.frequency_scale, self.topology_scale) > 0

    def augment(self, per_sample, pred, clean, noisy, timesteps):
        """Add scaled per-sample aux terms to ``per_sample`` and return logs."""
        logs = {}
        if not self._aux_enabled():
            return per_sample, logs

        if pred.dim() == 5:
            b, c, t = pred.shape[0], pred.shape[1], pred.shape[2]
            tail = tuple(pred.shape[3:])
            flat = lambda x: x.permute(0, 2, 1, 3, 4).reshape(b * t, c, *tail)
            p, cln, nse = flat(pred), flat(clean), flat(noisy)
            if timesteps.dim() == 1 and timesteps.shape[0] == b:
                ts = timesteps.repeat_interleave(t)
            else:
                ts = timesteps.reshape(b * t)
        elif pred.dim() == 4:
            b, t = pred.shape[0], 1
            p, cln, nse = pred, clean, noisy
            ts = timesteps
        else:
            raise ValueError("unsupported shape for aux losses")

        p = p.float()
        cln = cln.detach().float()
        nse = nse.detach().float()
        if ts.dim() == 1 and ts.shape[0] == b:
            ts = ts.repeat_interleave(t)
        ts = ts.reshape(-1)

        x0 = hf_token_loss.hf_x0_hat(
            p, nse, ts, self.x0_mode, noise_scheduler=self.scheduler,
            timesteps_in_sigma=False, eps_train=1e-3,
        ).to(torch.float32)
        target = cln

        if self.anchor_scale > 0:
            per_frame = anchor_loss.anchor_per_sample_loss(
                x0, target, self.anchor_levels).to(torch.float32)
            if self.anchor_snr_weight:
                w = anchor_loss.anchor_snr_weights(
                    ts, self.x0_mode, self.scheduler, False)
                per_frame = per_frame * w.to(per_frame.dtype)
            value = self._reduce(per_frame, b, t)
            per_sample = per_sample + self.anchor_scale * value
            logs["loss/anchor"] = value.detach().float().mean()

        if self.hf_scale > 0:
            per_frame = hf_token_loss.hf_per_sample_loss(
                x0, target, self.hf_patch, self.hf_exponent).to(torch.float32)
            value = self._reduce(per_frame, b, t)
            per_sample = per_sample + self.hf_scale * value
            logs["loss/hf"] = value.detach().float().mean()

        if self.frequency_scale > 0:
            frames = [
                self.frequency(x0[i:i + 1], target[i:i + 1]).reshape(())
                for i in range(x0.shape[0])
            ]
            per_frame = torch.stack(frames).to(torch.float32)
            value = self._reduce(per_frame, b, t)
            per_sample = per_sample + self.frequency_scale * value
            logs["loss/frequency"] = value.detach().float().mean()

        if self.topology_scale > 0:
            per_frame = self.topology(
                x0, cln, timesteps=ts, mask=None).to(torch.float32)
            value = self._reduce(per_frame, b, t)
            per_sample = per_sample + self.topology_scale * value
            logs["loss/topology"] = value.detach().float().mean()

        return per_sample, logs

    @staticmethod
    def _reduce(per_frame, b, t):
        if t == 1:
            return per_frame
        return per_frame.reshape(b, t).mean(dim=1)
