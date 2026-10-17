"""Element-wise loss weighting: token mining + wavelet masking.

Canonical discrete timestep convention: t = sigma * 1000, always applied
regardless of whether the scheduler is flow-matching. Units are never
inferred from the max of the timestep tensor.

Vendored helpers live in ``.vendor.sd_scripts.element_weighting``.
"""

from __future__ import annotations

import math

import torch

from .vendor.sd_scripts.element_weighting import (
    apply_token_mining,
    compute_wavelet_attention_map,
    get_wavelet_mask,
    setup_wavelet_dwt,
)

_TOKEN_KEYS = {
    "enabled",
    "alpha",
    "min_weight",
    "max_weight",
    "sigma_gate",
}
_WAVELET_KEYS = {"enabled", "minimum_fraction"}
_SECTIONS = {"token_mining", "wavelet"}


def _fs(v, name):
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise TypeError(f"{name} must be a real number, got {type(v).__name__}")
    if not math.isfinite(float(v)):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return float(v)


def _bool(v, name):
    if not isinstance(v, bool):
        raise TypeError(f"{name} must be a bool, got {type(v).__name__}")
    return v


def _reject_unknown(cfg: dict, allowed: set, where: str) -> None:
    if cfg is None:
        return
    if not isinstance(cfg, dict):
        raise TypeError(f"{where} must be a mapping, got {type(cfg).__name__}")
    extra = sorted(set(cfg) - allowed)
    if extra:
        raise ValueError(f"unknown key(s) in {where}: {extra}")


class ElementWeighting:
    """Apply token mining and/or wavelet masking to a per-element loss."""

    def __init__(self, cfg: dict, scheduler, is_flow: bool):
        cfg = {} if cfg is None else cfg
        _reject_unknown(cfg, _SECTIONS, "element_weighting config")

        tm = cfg.get("token_mining", {}) or {}
        _reject_unknown(tm, _TOKEN_KEYS, "token_mining")
        self.tm_enabled = _bool(tm.get("enabled", False), "token_mining.enabled")
        self.alpha = _fs(tm.get("alpha", 1.0), "token_mining.alpha")
        self.min_weight = _fs(tm.get("min_weight", 0.25), "token_mining.min_weight")
        self.max_weight = _fs(tm.get("max_weight", 4.0), "token_mining.max_weight")
        self.sigma_gate = _bool(
            tm.get("sigma_gate", True), "token_mining.sigma_gate"
        )
        if self.alpha < 0.0:
            raise ValueError(f"token_mining.alpha must be >= 0, got {self.alpha!r}")
        if not (0.0 < self.min_weight <= self.max_weight):
            raise ValueError(
                "require 0 < token_mining.min_weight <= token_mining.max_weight, "
                f"got {self.min_weight!r} / {self.max_weight!r}"
            )

        wv = cfg.get("wavelet", {}) or {}
        _reject_unknown(wv, _WAVELET_KEYS, "wavelet")
        self.wv_enabled = _bool(wv.get("enabled", False), "wavelet.enabled")
        self.minimum_fraction = _fs(
            wv.get("minimum_fraction", 0.3), "wavelet.minimum_fraction"
        )
        if not (0.0 <= self.minimum_fraction <= 1.0):
            raise ValueError(
                "require 0 <= wavelet.minimum_fraction <= 1, "
                f"got {self.minimum_fraction!r}"
            )

        self.scheduler = scheduler
        self.is_flow = bool(is_flow)
        self._dwt = None
        self._dwt_device = None

    def _get_dwt(self, device):
        if self._dwt is None or self._dwt_device != device:
            self._dwt = setup_wavelet_dwt(device)
            self._dwt_device = device
        return self._dwt

    def _sigma_from_timesteps(self, timesteps, like):
        t = timesteps.to(dtype=torch.float32)
        return t / float(self.scheduler.config.num_train_timesteps)

    def apply(self, loss: torch.Tensor, noisy: torch.Tensor, timesteps):
        if not (self.tm_enabled or self.wv_enabled): return loss, {}
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss must be a torch.Tensor")
        if not isinstance(noisy, torch.Tensor):
            raise TypeError("noisy must be a torch.Tensor")
        if loss.dim() not in (4, 5):
            raise ValueError(f"loss must be BCHW or BCTHW, got dim={loss.dim()}")
        if noisy.shape[:1] != loss.shape[:1]:
            raise ValueError(
                f"batch mismatch: loss {tuple(loss.shape)} vs noisy {tuple(noisy.shape)}"
            )

        is_video = loss.dim() == 5
        batch = loss.shape[0]

        if isinstance(timesteps, torch.Tensor):
            ts = timesteps.to(device=loss.device, dtype=torch.float32)
        else:
            ts = torch.as_tensor(timesteps, device=loss.device, dtype=torch.float32)
        ts = ts.reshape(-1)
        if ts.numel() == batch:
            per_video_t = ts
        elif is_video and ts.numel() == batch * loss.shape[2]:
            per_video_t = ts.reshape(batch, loss.shape[2])[:, 0]
        else:
            raise ValueError(
                f"timesteps numel {ts.numel()} incompatible with batch {batch}"
            )

        logs: dict = {}
        result = loss

        if self.wv_enabled:
            with torch.no_grad():
                src=noisy.detach()
                if is_video:
                    b,c,frames,h,w=src.shape
                    src=src.permute(0,2,1,3,4).reshape(b*frames,c,h,w)
                    wavelet_t=per_video_t.repeat_interleave(frames)
                else:wavelet_t=per_video_t
                attn=compute_wavelet_attention_map(src,self._get_dwt(src.device))
                mask=get_wavelet_mask(attn,self.minimum_fraction,int(self.scheduler.config.num_train_timesteps),wavelet_t,flow_matching=False)
                if is_video:mask=mask.reshape(b,frames,1,h,w).permute(0,2,1,3,4)
            result=result*mask.to(result)
            logs["wavelet_mask_fraction"]=mask.detach().float().mean()

        if self.tm_enabled:
            if bool((loss < 0).any()):
                raise ValueError(
                    "token mining requires non-negative per-element losses"
                )
            sigmas = self._sigma_from_timesteps(per_video_t, loss)
            result = apply_token_mining(
                result,
                sigmas=sigmas,
                alpha=self.alpha,
                min_weight=self.min_weight,
                max_weight=self.max_weight,
                sigma_gate=self.sigma_gate,
            )
            logs["token_mining_sigmas"] = sigmas.detach().mean()

        return result, logs
