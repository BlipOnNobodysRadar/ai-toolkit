# training_extras/edm.py
from __future__ import annotations

import hashlib
import json
import os

import torch

try:
    from .vendor.sd_scripts.edm2_loss import AdaptiveLossWeightMLP
except ImportError:  # pragma: no cover - direct script execution
    from vendor.sd_scripts.edm2_loss import AdaptiveLossWeightMLP

try:
    from . import optim as optim_mod
except ImportError:  # pragma: no cover
    import optim as optim_mod


_SINGLE_GPU_ONLY_MSG = (
    "EDMWeighting is single-GPU only; WORLD_SIZE>1 (DDP) is unsupported."
)


def _normalized_cfg(cfg: dict) -> dict:
    out = {
        "enabled": bool(cfg.get("enabled", False)),
        "channels": int(cfg.get("channels", 128)),
        "float64": bool(cfg.get("float64", False)),
        "importance": bool(cfg.get("importance", False)),
        "importance_max": float(cfg.get("importance_max", 10)),
        "importance_min_snr": float(cfg.get("importance_min_snr", 1)),
        "optimizer": str(cfg.get("optimizer", "easy:SCORN")),
        "lr": float(cfg.get("lr", 0.015)),
        "optimizer_args": dict(cfg.get("optimizer_args", {}) or {}),
        "scheduler": str(cfg.get("scheduler", "constant")),
        "scheduler_args": dict(cfg.get("scheduler_args", {}) or {}),
        "total_steps": int(cfg.get("total_steps", 1000)),
    }
    unknown=set(cfg)-set(out)
    if unknown:raise ValueError(f'Unknown EDM2 keys: {sorted(unknown)}')
    if out['importance_min_snr'] <= 0 or out['importance_max'] <= 0:
        raise ValueError('EDM2 importance parameters must be positive')
    return out


def _cfg_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class EDMWeighting(torch.nn.Module):
    """Adaptive EDM loss weighting with independently optimised aux nets."""

    def __init__(self, cfg: dict, scheduler, device):
        super().__init__()
        self.cfg = _normalized_cfg(cfg)
        self.device = torch.device(device)
        self.enabled = self.cfg["enabled"]
        self.model = None
        self.aux_optimizer = None
        self.aux_scheduler = None

        if not self.enabled:
            return

        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        if world_size > 1:
            raise RuntimeError(_SINGLE_GPU_ONLY_MSG)

        channels = self.cfg["channels"]
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"cfg.channels must be a positive int, got {channels!r}")

        dtype = torch.float64 if self.cfg["float64"] else torch.float32

        if torch.is_tensor(scheduler):
            raise TypeError("EDMWeighting needs a noise-scheduler object, not a tensor.")
        num_train_timesteps = getattr(scheduler, "config", None)
        num_train_timesteps = getattr(
            num_train_timesteps, "num_train_timesteps", None
        )
        if num_train_timesteps is None:
            raise ValueError("scheduler.config.num_train_timesteps is required.")

        self.model = AdaptiveLossWeightMLP(
            scheduler,
            logvar_channels=channels,
            lambda_weights=None,
            device=device,
            dtype=dtype,
            use_importance_weights=self.cfg["importance"],
            importance_weights_max_weight=self.cfg["importance_max"],
            importance_weights_min_snr_gamma=self.cfg["importance_min_snr"],
            importance_weights=None,
        ).to(self.device)
        for p in self.model.parameters():
            p.data = p.data.to(self.device)
        if hasattr(self.model, "lambda_weights") and self.model.lambda_weights is not None:
            self.model.lambda_weights = self.model.lambda_weights.to(self.device)

        self.aux_optimizer = optim_mod.create_optimizer(
            self.model.parameters(),
            self.cfg["optimizer"],
            self.cfg["lr"],
            self.cfg["optimizer_args"],
        )
        self.aux_scheduler = optim_mod.create_scheduler(
            self.aux_optimizer,
            self.cfg["scheduler"],
            self.cfg["total_steps"],
            self.cfg["scheduler_args"],
        )

    def weight(self, per_sample, timesteps):
        if not self.enabled:
            return per_sample, {}
        if timesteps.device != self.device:
            timesteps = timesteps.to(self.device)
        weighted_objective, weighted_data_loss = self.model(
            per_sample.to(self.device), timesteps
        )
        logs = {
            "loss/edm_objective": weighted_objective.detach().float().mean(),
            "loss/edm_data": weighted_data_loss.detach().float().mean(),
            "loss/raw": per_sample.detach().float().mean(),
        }
        return (weighted_objective, logs)

    def step(self):
        if not self.enabled:
            return
        self.aux_optimizer.step()
        self.aux_optimizer.zero_grad(set_to_none=True)
        self.aux_scheduler.step()

    def zero_grad(self):
        if not self.enabled:
            return
        self.aux_optimizer.zero_grad(set_to_none=True)

    def checkpoint(self, path):
        if not self.enabled:
            return
        payload = {
            "module": self.state_dict(),
            "optimizer": self.aux_optimizer.state_dict(),
            "scheduler": self.aux_scheduler.state_dict(),
            "config": self.cfg,
            "config_hash": _cfg_hash(self.cfg),
        }
        path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def restore(self, path):
        if not self.enabled:
            return
        payload = torch.load(
            os.fspath(path), map_location=self.device, weights_only=True
        )
        if payload.get("config_hash") != _cfg_hash(self.cfg):
            raise RuntimeError(
                "EDMWeighting checkpoint config hash mismatch; refusing to restore."
            )
        self.load_state_dict(payload["module"])
        self.aux_scheduler.load_state_dict(payload["scheduler"])
        self.aux_optimizer.load_state_dict(payload["optimizer"])
