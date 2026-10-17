import math
import torch
import torch.nn.functional as F


class CaptionLoss:
    """Objective for categorical caption token losses on already-shifted logits."""

    _VALID = {"type", "label_smoothing", "focal_gamma", "z_loss", "kl_anchor", "temperature"}
    _SPATIAL_DIFFUSION = {"edm2", "anchor", "frequency", "hf"}
    _TYPES = {"ce", "label_smoothed_ce", "focal_ce"}

    def __init__(self, config):
        if not isinstance(config, dict):
            raise TypeError(f"CaptionLoss config must be a dict, got {type(config).__name__}")
        spatial = set(config) & self._SPATIAL_DIFFUSION
        if spatial:
            raise ValueError(
                f"CaptionLoss received spatial/diffusion key(s) {sorted(spatial)}; "
                "these control spatial/diffusion objectives and are invalid for a categorical "
                "caption token loss. Remove them from the caption loss config."
            )
        unknown = set(config) - self._VALID
        if unknown:
            raise ValueError(f"Unknown CaptionLoss config keys: {sorted(unknown)}")
        self.loss_type = config.get("type", "ce")
        if self.loss_type not in self._TYPES:
            raise ValueError(f"Unknown loss type {self.loss_type!r}; expected one of {sorted(self._TYPES)}")
        self.label_smoothing = config.get("label_smoothing", 0.0)
        self.focal_gamma = config.get("focal_gamma", 2.0)
        self.z_loss = config.get("z_loss", 0.0)
        self.kl_anchor = config.get("kl_anchor", 0.0)
        self.temperature = config.get("temperature", 1.0)
        for name, val in (("label_smoothing", self.label_smoothing), ("focal_gamma", self.focal_gamma),
                          ("z_loss", self.z_loss), ("kl_anchor", self.kl_anchor)):
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be a finite nonnegative float, got {val!r}")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) \
                or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError(f"temperature must be a finite float > 0, got {self.temperature!r}")
        if self.label_smoothing >= 1.0:
            raise ValueError(f"label_smoothing must satisfy 0 <= s < 1, got {self.label_smoothing!r}")

    def _validate(self, logits, targets, reference_logits):
        if not torch.is_tensor(logits) or logits.dim() != 3:
            raise ValueError("logits must be a [B, M, V] tensor")
        if not torch.is_tensor(targets) or targets.dim() != 2:
            raise ValueError("targets must be a [B, M] long tensor")
        b, m, v = logits.shape
        if targets.shape != (b, m):
            raise ValueError(f"targets shape {tuple(targets.shape)} does not match logits [B, M] = {(b, m)}")
        if targets.dtype != torch.long:
            raise ValueError(f"targets must be dtype long, got {targets.dtype}")
        valid = targets != -100
        if not bool(valid.any()):
            raise ValueError("no valid (non -100) target tokens to compute a caption loss over")
        if reference_logits is not None:
            if not torch.is_tensor(reference_logits) or tuple(reference_logits.shape) != (b, m, v):
                raise ValueError("reference_logits must match logits shape [B, M, V]")
        return valid

    def __call__(self, logits, targets, reference_logits=None):
        valid = self._validate(logits, targets, reference_logits)
        logits = logits.float()
        mask = valid.to(logits.dtype)
        denom = mask.sum().clamp(min=1.0)

        raw = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none", ignore_index=-100).reshape_as(targets)
        raw_masked = torch.where(valid, raw, torch.zeros_like(raw))

        if self.loss_type == "label_smoothed_ce":
            per_token = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none", ignore_index=-100,
                                        label_smoothing=self.label_smoothing).reshape_as(targets)
        elif self.loss_type == "focal_ce":
            per_token = raw_masked * (1.0 - torch.exp(-raw_masked)).pow(self.focal_gamma)
        else:
            per_token = raw_masked
        per_token = torch.where(valid, per_token, torch.zeros_like(per_token))

        objective = (per_token * mask).sum() / denom
        metrics = {}

        raw_ce = (raw_masked * mask).sum() / denom
        metrics["raw_ce"] = float(raw_ce.detach())
        metrics["perplexity"] = float(math.exp(min(float(raw_ce.detach()), 80.0)))
        preds = logits.detach().argmax(dim=-1)
        metrics["token_accuracy"] = float(((preds == targets) & valid).sum().item() / max(int(valid.sum().item()), 1))

        if self.z_loss > 0.0:
            z = torch.logsumexp(logits.float(), dim=-1).pow(2)
            z_term = (z * mask).sum() / denom * self.z_loss
            objective = objective + z_term.to(objective.dtype)
            metrics["z_loss"] = float(z_term.detach())

        if self.kl_anchor > 0.0:
            if reference_logits is None:
                raise ValueError("kl_anchor > 0 requires reference_logits")
            ref = reference_logits.detach().float()
            with torch.no_grad():
                p_ref = F.softmax(ref / self.temperature, dim=-1)
            log_p_student = F.log_softmax(logits.float() / self.temperature, dim=-1)
            kl = F.kl_div(log_p_student, p_ref, reduction="none").sum(dim=-1)
            kl = kl * (self.temperature ** 2)
            kl_term = (kl * mask).sum() / denom * self.kl_anchor
            objective = objective + kl_term.to(objective.dtype)
            metrics["kl_anchor"] = float(kl_term.detach())

        metrics["loss_objective"] = float(objective.detach())
        return objective, metrics
