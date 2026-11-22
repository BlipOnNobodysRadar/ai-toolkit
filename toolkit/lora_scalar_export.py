"""Inference export for LoRAs trained with ``use_scalar`` (network_kwargs.use_scalar: true).

A module trained with the learned scalar computes ``up(down(x)) * scalar * alpha / rank`` and its save keeps the raw
``<module>.scalar`` beside the two factors, which is what resume needs (toolkit/lora_special.py). Loaders outside this
trainer do not know the key: ComfyUI logs it as not loaded and applies ``up @ down`` without the scalar, so a LoRA
whose scalar ended near 0.1 is applied about ten times too strong. This module folds every scalar into its up factor
and drops the key, which gives a file any LoRA loader reads correctly. The folded file is for inference only: do not
resume from it (the trainer would start a fresh scalar on top of the folded factors).

MOSS: the trainer's q/k/v projection is one fused module, saved as three projections that share the down factor
(MossImageModel.convert_lora_weights_before_save) while the scalar stays on the fused name (``<block>.proj.qkv.scalar``).
That one scalar is folded into all three up factors.

    python -m toolkit.lora_scalar_export <in.safetensors> [<out.safetensors>]     (default out: <in>-comfy.safetensors)
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

import torch

SCALAR_SUFFIX = ".scalar"
UP_SUFFIXES = ("lora_B.weight", "lora_up.weight")
FUSED_QKV = ".proj.qkv"
QKV_PARTS = ("q", "k", "v")


def _up_keys(state_dict, prefix: str) -> list:
    """The up-factor keys a scalar at ``prefix`` scales: the module's own, or q/k/v of a fused MOSS projection."""
    own = [f"{prefix}.{s}" for s in UP_SUFFIXES if f"{prefix}.{s}" in state_dict]
    if own:
        return own
    if prefix.endswith(FUSED_QKV):
        base = prefix[: -len("qkv")]
        for s in UP_SUFFIXES:
            keys = [f"{base}{p}.{s}" for p in QKV_PARTS]
            if all(k in state_dict for k in keys):
                return keys
    return []


def fold_scalars(state_dict) -> tuple:
    """Return (new state dict without ``.scalar`` keys, {scalar key: value}). Up factors keep their dtype; the
    product is taken in float32. Raises when a scalar has no up factor to fold into, or is not a single finite value."""
    out = OrderedDict((k, v) for k, v in state_dict.items() if not k.endswith(SCALAR_SUFFIX))
    folded = {}
    for key, value in state_dict.items():
        if not key.endswith(SCALAR_SUFFIX):
            continue
        if value.numel() != 1:
            raise ValueError(f"{key}: expected one value, got shape {tuple(value.shape)}")
        scalar = float(value.detach().float().reshape(()).item())
        if scalar != scalar or scalar in (float("inf"), float("-inf")):
            raise ValueError(f"{key}: scalar is not finite ({scalar})")
        ups = _up_keys(state_dict, key[: -len(SCALAR_SUFFIX)])
        if not ups:
            raise KeyError(f"{key}: no up factor found to fold the scalar into")
        for up_key in ups:
            up = state_dict[up_key]
            out[up_key] = (up.detach().float() * scalar).to(up.dtype)
        folded[key] = scalar
    return out, folded


def export_file(src: str, dst: str | None = None) -> tuple:
    """Write the folded copy of a .safetensors LoRA. Returns (dst, number of scalars folded, their min, their max)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if dst is None:
        root, ext = os.path.splitext(src)
        dst = f"{root}-comfy{ext}"
    if os.path.abspath(dst) == os.path.abspath(src):
        raise ValueError("the folded copy must not overwrite the training file (resume needs the raw scalar)")
    state = OrderedDict()
    with safe_open(src, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            state[k] = f.get_tensor(k)
    out, folded = fold_scalars(state)
    meta["scalar_folded"] = str(len(folded))
    save_file(out, dst, meta)
    vals = list(folded.values())
    return dst, len(vals), (min(vals) if vals else None), (max(vals) if vals else None)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    path, n, lo, hi = export_file(*sys.argv[1:])
    print(f"{path}: folded {n} scalars" + (f" (values {lo:.4f} to {hi:.4f})" if n else " (the file had none; plain copy)"))
