"""Optimizer and scheduler factory for training_extras.

Resolves optimizer/scheduler names to classes, coerces string kwargs,
and returns configured instances. Vendored custom optimizers under
``vendor/LoraEasyCustomOptimizer`` are imported lazily and individually.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import sys
from contextlib import contextmanager

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX_PATH = os.path.join(_HERE, "optimizer-index.json")
_VENDOR_DIR = os.path.join(_HERE, "vendor")
_VENDOR_PKG = "LoraEasyCustomOptimizer"

_INDEX = None
_SYS_PATH_READY = False

_TORCH_ALIASES = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "sgd": "torch.optim.SGD",
}


def _load_index():
    global _INDEX
    if _INDEX is None:
        with open(_INDEX_PATH, "r", encoding="utf-8") as fh:
            _INDEX = json.load(fh)
    return _INDEX


def _ensure_vendor_path():
    global _SYS_PATH_READY
    if not _SYS_PATH_READY:
        if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
            sys.path.insert(0, _VENDOR_DIR)
        _SYS_PATH_READY = True


def _coerce_scalar(value):
    """Return (coerced, ok). ok=False means keep value as-is (plain string)."""
    if not isinstance(value, str):
        return value, False
    low = value.strip().lower()
    if low == "true":
        return True, True
    if low == "false":
        return False, True
    try:
        return ast.literal_eval(value), True
    except (ValueError, SyntaxError):
        return value, False


def coerce_args(mapping):
    """Recursively coerce string kwargs. Never evaluates plain strings."""
    if isinstance(mapping, dict):
        return {k: coerce_args(v) for k, v in mapping.items()}
    if isinstance(mapping, (list, tuple)):
        return type(mapping)(coerce_args(v) for v in mapping)
    coerced, ok = _coerce_scalar(mapping)
    return coerced if ok else mapping


def _lookup_index_key(name):
    index = _load_index()
    if not isinstance(index, dict):
        raise ValueError("optimizer-index.json must contain a mapping")
    low = name.lower()
    matches = [k for k in index if k.lower() == low]
    if len(matches) == 1:
        return index[matches[0]]
    if len(matches) > 1:
        raise ValueError(
            "Ambiguous optimizer name %r matches keys: %s" % (name, sorted(matches))
        )
    return None


def _import_target(module_name):
    _ensure_vendor_path()
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "Missing dependency or module for %r: %s" % (module_name, exc)
        ) from exc


def _resolve_dotted(path):
    parts = path.split(".")
    last_exc = None
    for split in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split])
        attrs = parts[split:]
        try:
            obj = _import_target(mod_name)
        except ImportError as exc:
            last_exc = exc
            continue
        try:
            for attr in attrs:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            raise ImportError(
                "Cannot resolve %r: %s" % (path, exc)
            ) from exc
        return obj
    raise ImportError("Cannot resolve %r: %s" % (path, last_exc))


def resolve(name):
    """Resolve an optimizer class from a name string, path, or FQN."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Optimizer name must be a non-empty string")
    raw = name.strip()

    low = raw.lower()
    if low in _TORCH_ALIASES:
        return _resolve_dotted(_TORCH_ALIASES[low])

    easy_target = None
    if raw.lower().startswith("easy:"):
        easy_target = raw[5:].strip()
    elif raw.startswith(_VENDOR_PKG + "."):
        easy_target = raw[len(_VENDOR_PKG) + 1:].strip()

    if easy_target is not None:
        if easy_target.lower() in _TORCH_ALIASES:
            return _resolve_dotted(_TORCH_ALIASES[easy_target.lower()])
        if "." in easy_target:
            return _resolve_dotted(_VENDOR_PKG + "." + easy_target)
        mapped = _lookup_index_key(easy_target)
        if mapped is None:
            raise ValueError("Unknown optimizer in index: %r" % easy_target)
        return _resolve_dotted(_VENDOR_PKG + "." + mapped)

    return _resolve_dotted(raw)


def _reject_closure(klass):
    try:
        params = inspect.signature(klass.step).parameters
    except (TypeError, ValueError):
        try:
            params = inspect.signature(klass.__init__).parameters
        except (TypeError, ValueError):
            return
    if "closure" in params and params["closure"].default is inspect.Parameter.empty:
        raise ValueError(
            "Optimizer %r requires a 'closure' argument; not supported by create_optimizer"
            % getattr(klass, "__name__", klass)
        )


def _to_param_groups(params):
    if isinstance(params, dict):
        raise TypeError(
            "params must be an iterable of tensors or a list of param-group dicts"
        )
    if isinstance(params, (list, tuple)):
        items = list(params)
        if items and all(isinstance(p, dict) for p in items):
            return [dict(g) for g in items]
    return list(params)


def create_optimizer(params, name, lr, kwargs=None):
    klass = resolve(name)
    if not (isinstance(klass, type) and issubclass(klass, Optimizer)):
        raise TypeError(
            "%r did not resolve to a torch.optim.Optimizer subclass" % (name,)
        )
    _reject_closure(klass)

    groups = _to_param_groups(params)
    extra = coerce_args(dict(kwargs)) if kwargs else {}
    if "lr" in extra:
        raise ValueError("Specify the learning rate once, outside optimizer arguments")

    if groups and isinstance(groups[0], dict):
        copy = [dict(g) for g in groups]
        for group in copy:
            group.setdefault("lr", lr)
        return klass(copy, lr=lr, **extra)
    return klass(groups, lr=lr, **extra)


def _step_count(total_steps):
    steps = int(total_steps)
    if steps <= 0:
        raise ValueError("total_steps must be positive")
    return steps


def _base_lrs(optimizer):
    return [g.get("lr", 0.0) for g in optimizer.param_groups]


def _warmup_lambda(warmup_steps):
    warmup_steps = int(warmup_steps or 0)

    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return fn


def _schedule_lambda(kind, total_steps, warmup_steps, min_lr, base_lrs, default_min=0.0):
    warmup_steps = int(warmup_steps or 0)

    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
        if kind == "cosine":
            import math
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif kind == "linear":
            factor = 1.0 - progress
        else:
            factor = 1.0
        if min_lr is None:
            return factor
        ratios = [
            (min_lr / base) if base else default_min for base in base_lrs
        ]
        floor = sum(ratios) / len(ratios) if ratios else default_min
        return factor * (1.0 - floor) + floor

    return fn


def _rawr_alias(name):
    if not isinstance(name, str):
        return False
    raw = name.strip()
    low = raw.lower()
    return low in ("easy:rawr", "rawr")


def _resolve_rawr():
    try:
        return _resolve_dotted(_VENDOR_PKG + ".RexAnnealingWarmRestarts.RexAnnealingWarmRestarts")
    except ImportError:
        mapped = _lookup_index_key("RexAnnealingWarmRestarts")
        if mapped is None:
            raise ImportError("RexAnnealingWarmRestarts is not available in the vendor package")
        return _resolve_dotted(mapped)


def _normalize_total_iters(extra):
    if "total_iters" in extra and "total_steps" not in extra:
        extra["total_steps"] = extra.pop("total_iters")
    return extra


def create_scheduler(optimizer, name, total_steps, kwargs=None):
    steps = _step_count(total_steps)
    extra = coerce_args(dict(kwargs)) if kwargs else {}

    if name is None or (isinstance(name, str) and not name.strip()):
        kind = "constant"
    else:
        kind = str(name).strip()

    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer instance")

    low = kind.lower()
    extra.pop("total_iters", None)
    extra.pop("total_steps", None)

    if low in ("constant", "default"):
        if extra:
            raise TypeError(f"Unexpected constant scheduler arguments: {sorted(extra)}")
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    if low == 'constant_with_warmup':
        warmup = int(extra.pop('warmup_steps', 0))
        if warmup < 0 or extra: raise ValueError('constant_with_warmup accepts nonnegative warmup_steps only')
        return LambdaLR(optimizer, lr_lambda=_warmup_lambda(warmup))

    if low in ("cosine", "linear"):
        warmup = int(extra.pop("warmup_steps", 0) or 0)
        min_lr = extra.pop("min_lr", None)
        if extra:
            raise TypeError(
                "Unexpected scheduler kwargs for %r: %s" % (kind, sorted(extra))
            )
        functions = [_schedule_lambda(low, steps, warmup, min_lr, [base]) for base in _base_lrs(optimizer)]
        return LambdaLR(optimizer, lr_lambda=functions)

    if _rawr_alias(kind):
        klass = _resolve_rawr()
        if not (isinstance(klass, type) and issubclass(klass, LRScheduler)):
            raise TypeError("RexAnnealingWarmRestarts is not an LRScheduler subclass")
        params = {
            "gamma": extra.pop("gamma", 0.9),
            "d": extra.pop("d", 0.9),
            "first_cycle_max_steps": extra.pop("first_cycle_max_steps", steps),
            "warmup_steps": int(extra.pop("warmup_steps", 0) or 0),
            "min_lr": extra.pop("min_lr", 1e-6),
        }
        _normalize_total_iters(extra)
        params.update(extra)
        return klass(optimizer, **params)

    if isinstance(kind, str) and kind.lower().startswith("easy:"):
        klass = resolve(kind)
    else:
        klass = resolve(kind)

    if not (isinstance(klass, type) and issubclass(klass, LRScheduler)):
        raise TypeError("%r did not resolve to an LRScheduler subclass" % (kind,))
    _normalize_total_iters(extra)
    return klass(optimizer, **extra)


@contextmanager
def evaluation_optimizer(optimizer):
    has_pair = (
        callable(getattr(optimizer, "eval", None))
        and callable(getattr(optimizer, "train", None))
    )
    if has_pair:
        optimizer.eval()
    try:
        yield optimizer
    finally:
        if has_pair:
            optimizer.train()


def lora_plus_groups(groups, network, default_lr, ratio, text_encoder_ratio=None):
    """Preserve block/base LRs while giving each up factor its own ratio."""
    te_ratio=ratio if text_encoder_ratio is None else text_encoder_ratio
    factor_ids={id(p):(te_ratio if "lora_te" in n else ratio) for n,p in network.named_parameters() if "lora_up." in n or ".lora_B." in n}
    if not factor_ids:raise ValueError("LoRA+ requested but no up/B factors found")
    result=[]
    for group in groups:
        by_ratio={}
        for p in group["params"]:by_ratio.setdefault(factor_ids.get(id(p),1.0),[]).append(p)
        for r,params in by_ratio.items():
            g=dict(group,params=params)
            g["lr"]=group.get("lr",default_lr)*r
            result.append(g)
    return result
