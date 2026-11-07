"""Where a parameter's unfinished (accumulating) gradient lives, for every consumer of it.

Several optimizers here accumulate low-precision gradients with stochastic rounding: a post-accumulate-grad hook
moves each backward's gradient into ``p._accum_grad`` and deletes ``p.grad``, and ``step()`` moves it back. Between
backward and step the pending gradient is therefore invisible to code that only looks at ``p.grad``: zero_grad,
the trainer's OOM discard, grad clipping and the RAWR checkpoint. These helpers give those consumers one view of
it (docs/STATE_MACHINES.md, M2). Torch-only on purpose: the RAWR serializer imports this.
"""
import torch

ACCUM_GRAD_ATTR = "_accum_grad"
GRAD = "grad"
HIDDEN = "hidden"


def _params(params_or_optimizer):
    groups = getattr(params_or_optimizer, "param_groups", None)
    if groups is not None:
        for group in groups:
            yield from group["params"]
        return
    for item in params_or_optimizer:
        if isinstance(item, dict):
            yield from item["params"]
        else:
            yield item


def capture(p):
    """(gradient, location) of p's pending gradient; (None, None) when there is none."""
    hidden = getattr(p, ACCUM_GRAD_ATTR, None)
    if hidden is not None and p.grad is not None:
        # The hooks never leave both: a later backward would fold p.grad into the buffer, step() would drop it.
        raise ValueError("Parameter holds both a gradient and a stochastic-accumulation buffer; refusing to "
                         "guess which one is pending")
    if hidden is not None:
        return hidden, HIDDEN
    if p.grad is not None:
        return p.grad, GRAD
    return None, None


def restore(p, grad, location):
    """Put a captured pending gradient back where it was held, replacing whatever p holds now."""
    if hasattr(p, ACCUM_GRAD_ATTR):
        delattr(p, ACCUM_GRAD_ATTR)
    p.grad = None
    if grad is None:
        return
    if location == HIDDEN:
        setattr(p, ACCUM_GRAD_ATTR, grad)
    else:
        p.grad = grad


def discard(params_or_optimizer):
    """Drop every pending gradient, hidden buffers included (OOM window reset, zero_grad)."""
    for p in _params(params_or_optimizer):
        if hasattr(p, ACCUM_GRAD_ATTR):
            delattr(p, ACCUM_GRAD_ATTR)


def materialize(params_or_optimizer):
    """Move hidden buffers into p.grad so clipping sees them; step() then uses p.grad unchanged."""
    for p in _params(params_or_optimizer):
        hidden = getattr(p, ACCUM_GRAD_ATTR, None)
        if hidden is None:
            continue
        if p.grad is not None:
            raise ValueError("Parameter holds both a gradient and a stochastic-accumulation buffer")
        delattr(p, ACCUM_GRAD_ATTR)
        p.grad = hidden


def steps_in_backward(optimizer):
    """True for optimizers that update parameters inside backward (fused post-accumulate-grad hooks).

    Not torch's Adam/AdamW ``fused=True``: that is a CUDA kernel that still steps in ``step()``."""
    optimizer = getattr(optimizer, "optimizer", optimizer)  # accelerate's AcceleratedOptimizer
    return bool(getattr(optimizer, "steps_in_backward", False))


class PendingGradZeroGradMixin:
    """zero_grad that also drops the stochastic-accumulation buffers, so "no pending gradient" means none."""

    def zero_grad(self, set_to_none: bool = True):
        discard(self)
        super().zero_grad(set_to_none=set_to_none)
