"""RelAdamW: Relative AdamW with Sinkhorn-balanced normalized gradients.

Step order per parameter tensor (as specified):
    1. Multi-axis RMS Sinkhorn gradient balancing (copied from RelNormSinkhorn).
    2. Rescale balanced grad to parameter L2 norm, floored at ``unit_clamp`` (default 1.0).
    3. Update Adam second moment on the surprise ``d = g_normed - m_{t-1}``
       (old momentum, before the momentum update), then craft per-coordinate
       denominator and instantly normalize it to RMS 1.0 per-tensor.
    4. Update momentum on ``g_normed``, craft direction with optional Nesterov lerp,
       divide by the RMS-1 denominator.
    5. Relative weight decay first, then cautious masking (in that order).

Because the denominator is normalized to RMS 1.0, the update keeps the
``O(target_norm)`` scale of the normalized stream, so ``lr`` behaves as a
relative lr (RelNorm-style ``1e-2``), not an Adam-style absolute lr.

Memory: 2 state buffers per parameter (momentum + variance).
"""

from typing import Iterable, Union
import torch
from torch.optim import Optimizer


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor) -> None:
    """Stochastically round a float32 source into bfloat16 to avoid stagnation."""
    assert source.dtype is torch.float32, f"source must be float32, got {source.dtype}"
    assert target.dtype is torch.bfloat16, f"target must be bfloat16, got {target.dtype}"
    with torch.no_grad():
        result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
        result.add_(source.view(dtype=torch.int32))
        result.bitwise_and_(-65536)
        target.copy_(result.view(dtype=torch.float32).to(target.dtype))


def sinkhorn_rms_balance(
    grad: torch.Tensor,
    num_iter: int = 3,
    shape_mode: str = "multi_axis",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Balance gradient energy across tensor dimensions via iterative RMS normalization.

    Copied from RelNormSinkhorn: 0D/1D -> global RMS; 2D -> alternate row/col
    RMS; ND -> multi_axis (per-axis) or flatten_2d.
    """
    if grad.ndim <= 1 or num_iter <= 0:
        return grad * torch.rsqrt(grad.square().mean() + eps)

    out = grad * torch.rsqrt(grad.square().mean() + eps)

    if grad.ndim == 2:
        for _ in range(num_iter):
            out.mul_(torch.rsqrt(out.square().mean(dim=1, keepdim=True) + eps))
            out.mul_(torch.rsqrt(out.square().mean(dim=0, keepdim=True) + eps))
        return out

    if shape_mode == "flatten_2d":
        orig_shape = out.shape
        flat = out.reshape(orig_shape[0], -1)
        for _ in range(num_iter):
            flat.mul_(torch.rsqrt(flat.square().mean(dim=1, keepdim=True) + eps))
            flat.mul_(torch.rsqrt(flat.square().mean(dim=0, keepdim=True) + eps))
        return flat.reshape(orig_shape)
    elif shape_mode == "multi_axis":
        ndim = out.ndim
        reduce_dims_per_axis = [tuple(d for d in range(ndim) if d != axis) for axis in range(ndim)]
        for _ in range(num_iter):
            for r_dims in reduce_dims_per_axis:
                out.mul_(torch.rsqrt(out.square().mean(dim=r_dims, keepdim=True) + eps))
        return out
    else:
        raise ValueError(f"Invalid shape_mode: {shape_mode}. Must be 'flatten_2d' or 'multi_axis'")


class RelAdamW(Optimizer):
    r"""RelAdamW: Relative AdamW on a Sinkhorn-balanced, weight-normed stream.

    Per parameter tensor W with raw gradient G at step t:

        1. ``G_sink = SinkhornRMS(G)``
        2. ``target = max(||W||_2, unit_clamp)``,
           ``G_norm = G_sink * (target / (||G_sink||_2 + eps))``
        3. ``d = G_norm - M_{t-1}``,
           ``V_t = beta2*V_{t-1} + (1-beta2)*d^2``,
           ``V_hat = V_t/(1-beta2^t)`` if debias else ``V_t``,
           ``D = sqrt(V_hat)/sqrt(mean(V_hat)+eps) + eps`` (RMS(D)==~1)
        4. ``M_t = beta1*M_{t-1} + (1-beta1)*G_norm``,
           ``M_hat = M_t/(1-beta1^t)`` if debias else ``M_t``,
           ``U = lerp(G_norm, M_hat, beta1)`` if nesterov else ``M_hat``,
           ``U_raw = U / D``
        5. Relative WD: ``scale = ||U_raw||_2/max(||W||_2,eps)``,
           ``U_wd = U_raw + wd*scale*W`` (with optional cautious_wd mask
           and overshoot flip guard).
        6. Cautious: ``mask = (G * U_wd > 0)``,
           ``U_final = U_wd*mask/mean(mask)``.
        7. ``W -= lr * U_final``

    Arguments:
        params: parameters to optimize.
        lr: relative learning rate (default: 1e-2).
        betas: (beta1 momentum, beta2 variance) (default: (0.95, 0.999)).
        eps: numerical stability constant (default: 1e-12).
        weight_decay: relative WD coefficient (default: 0.1). Per-parameter
            scaling is supported: an explicit ``p.weight_decay_ratio``
            attribute scales the effective WD (``weight_decay * ratio``);
            parameters without it skip weight decay (ratio 0.0) when flagged as
            bias/norm/scalar/DoRA-scale and otherwise use the full coefficient
            (ratio 1.0).
        sinkhorn_iter: Sinkhorn iterations (default: 3).
        shape_mode: 'multi_axis' or 'flatten_2d' for >2D (default: 'multi_axis').
        unit_clamp: floor for target norm (default: 1.0).
        nesterov: blend G_norm with momentum (default: True).
        cautious_update: mask final update against raw G sign (default: True).
        cautious_wd: apply WD only where sign(U_raw)==sign(W) (default: True).
        overshoot_protection: zero coords where WD flips update sign (default: True).
        debias: standard Adam bias correction on M and V (default: True).
            Note: V bias correction cancels in the RMS-1 ratio; kept for the
            eps floor semantics.
        stochastic_fp: stochastic rounding for bf16 params (default: True).
        compile_step: torch.compile the per-parameter core (default: False).
        foreach: batched multi-tensor path (default: False).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-2,
        betas=(0.95, 0.999),
        eps: float = 1e-12,
        weight_decay: float = 0.1,
        sinkhorn_iter: int = 3,
        shape_mode: str = "multi_axis",
        unit_clamp: float = 1.0,
        nesterov: bool = True,
        cautious_update: bool = True,
        cautious_wd: bool = True,
        overshoot_protection: bool = True,
        debias: bool = True,
        stochastic_fp: bool = True,
        compile_step: bool = False,
        foreach: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if sinkhorn_iter < 0:
            raise ValueError(f"Invalid sinkhorn_iter: {sinkhorn_iter}")
        if shape_mode not in ("flatten_2d", "multi_axis"):
            raise ValueError(f"Invalid shape_mode: {shape_mode}. Must be 'flatten_2d' or 'multi_axis'")
        if unit_clamp < 0.0:
            raise ValueError(f"Invalid unit_clamp value: {unit_clamp}")

        defaults = dict(
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
            sinkhorn_iter=sinkhorn_iter,
            shape_mode=shape_mode,
            unit_clamp=unit_clamp,
            nesterov=nesterov,
            cautious_update=cautious_update,
            cautious_wd=cautious_wd,
            overshoot_protection=overshoot_protection,
            debias=debias,
            stochastic_fp=stochastic_fp,
        )
        super().__init__(params, defaults)

        self._compile_step = compile_step
        self._foreach = foreach

        if self._compile_step:
            try:
                torch._dynamo.config.recompile_limit = max(
                    torch._dynamo.config.recompile_limit, 64
                )
                self._compiled_step = torch.compile(
                    self._reladamw_step_core,
                    fullgraph=True,
                    dynamic=None,
                )
            except Exception as e:
                import logging
                logging.warning(
                    f"torch.compile failed to initialize: {e}. Falling back to uncompiled step."
                )
                self._compiled_step = self._reladamw_step_core
        else:
            self._compiled_step = self._reladamw_step_core

    @staticmethod
    def _parameter_weight_decay(p: torch.Tensor, group: dict) -> float:
        """Effective per-parameter weight decay (ported from WarpAINO).

        An explicit ``p.weight_decay_ratio`` attribute scales the group's
        ``weight_decay``. Without it, bias, norm, scalar, and DoRA-scale
        parameters skip weight decay (ratio 0.0); all other parameters use the
        full coefficient (ratio 1.0).
        """
        wd_ratio = getattr(p, "weight_decay_ratio", None)
        if wd_ratio is None:
            if (
                getattr(p, "is_bias", False)
                or getattr(p, "is_norm", False)
                or getattr(p, "is_scalar", False)
                or getattr(p, "_is_dora_scale", False)
            ):
                wd_ratio = 0.0
            else:
                wd_ratio = 1.0
        return float(group["weight_decay"]) * float(wd_ratio)

    @staticmethod
    def _reladamw_step_core(
        p_data: torch.Tensor,
        g_data: torch.Tensor,
        momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        step_t: torch.Tensor,
        beta1: float,
        beta2: float,
        weight_decay: float,
        sinkhorn_iter: int,
        shape_mode: str,
        unit_clamp: float,
        nesterov: bool,
        cautious_update: bool,
        cautious_wd: bool,
        overshoot_protection: bool,
        debias: bool,
        eps: float,
    ) -> torch.Tensor:
        """Core math for one parameter tensor. Mutates momentum/exp_avg_sq."""
        # 1. Sinkhorn RMS balancing.
        g_sink = sinkhorn_rms_balance(g_data, num_iter=sinkhorn_iter, shape_mode=shape_mode, eps=eps)

        # 2. Rescale to weight norm, floored at unit_clamp.
        p_norm = torch.sqrt(torch.sum(p_data.square()) + eps)
        s_norm = torch.sqrt(torch.sum(g_sink.square()) + eps)
        target_norm = torch.clamp_min(p_norm, unit_clamp)
        g_normed = g_sink * (target_norm / (s_norm + eps))

        # 3. Denominator on surprise vs OLD momentum (before momentum update).
        d_sq = (g_normed - momentum).square()
        exp_avg_sq.lerp_(d_sq, 1.0 - beta2)

        if debias:
            bc1 = 1.0 - beta1 ** step_t
            bc2 = 1.0 - beta2 ** step_t
            v_hat = exp_avg_sq / bc2
        else:
            bc1 = 1.0
            v_hat = exp_avg_sq

        # Instant per-tensor RMS-1 normalization of the denominator.
        # RMS(sqrt(v_hat)) == sqrt(mean(v_hat)), so this ratio has RMS 1.
        s_var = torch.sqrt(torch.mean(v_hat) + eps)
        denom = torch.sqrt(v_hat) / s_var + eps

        # 4. Momentum update, then direction.
        momentum.lerp_(g_normed, 1.0 - beta1)
        m_hat = momentum / bc1 if debias else momentum

        if nesterov:
            u_m = torch.lerp(g_normed, m_hat, beta1)
        else:
            u_m = m_hat

        u_raw = u_m / denom

        # 5a. Relative weight decay FIRST.
        if weight_decay > 0.0:
            u_norm = torch.sqrt(torch.sum(u_raw.square()) + eps)
            # p_norm reused: p_data has not changed.
            scale = u_norm / torch.clamp_min(p_norm, eps)
            wd_full = weight_decay * scale * p_data
            if cautious_wd:
                wd_mask = (u_raw.sign() == p_data.sign()).to(u_raw.dtype)
                wd_term = wd_full * wd_mask
            else:
                wd_term = wd_full
            if overshoot_protection:
                flip = (u_raw * (u_raw + wd_term)) < 0
                u_wd = (u_raw + wd_term) * (~flip)
            else:
                u_wd = u_raw + wd_term
        else:
            u_wd = u_raw

        # 5b. Cautious masking LAST, against raw gradient sign.
        if cautious_update:
            mask = (g_data * u_wd > 0).to(u_wd.dtype)
            mask_mean = torch.clamp_min(torch.mean(mask), 1e-3)
            u_final = u_wd * mask / mask_mean
        else:
            u_final = u_wd

        return u_final

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["momentum"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                elif isinstance(state["step"], int):
                    state["step"] = torch.tensor(float(state["step"]), dtype=torch.float32, device=p.device)

            if self._foreach:
                self._step_foreach(group)
            else:
                self._step_native(group)

        return loss

    def _step_native(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        sinkhorn_iter = group["sinkhorn_iter"]
        shape_mode = group["shape_mode"]
        unit_clamp = group["unit_clamp"]
        nesterov = group["nesterov"]
        cautious_update = group["cautious_update"]
        cautious_wd = group["cautious_wd"]
        overshoot_protection = group["overshoot_protection"]
        debias = group["debias"]
        stochastic_fp = group["stochastic_fp"]

        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            state["step"].add_(1)
            step_t = state["step"]

            grad = p.grad.data
            if grad.dtype in (torch.bfloat16, torch.float16):
                grad = grad.float()

            is_low_prec = p.dtype in (torch.bfloat16, torch.float16)
            p_fp32 = p.data.float() if is_low_prec else p.data

            p_weight_decay = self._parameter_weight_decay(p, group)

            update = self._compiled_step(
                p_fp32,
                grad,
                state["momentum"],
                state["exp_avg_sq"],
                step_t,
                beta1,
                beta2,
                p_weight_decay,
                sinkhorn_iter,
                shape_mode,
                unit_clamp,
                nesterov,
                cautious_update,
                cautious_wd,
                overshoot_protection,
                debias,
                eps,
            )

            p_fp32.add_(update, alpha=-lr)

            if is_low_prec:
                if p.dtype is torch.bfloat16 and stochastic_fp:
                    copy_stochastic_(p.data, p_fp32)
                else:
                    p.data.copy_(p_fp32.to(p.dtype))

    def _step_foreach(self, group):
        """Batched path. Math is identical to native; per-tensor RMS stays per-tensor."""
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        sinkhorn_iter = group["sinkhorn_iter"]
        shape_mode = group["shape_mode"]
        unit_clamp = group["unit_clamp"]
        nesterov = group["nesterov"]
        cautious_update = group["cautious_update"]
        cautious_wd = group["cautious_wd"]
        overshoot_protection = group["overshoot_protection"]
        debias = group["debias"]
        stochastic_fp = group["stochastic_fp"]

        params, p_list, g_list, m_list, v_list, s_list, lp_flags, wd_list = [], [], [], [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            state["step"].add_(1)
            params.append(p)
            is_lp = p.dtype in (torch.bfloat16, torch.float16)
            lp_flags.append(is_lp)
            p_list.append(p.data.float() if is_lp else p.data)
            g = p.grad.data
            g_list.append(g.float() if g.dtype in (torch.bfloat16, torch.float16) else g)
            m_list.append(state["momentum"])
            v_list.append(state["exp_avg_sq"])
            s_list.append(state["step"])
            wd_list.append(self._parameter_weight_decay(p, group))

        if not params:
            return

        # 1+2. Sinkhorn + target-norm scaling (per-tensor, loop: iterative + shapes differ).
        g_normed_list = []
        p_norms = []
        for p_f, g_f in zip(p_list, g_list):
            g_sink = sinkhorn_rms_balance(g_f, num_iter=sinkhorn_iter, shape_mode=shape_mode, eps=eps)
            p_norm = torch.sqrt(torch.sum(p_f.square()) + eps)
            s_norm = torch.sqrt(torch.sum(g_sink.square()) + eps)
            p_norms.append(p_norm)
            target = torch.clamp_min(p_norm, unit_clamp)
            g_normed_list.append(g_sink * (target / (s_norm + eps)))

        # 3. Variance on surprise vs OLD momentum.
        d_sq_list = [(g_n - m).square() for g_n, m in zip(g_normed_list, m_list)]
        torch._foreach_lerp_(v_list, d_sq_list, 1.0 - beta2)

        denom_list = []
        bc1_list = []
        for v, s_t in zip(v_list, s_list):
            if debias:
                bc1 = 1.0 - beta1 ** s_t
                bc2 = 1.0 - beta2 ** s_t
                v_hat = v / bc2
            else:
                bc1 = 1.0
                v_hat = v
            bc1_list.append(bc1)
            s_var = torch.sqrt(torch.mean(v_hat) + eps)
            denom_list.append(torch.sqrt(v_hat) / s_var + eps)

        # 4. Momentum update, then direction / precondition.
        torch._foreach_lerp_(m_list, g_normed_list, 1.0 - beta1)
        if debias:
            m_hat_list = [m / bc1 for m, bc1 in zip(m_list, bc1_list)]
        else:
            m_hat_list = m_list
        if nesterov:
            u_list = [torch.lerp(g_n, m_h, beta1) for g_n, m_h in zip(g_normed_list, m_hat_list)]
        else:
            u_list = m_hat_list
        u_raw_list = [u / d for u, d in zip(u_list, denom_list)]

        # 5a. Relative WD first (per-parameter: bias/norm/scalar/DoRA-scale
        # parameters skip WD unless they carry an explicit weight_decay_ratio).
        if weight_decay > 0.0:
            u_wd_list = []
            for p_f, u_r, p_norm, wd in zip(p_list, u_raw_list, p_norms, wd_list):
                u_norm = torch.sqrt(torch.sum(u_r.square()) + eps)
                scale = u_norm / torch.clamp_min(p_norm, eps)
                wd_full = wd * scale * p_f
                if cautious_wd:
                    wd_term = wd_full * (u_r.sign() == p_f.sign()).to(u_r.dtype)
                else:
                    wd_term = wd_full
                if overshoot_protection:
                    flip = (u_r * (u_r + wd_term)) < 0
                    u_wd_list.append((u_r + wd_term) * (~flip))
                else:
                    u_wd_list.append(u_r + wd_term)
        else:
            u_wd_list = u_raw_list

        # 5b. Cautious last.
        if cautious_update:
            u_final_list = []
            for g_f, u_w in zip(g_list, u_wd_list):
                mask = (g_f * u_w > 0).to(u_w.dtype)
                mask_mean = torch.clamp_min(mask.mean(), 1e-3)
                u_final_list.append(u_w * mask / mask_mean)
        else:
            u_final_list = u_wd_list

        torch._foreach_add_(p_list, u_final_list, alpha=-lr)

        for p, p_f, is_lp in zip(params, p_list, lp_flags):
            if is_lp:
                if p.dtype is torch.bfloat16 and stochastic_fp:
                    copy_stochastic_(p.data, p_f)
                else:
                    p.data.copy_(p_f.to(p.dtype))


# Aliases
RelAdam = RelAdamW
RelativeAdamW = RelAdamW
