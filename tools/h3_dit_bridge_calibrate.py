#!/usr/bin/env python3
"""Calibrate the H3-DiT -> Qwen mouth bridge before interpreting deep states.

The first mouth-bridge experiment produced repetitive garbage, but its required
positive control (round-tripping H3's exact condition_proj output back into
Qwen space) was not close enough to the real Qwen layer-50 state.  Therefore
that run cannot say whether deep H3 states are decodable.

This probe isolates the failure into three pieces:

  1. hook control: inject stock Qwen's own layer-50 query state back into layer
     50.  This should reproduce the stock baseline essentially exactly.
  2. conditioner control: load H3's *actual* truncated Qwen3-VL conditioner,
     recompute the query hidden_states[50], and inject that exact 5120-d state
     directly into the stock Qwen tail.  This bypasses condition_proj inversion.
  3. inverse control: solve H3 condition_proj -> Qwen using a direct least-
     squares solve (CUDA QR when available), then inject that result.  This tells
     us whether the earlier CG inversion was the weak link.

It also verifies that the recomputed H3 conditioner state projects back to the
saved `projected` capture from h3_dit_text_probe.py.  Only after these controls
pass should we try to decode block_20..block_50 again.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.h3_dit_mouth_bridge import (
    _capture_layer_input,
    _cos,
    _generate,
    _load_condition_proj,
    _load_probe,
    _match_scale,
    _move_inputs,
    _raw_inputs,
    _resolve_dit,
    _rms,
)
from tools.h3_mouth_probe import (
    DEFAULT_H3_MODEL,
    DEFAULT_STOCK_MODEL,
    LOWER_LAYER_COUNT,
    _cleanup_cuda,
    _cuda_mem,
    _load_h3_conditioner,
    _load_stock,
)


def _direct_inverse(
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Exact least-squares inversion of y = x W^T + b.

    torch.linalg.lstsq solves W x^T ~= (y-b)^T.  W is only ~105 MiB fp32;
    this is computationally heavier than CG but is a one-time calibration and
    avoids squaring the condition number through normal equations.
    """
    if y.ndim != 3 or y.shape[0] != 1:
        raise ValueError(f"Expected (1,L,5376), got {tuple(y.shape)}")

    w = weight.to(device=device, dtype=torch.float32)
    b = bias.to(device=device, dtype=torch.float32)
    target = y[0].to(device=device, dtype=torch.float32) - b
    rhs = target.T.contiguous()  # (5376, L)

    print(f"[H3 calibrate] Direct least-squares solve on {device}: W={tuple(w.shape)}, rhs={tuple(rhs.shape)}")
    try:
        # CUDA currently supports the QR-style `gels` driver.  On CPU, letting
        # PyTorch choose its default is more robust, but this code path normally
        # runs before the 32B donor is loaded so CUDA has ample headroom.
        if device.type == "cuda":
            sol = torch.linalg.lstsq(w, rhs, driver="gels").solution
        else:
            sol = torch.linalg.lstsq(w, rhs).solution
    except Exception as exc:
        print(f"[H3 calibrate] CUDA/direct solve failed ({type(exc).__name__}: {exc}); retrying on CPU")
        _cleanup_cuda()
        wc = weight.float().contiguous()
        rhsc = (y[0].float() - bias.float()).T.contiguous()
        sol = torch.linalg.lstsq(wc, rhsc).solution
        sol = sol.to(device)
        w = weight.to(device=device, dtype=torch.float32)
        b = bias.to(device=device, dtype=torch.float32)

    x = sol.T.unsqueeze(0)  # (1,L,5120)
    recon = x[0] @ w.T + b
    target_y = y[0].to(device=device, dtype=torch.float32)
    rel = _rms((recon - target_y).cpu()) / max(_rms((target_y - b).cpu()), 1e-20)
    out = x.detach().float().cpu()
    stats = {
        "recon_relative_rms": float(rel),
        "solution_rms": _rms(out),
    }
    del w, b, rhs, sol, recon, target_y, x
    _cleanup_cuda()
    return out, stats


def _encode_exact_h3_query(models_path: Path, h3_model: str, query: str) -> torch.Tensor:
    """Recompute the exact 5120-d Qwen state H3 uses before condition_proj."""
    # Critical: set this before importing H3 internals.  toolkit.paths freezes
    # MODELS_PATH at import time.
    os.environ["MODELS_PATH"] = str(models_path)

    holder, tokenizer, processor, te = _load_h3_conditioner(str(models_path), h3_model)
    from extensions_built_in.diffusion_models.minimax_h3.src.text_encoder import (
        encode_minimax_h3_prompt,
    )

    print("[H3 calibrate] Moving H3 truncated Qwen conditioner to CUDA")
    te.to("cuda")
    with torch.inference_mode():
        emb, _tags = encode_minimax_h3_prompt(
            te,
            tokenizer,
            processor,
            query.strip(),
            keyframes=None,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            max_length=512,
        )
    out = emb.detach().float().cpu().unsqueeze(0)

    del emb, _tags, te, tokenizer, processor, holder
    _cleanup_cuda()
    return out


def _print_generation(label: str, text: str) -> None:
    print(f"\n[{label}]\n{text or '<empty>'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", help=".h3textprobe.pt from tools/h3_dit_text_probe.py")
    p.add_argument("--models-path", required=True)
    p.add_argument("--dit-path")
    p.add_argument("--stock-model", default=DEFAULT_STOCK_MODEL)
    p.add_argument("--h3-model", default=DEFAULT_H3_MODEL)
    p.add_argument("--suffix", default="\nCaption:")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument(
        "--h3-scale",
        choices=("none", "per-token", "global"),
        default="none",
        help="Optional scale adjustment for exact H3 conditioner injection; none is the clean control.",
    )
    p.add_argument(
        "--inverse-scale",
        choices=("none", "per-token", "global"),
        default="none",
        help="Optional scale adjustment for direct least-squares inverse injection.",
    )
    p.add_argument("--output")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    capture_path = Path(args.capture).expanduser().resolve()
    models_path = Path(args.models_path).expanduser().resolve()
    payload = _load_probe(capture_path)
    query = str(payload["query"])
    projected = payload["captures"]["real"]["projected"].float()

    dit_path = _resolve_dit(models_path, str(payload["partition"]), args.dit_path)
    weight, bias = _load_condition_proj(dit_path)

    print(f"[H3 calibrate] Recomputing exact H3 Qwen state for query: {query!r}")
    exact_h3 = _encode_exact_h3_query(models_path, args.h3_model, query)
    if exact_h3.shape[1] != projected.shape[1]:
        raise RuntimeError(
            f"Exact H3 query has {exact_h3.shape[1]} tokens but capture has {projected.shape[1]}"
        )

    # First verify that the checkpoint's condition_proj and recomputed H3 query
    # reproduce the state captured during the original DiT probe.
    with torch.inference_mode():
        reproj = exact_h3[0].float() @ weight.float().T + bias.float()
    projected_repro_rel = _rms(reproj - projected[0]) / max(_rms(projected[0]), 1e-20)
    projected_repro_cos = _cos(reproj, projected[0])
    print(
        f"[H3 calibrate] exact H3 -> condition_proj vs saved projected: "
        f"cos={projected_repro_cos:.9f} rel_rms={projected_repro_rel:.6e}"
    )

    direct_inv, direct_stats = _direct_inverse(projected, weight, bias, device)
    print(
        f"[H3 calibrate] direct inverse: recon_rel_rms={direct_stats['recon_relative_rms']:.6e} "
        f"rms={direct_stats['solution_rms']:.5f}; "
        f"vs exact H3 cos={_cos(direct_inv, exact_h3):.9f} "
        f"delta_rms={_rms(direct_inv - exact_h3):.6f}"
    )

    # condition_proj is no longer needed; free it before loading the 32B donor.
    del weight, bias, reproj
    _cleanup_cuda()

    print(f"[H3 calibrate] Loading stock Qwen donor: {args.stock_model}")
    stock, processor = _load_stock(args.stock_model)
    stock.model.visual = nn.Identity()
    _cleanup_cuda()
    _cuda_mem("after discarding unused donor vision tower")

    raw_inputs, query_len = _raw_inputs(processor, query, args.suffix)
    if query_len != exact_h3.shape[1]:
        raise RuntimeError(
            f"Stock tokenizer produced {query_len} query tokens; H3 has {exact_h3.shape[1]}"
        )
    raw_inputs = _move_inputs(raw_inputs, device)

    stock_h = _capture_layer_input(stock, raw_inputs, LOWER_LAYER_COUNT)
    stock_query = stock_h[:, :query_len]
    print("\n=== REPRESENTATION CALIBRATION ===")
    print(
        f"stock layer50: rms={_rms(stock_query):.6f}\n"
        f"exact H3 Qwen: rms={_rms(exact_h3):.6f} cos_to_stock={_cos(exact_h3, stock_query):.9f} "
        f"delta_rms={_rms(exact_h3-stock_query):.6f}\n"
        f"direct inverse: rms={_rms(direct_inv):.6f} cos_to_stock={_cos(direct_inv, stock_query):.9f} "
        f"cos_to_exact_h3={_cos(direct_inv, exact_h3):.9f}"
    )

    results = {
        "capture": str(capture_path),
        "query": query,
        "suffix": args.suffix,
        "projected_reprojection_cos": projected_repro_cos,
        "projected_reprojection_relative_rms": projected_repro_rel,
        "direct_inverse_stats": direct_stats,
        "representations": {
            "stock_rms": _rms(stock_query),
            "exact_h3_rms": _rms(exact_h3),
            "exact_h3_to_stock_cos": _cos(exact_h3, stock_query),
            "exact_h3_to_stock_delta_rms": _rms(exact_h3 - stock_query),
            "direct_inverse_to_exact_h3_cos": _cos(direct_inv, exact_h3),
            "direct_inverse_to_stock_cos": _cos(direct_inv, stock_query),
        },
        "generations": {},
    }

    print("\n=== GENERATION CONTROLS ===")

    baseline, _ = _generate(
        stock, processor, raw_inputs,
        replacement=None, replace_len=query_len,
        layer_idx=LOWER_LAYER_COUNT, max_new_tokens=args.max_new_tokens,
    )
    results["generations"]["stock_baseline"] = baseline
    _print_generation("stock baseline", baseline)

    # Hook positive control: replacing the query with the *same* hidden state we
    # just captured must be transparent.  If not, the hook/generation logic is wrong.
    stock_reinject, used = _generate(
        stock, processor, raw_inputs,
        replacement=stock_query, replace_len=query_len,
        layer_idx=LOWER_LAYER_COUNT, max_new_tokens=args.max_new_tokens,
    )
    if not used:
        raise RuntimeError("Stock self-injection hook never fired")
    results["generations"]["stock_self_injection"] = stock_reinject
    _print_generation("stock self-injection", stock_reinject)

    exact_for_injection = _match_scale(exact_h3, stock_query, args.h3_scale)
    exact_text, used = _generate(
        stock, processor, raw_inputs,
        replacement=exact_for_injection, replace_len=query_len,
        layer_idx=LOWER_LAYER_COUNT, max_new_tokens=args.max_new_tokens,
    )
    if not used:
        raise RuntimeError("Exact H3 injection hook never fired")
    results["generations"]["exact_h3_conditioner"] = exact_text
    _print_generation(f"exact H3 conditioner | scale={args.h3_scale}", exact_text)

    inv_for_injection = _match_scale(direct_inv, stock_query, args.inverse_scale)
    inv_text, used = _generate(
        stock, processor, raw_inputs,
        replacement=inv_for_injection, replace_len=query_len,
        layer_idx=LOWER_LAYER_COUNT, max_new_tokens=args.max_new_tokens,
    )
    if not used:
        raise RuntimeError("Direct inverse injection hook never fired")
    results["generations"]["direct_inverse"] = inv_text
    _print_generation(f"direct projected inverse | scale={args.inverse_scale}", inv_text)

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[H3 calibrate] Wrote {out}")

    print(
        "\n[H3 calibrate] Decision tree:\n"
        "  • stock self-injection != baseline -> fix hook before anything else.\n"
        "  • exact H3 conditioner is coherent but direct inverse is not -> inversion was the problem.\n"
        "  • exact H3 conditioner itself is incoherent -> investigate H3-vs-donor representation/context mismatch.\n"
        "  • both exact H3 and direct inverse are coherent -> bridge calibration passes; return to deep H3 states."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
