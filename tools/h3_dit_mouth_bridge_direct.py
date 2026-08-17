#!/usr/bin/env python3
"""Decode deep MiniMax-H3 text-row states with a calibrated direct Qwen bridge.

This is the follow-up to h3_dit_bridge_calibrate.py.  The calibration established
that:
  * the H3 Qwen hidden_states[50] boundary is compatible with the stock Qwen tail;
  * the layer-50 injection hook is transparent;
  * directly least-squares-inverting H3 condition_proj recovers the original
    Qwen state closely enough to preserve generation;
  * the earlier CG/normal-equation bridge was the failed component.

This probe therefore returns to the actual H3 DiT captures.  It solves all
requested real/zero/reversed H3 text-row states in ONE QR least-squares solve,
maps them from H3's 5376-d residual stream to the nearest 5120-d Qwen state,
RMS-matches them to the stock layer-50 query state, and lets stock Qwen layers
50..63 + norm + LM head generate continuations.

Early taps are included by default because they remain closer to the original
text-conditioning manifold while already containing media-dependent signal.
Late taps are included because the prior text-row probe showed especially large
real-vs-reversed effects there.

This still is not a trained caption decoder.  The important positive result
would be source-specific facts appearing preferentially for REAL media, with
ZERO/REVERSED acting as controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    _parse_csv,
    _raw_inputs,
    _resolve_dit,
    _rms,
)
from tools.h3_mouth_probe import (
    DEFAULT_STOCK_MODEL,
    LOWER_LAYER_COUNT,
    _cleanup_cuda,
    _cuda_mem,
    _load_stock,
)

DEFAULT_TAPS = "block_1,block_5,block_10,block_20,block_30,block_40,block_50"
DEFAULT_CONTROLS = "real,zero,reversed"


def _direct_inverse_many(
    states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Solve y = x W^T + b for states (N,L,5376) in one QR least-squares solve."""
    if states.ndim != 3 or states.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"Expected (N,L,{weight.shape[0]}) states, got {tuple(states.shape)}"
        )
    n, l, d = states.shape
    w = weight.to(device=device, dtype=torch.float32)
    b = bias.to(device=device, dtype=torch.float32)
    flat = states.reshape(n * l, d).to(device=device, dtype=torch.float32)
    rhs = (flat - b).T.contiguous()  # (5376, N*L)

    print(
        f"[H3 direct bridge] One least-squares solve: W={tuple(w.shape)}, "
        f"rhs={tuple(rhs.shape)} on {device}"
    )
    try:
        if device.type == "cuda":
            sol = torch.linalg.lstsq(w, rhs, driver="gels").solution
        else:
            sol = torch.linalg.lstsq(w, rhs).solution
    except Exception as exc:
        print(
            f"[H3 direct bridge] GPU solve failed ({type(exc).__name__}: {exc}); "
            "retrying on CPU"
        )
        _cleanup_cuda()
        wc = weight.float().contiguous()
        rhsc = (states.reshape(n * l, d).float() - bias.float()).T.contiguous()
        sol = torch.linalg.lstsq(wc, rhsc).solution
        sol = sol.to(device)
        w = weight.to(device=device, dtype=torch.float32)
        b = bias.to(device=device, dtype=torch.float32)
        flat = states.reshape(n * l, d).to(device=device, dtype=torch.float32)

    x = sol.T.reshape(n, l, weight.shape[1])
    recon = (x.reshape(n * l, -1) @ w.T + b).reshape(n, l, d)
    target = states.to(device=device, dtype=torch.float32)

    per_state = []
    for i in range(n):
        denom = max(_rms((target[i] - b).detach().cpu()), 1e-20)
        per_state.append(
            {
                "recon_relative_rms": _rms((recon[i] - target[i]).detach().cpu()) / denom,
                "solution_rms": _rms(x[i].detach().cpu()),
            }
        )

    out = x.detach().float().cpu()
    stats = {
        "count": n,
        "tokens_per_state": l,
        "per_state": per_state,
        "mean_recon_relative_rms": float(
            sum(s["recon_relative_rms"] for s in per_state) / max(1, len(per_state))
        ),
    }
    del w, b, flat, rhs, sol, x, recon, target
    _cleanup_cuda()
    return out, stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", help=".h3textprobe.pt from tools/h3_dit_text_probe.py")
    p.add_argument("--models-path", required=True)
    p.add_argument("--dit-path")
    p.add_argument("--stock-model", default=DEFAULT_STOCK_MODEL)
    p.add_argument("--taps", default=DEFAULT_TAPS)
    p.add_argument("--controls", default=DEFAULT_CONTROLS)
    p.add_argument("--suffix", default="\nCaption:")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument(
        "--scale",
        choices=("per-token", "global", "none"),
        default="per-token",
        help="RMS normalization before injecting deep H3-derived states into Qwen layer 50.",
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
    captures = payload["captures"]
    taps = _parse_csv(args.taps)
    controls = _parse_csv(args.controls)

    for control in controls:
        if control not in captures:
            raise KeyError(f"Missing control {control!r}; available={list(captures)}")
    for tap in taps:
        if tap not in captures[controls[0]]:
            raise KeyError(
                f"Missing tap {tap!r}; available={list(captures[controls[0]])}"
            )

    # Build one stacked solve in stable tap/control order.
    labels: List[Tuple[str, str]] = []
    chunks = []
    for tap in taps:
        for control in controls:
            labels.append((tap, control))
            chunks.append(captures[control][tap].float())
    stacked = torch.cat(chunks, dim=0)

    dit_path = _resolve_dit(models_path, str(payload["partition"]), args.dit_path)
    weight, bias = _load_condition_proj(dit_path)
    solved, solve_stats = _direct_inverse_many(stacked, weight, bias, device)

    print(
        f"[H3 direct bridge] Mean nearest-subspace reconstruction relative RMS: "
        f"{solve_stats['mean_recon_relative_rms']:.5f}"
    )
    for i, (tap, control) in enumerate(labels):
        s = solve_stats["per_state"][i]
        print(
            f"  {tap:9s} {control:8s} recon_rel={s['recon_relative_rms']:.5f} "
            f"inverse_rms={s['solution_rms']:.3f}"
        )

    del weight, bias, stacked, chunks
    _cleanup_cuda()

    print(f"[H3 direct bridge] Loading Qwen donor: {args.stock_model}")
    stock, processor = _load_stock(args.stock_model)
    stock.model.visual = nn.Identity()
    _cleanup_cuda()
    _cuda_mem("after discarding unused donor vision tower")

    raw_inputs, query_len = _raw_inputs(processor, query, args.suffix)
    if query_len != solved.shape[1]:
        raise RuntimeError(
            f"Capture has {solved.shape[1]} query rows but stock tokenizer produced {query_len}"
        )
    raw_inputs = _move_inputs(raw_inputs, device)

    stock_h = _capture_layer_input(stock, raw_inputs, LOWER_LAYER_COUNT)
    stock_query = stock_h[:, :query_len]
    print(
        f"[H3 direct bridge] stock layer50 query RMS={_rms(stock_query):.5f}; "
        f"scale={args.scale}"
    )

    print("\n=== DIRECT H3 -> QWEN MOUTH DECODING ===")
    print(f"Query:  {query!r}")
    print(f"Suffix: {args.suffix!r}")

    results: Dict = {
        "capture": str(capture_path),
        "query": query,
        "suffix": args.suffix,
        "scale": args.scale,
        "solve_stats": solve_stats,
        "generations": {},
    }

    baseline, _ = _generate(
        stock,
        processor,
        raw_inputs,
        replacement=None,
        replace_len=query_len,
        layer_idx=LOWER_LAYER_COUNT,
        max_new_tokens=args.max_new_tokens,
    )
    results["generations"]["stock_baseline"] = baseline
    print(f"\n[stock baseline]\n{baseline or '<empty>'}")

    # A nearby projection capture is useful here as an end-to-end sanity check,
    # but the calibrated script already proved exact inversion.  Solve it with
    # the same direct machinery only when it is available.
    if "projected" in captures["real"]:
        dit_path2 = _resolve_dit(models_path, str(payload["partition"]), args.dit_path)
        weight2, bias2 = _load_condition_proj(dit_path2)
        projected_inv, _ = _direct_inverse_many(
            captures["real"]["projected"].float(), weight2, bias2, device
        )
        del weight2, bias2
        _cleanup_cuda()
        projected_text, used = _generate(
            stock,
            processor,
            raw_inputs,
            replacement=_match_scale(projected_inv, stock_query, "none"),
            replace_len=query_len,
            layer_idx=LOWER_LAYER_COUNT,
            max_new_tokens=args.max_new_tokens,
        )
        if not used:
            raise RuntimeError("Projected direct-inverse hook never fired")
        results["generations"]["projected_direct_control"] = projected_text
        print(f"\n[projected direct-inverse control]\n{projected_text or '<empty>'}")

    for i, (tap, control) in enumerate(labels):
        bridge = solved[i : i + 1]
        bridge = _match_scale(bridge, stock_query, args.scale)
        text, used = _generate(
            stock,
            processor,
            raw_inputs,
            replacement=bridge,
            replace_len=query_len,
            layer_idx=LOWER_LAYER_COUNT,
            max_new_tokens=args.max_new_tokens,
        )
        if not used:
            raise RuntimeError(f"Injection hook never fired for {tap}/{control}")
        key = f"{tap}/{control}"
        results["generations"][key] = {
            "text": text,
            "injected_rms": _rms(bridge),
            "recon_relative_rms": solve_stats["per_state"][i]["recon_relative_rms"],
        }
        print(f"\n[{tap} | {control}]\n{text or '<empty>'}")

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[H3 direct bridge] Wrote {out}")

    print(
        "\n[H3 direct bridge] Read this conservatively:\n"
        "  • projected direct-inverse should remain coherent; calibration already showed this path can work.\n"
        "  • compare REAL against ZERO and REVERSED at the SAME tap.\n"
        "  • source-specific beach/interview/people/clothing/dialogue/audio facts preferentially in REAL are signal.\n"
        "  • fluent but unrelated text, or identical collapse across controls, means the linear nearest-subspace bridge is insufficient.\n"
        "  • early taps may decode better even if late taps contain stronger media dependence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
