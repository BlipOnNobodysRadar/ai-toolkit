#!/usr/bin/env python3
"""Try to decode MiniMax-H3 DiT text-row states with a Qwen3-VL language mouth.

Consumes a .h3textprobe.pt produced by tools/h3_dit_text_probe.py.

The experiment is deliberately simple:
  1. load H3's learned condition_proj (Qwen 5120 -> H3 5376);
  2. approximately invert that affine map with ridge-regularized CG;
  3. map captured H3 text-row states back into 5120-d Qwen space;
  4. load a complete stock Qwen3-VL-32B-Instruct donor;
  5. during the prefill only, replace the hidden states entering decoder layer 50
     for the original query-token positions with the bridged H3 states;
  6. let stock Qwen layers 50..63 + final norm + LM head generate a continuation.

Why layer 50? MiniMax-H3 consumes Qwen hidden_states[50], i.e. the unnormalized
output after decoder layer 49. Thus the input to stock decoder layer 50 is the
natural place to inject a recovered H3-derived state.

Controls matter more than pretty prose. The script always runs:
  - ordinary stock-Qwen continuation;
  - a "projected" round-trip control (invert the exact condition_proj output),
    which should remain close to ordinary Qwen if the bridge/hook is correct;
then selected H3 depths for real / zero / time-reversed media.

A deep H3 state is not expected to lie exactly in condition_proj's column space.
The inverse therefore returns the nearest ridge-regularized Qwen-space vector.
By default each injected token is RMS-matched to the stock layer-50 query state
before generation; Qwen's upper stack expects that scale, while H3 residual
states grow dramatically with depth.

This does NOT train an inverse adapter. It is the cheapest possible test of
whether H3's media-dependent internal text states already contain directions a
Qwen language tail can interpret.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.h3_mouth_probe import DEFAULT_STOCK_MODEL, LOWER_LAYER_COUNT, _cleanup_cuda, _cuda_mem, _load_stock


DEFAULT_TAPS = "block_20,block_30,block_40,block_50"
DEFAULT_CONTROLS = "real,zero,reversed"


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float().square())).item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return float("nan")
    return float((torch.dot(a, b) / denom).item())


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _resolve_dit(models_path: Path, partition: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    filename = f"minimax_h3_{partition}_int8_convrot.safetensors"
    path = models_path / "diffusion_models" / filename
    if path.is_file():
        return path
    # Friendly fallback for a custom subfolder under diffusion_models.
    matches = sorted((models_path / "diffusion_models").rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {filename} under {models_path / 'diffusion_models'}; pass --dit-path"
    )


def _load_condition_proj(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"[H3 bridge] Loading condition_proj from {path}")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        weight_keys = [k for k in keys if k.endswith("condition_proj.weight")]
        bias_keys = [k for k in keys if k.endswith("condition_proj.bias")]
        if len(weight_keys) != 1:
            raise KeyError(
                f"Expected one condition_proj.weight in checkpoint, found {weight_keys[:8]}"
            )
        weight = f.get_tensor(weight_keys[0]).float().contiguous()
        if bias_keys:
            bias = f.get_tensor(bias_keys[0]).float().contiguous()
        else:
            bias = torch.zeros(weight.shape[0], dtype=torch.float32)
    if tuple(weight.shape) != (5376, 5120):
        raise ValueError(f"Unexpected condition_proj weight shape {tuple(weight.shape)}")
    print(
        f"[H3 bridge] condition_proj W={tuple(weight.shape)} b={tuple(bias.shape)} "
        f"({weight.numel() * 4 / 1024**2:.1f} MiB fp32)"
    )
    return weight, bias


def _cg_inverse(
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    iters: int,
    ridge_ratio: float,
    tol: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Solve min_x ||x W^T + b - y||^2 + lambda ||x||^2 for many rows.

    y may be (1,L,5376) or (N,L,5376). Rows are independent right-hand sides,
    solved together with preconditioned conjugate gradient on the normal equations.
    """
    orig_shape = y.shape
    if y.ndim == 2:
        y2 = y
    elif y.ndim == 3:
        y2 = y.reshape(-1, y.shape[-1])
    else:
        raise ValueError(f"Expected 2D/3D H3 states, got {tuple(y.shape)}")

    w = weight.to(device=device, dtype=torch.float32)
    b = bias.to(device=device, dtype=torch.float32)
    yc = y2.to(device=device, dtype=torch.float32) - b

    diag = w.square().sum(dim=0)
    lam = float(ridge_ratio) * float(diag.mean().item())
    precond = (diag + lam).clamp_min(1e-12)
    rhs = yc @ w

    def apply_a(x: torch.Tensor) -> torch.Tensor:
        return (x @ w.T) @ w + lam * x

    # Jacobi starting point is much closer than zero if W is nearly orthogonal.
    x = rhs / precond
    r = rhs - apply_a(x)
    z = r / precond
    p = z.clone()
    rz = (r * z).sum(dim=1, keepdim=True)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=1, keepdim=True).clamp_min(1e-20)
    used = 0

    for i in range(max(1, iters)):
        ap = apply_a(p)
        denom = (p * ap).sum(dim=1, keepdim=True).clamp_min(1e-30)
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * ap
        used = i + 1
        rel = torch.linalg.vector_norm(r, dim=1, keepdim=True) / rhs_norm
        if float(rel.max().item()) <= tol:
            break
        z = r / precond
        rz_new = (r * z).sum(dim=1, keepdim=True)
        beta = rz_new / rz.clamp_min(1e-30)
        p = z + beta * p
        rz = rz_new

    recon = x @ w.T + b
    target = y2.to(device=device, dtype=torch.float32)
    err = torch.sqrt(torch.mean((recon - target).square()))
    target_centered_rms = torch.sqrt(torch.mean((target - b).square())).clamp_min(1e-20)
    rel_recon = float((err / target_centered_rms).item())
    normal_rel = float(
        (
            torch.linalg.vector_norm(r, dim=1)
            / torch.linalg.vector_norm(rhs, dim=1).clamp_min(1e-20)
        ).max().item()
    )
    stats = {
        "iterations": used,
        "ridge_lambda": lam,
        "normal_residual_max": normal_rel,
        "relative_reconstruction_rms": rel_recon,
        "input_rms": _rms(target),
        "solution_rms": _rms(x),
    }
    out_shape = orig_shape[:-1] + (weight.shape[1],)
    return x.reshape(out_shape).detach().cpu(), stats


def _match_scale(x: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    x = x.float()
    target = target.float()
    if mode == "global":
        return x * (_rms(target) / max(_rms(x), 1e-20))
    if mode == "per-token":
        xr = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True)).clamp_min(1e-20)
        tr = torch.sqrt(torch.mean(target.square(), dim=-1, keepdim=True)).clamp_min(1e-20)
        return x * (tr / xr)
    raise ValueError(mode)


def _raw_inputs(processor, query: str, suffix: str) -> tuple[dict, int]:
    tokenizer = processor.tokenizer
    q_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
    s_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    ids = q_ids + s_ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    # Qwen3-VL uses this to mark multimodal placeholders. Text-only is all text,
    # but supplying the processor-generated value keeps us on its expected path.
    creator = getattr(processor, "create_mm_token_type_ids", None)
    if creator is not None:
        try:
            mm = creator([ids])
            inputs["mm_token_type_ids"] = torch.tensor(mm, dtype=torch.long)
        except Exception:
            pass
    return inputs, len(q_ids)


def _move_inputs(inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _capture_layer_input(model, inputs: dict, layer_idx: int) -> torch.Tensor:
    layer = model.model.language_model.layers[layer_idx]
    box = {}

    def hook(_module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is not None and "h" not in box:
            box["h"] = h.detach().float().cpu()

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if "h" not in box:
        raise RuntimeError(f"Layer-{layer_idx} pre-hook never observed hidden_states")
    return box["h"]


def _generate(
    model,
    processor,
    inputs: dict,
    *,
    replacement: torch.Tensor | None,
    replace_len: int,
    layer_idx: int,
    max_new_tokens: int,
) -> tuple[str, bool]:
    layer = model.model.language_model.layers[layer_idx]
    used = {"value": False}
    handle = None

    if replacement is not None:
        replacement = replacement.detach().cpu()

        def hook(_module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if h is None or used["value"] or h.shape[1] < replace_len:
                return None
            if replacement.shape[1] != replace_len or replacement.shape[-1] != h.shape[-1]:
                raise RuntimeError(
                    f"Replacement {tuple(replacement.shape)} incompatible with layer state {tuple(h.shape)}"
                )
            new_h = h.clone()
            new_h[:, :replace_len] = replacement.to(device=h.device, dtype=h.dtype)
            used["value"] = True
            if args:
                return (new_h,) + tuple(args[1:]), kwargs
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = new_h
            return args, new_kwargs

        handle = layer.register_forward_pre_hook(hook, with_kwargs=True)

    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    finally:
        if handle is not None:
            handle.remove()

    prompt_len = int(inputs["input_ids"].shape[-1])
    text = processor.decode(
        generated[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return text, bool(used["value"] if replacement is not None else True)


def _load_probe(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("query", "partition", "captures")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Probe file is missing fields: {missing}")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", help=".h3textprobe.pt from tools/h3_dit_text_probe.py")
    p.add_argument("--models-path", required=True, help="ComfyUI models root")
    p.add_argument("--dit-path", help="Explicit MiniMax-H3 transformer safetensors")
    p.add_argument("--stock-model", default=DEFAULT_STOCK_MODEL)
    p.add_argument("--taps", default=DEFAULT_TAPS)
    p.add_argument("--controls", default=DEFAULT_CONTROLS)
    p.add_argument(
        "--suffix",
        default="\nCaption:",
        help="Raw text appended after the captured query before Qwen continues. Query tokens are tokenized separately so their IDs exactly match the H3 probe.",
    )
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--solver-iters", type=int, default=48)
    p.add_argument("--solver-tol", type=float, default=1e-5)
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-5,
        help="Ridge lambda as a fraction of mean diag(W^T W).",
    )
    p.add_argument(
        "--scale",
        choices=("per-token", "global", "none"),
        default="per-token",
        help="Scale bridged states to stock Qwen layer-50 RMS before injection.",
    )
    p.add_argument("--output", help="Optional JSON result path")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    capture_path = Path(args.capture).expanduser().resolve()
    if not capture_path.is_file():
        raise FileNotFoundError(capture_path)
    models_path = Path(args.models_path).expanduser().resolve()
    payload = _load_probe(capture_path)
    query = str(payload["query"])
    captures = payload["captures"]
    taps = _parse_csv(args.taps)
    controls = _parse_csv(args.controls)

    for control in controls:
        if control not in captures:
            raise KeyError(f"Control {control!r} not present; available={list(captures)}")
    for tap in taps:
        if tap not in captures[controls[0]]:
            raise KeyError(
                f"Tap {tap!r} not present; available={list(captures[controls[0]])}"
            )

    dit_path = _resolve_dit(models_path, str(payload["partition"]), args.dit_path)
    weight, bias = _load_condition_proj(dit_path)

    print(f"[H3 bridge] Inverting exact projected-state control ({captures['real']['projected'].shape[1]} tokens)")
    x_projected, projected_stats = _cg_inverse(
        captures["real"]["projected"],
        weight,
        bias,
        iters=max(args.solver_iters, 64),
        ridge_ratio=args.ridge,
        tol=args.solver_tol,
        device=device,
    )
    print(
        "[H3 bridge] projected inverse: "
        f"iters={projected_stats['iterations']} "
        f"normal_res={projected_stats['normal_residual_max']:.3e} "
        f"recon_rel_rms={projected_stats['relative_reconstruction_rms']:.3e} "
        f"qwen_rms={projected_stats['solution_rms']:.5f}"
    )

    solved: Dict[str, Dict[str, Tuple[torch.Tensor, dict]]] = {}
    for tap in taps:
        print(f"[H3 bridge] Inverting {tap}: {', '.join(controls)}")
        stacked = torch.cat([captures[c][tap] for c in controls], dim=0)
        x_all, stats = _cg_inverse(
            stacked,
            weight,
            bias,
            iters=args.solver_iters,
            ridge_ratio=args.ridge,
            tol=args.solver_tol,
            device=device,
        )
        # stacked shape is (num_controls, L, 5376), so inverse preserves it.
        solved[tap] = {}
        for i, control in enumerate(controls):
            state = x_all[i : i + 1]
            # Reconstruction stat above is aggregate; retain it as a useful rough diagnostic.
            solved[tap][control] = (state, dict(stats))
        print(
            f"[H3 bridge]   nearest condition_proj-subspace recon_rel_rms="
            f"{stats['relative_reconstruction_rms']:.4f}; inverse_rms={stats['solution_rms']:.4f}"
        )

    # Free the 100+ MiB fp32 projection before loading the 32B donor.
    del weight, bias
    _cleanup_cuda()

    print(f"[H3 bridge] Loading Qwen mouth donor: {args.stock_model}")
    stock, processor = _load_stock(args.stock_model)
    # No visual inputs are used here. Free the donor vision tower for headroom while
    # retaining all 64 language layers; layers 0..49 are needed for generated tokens.
    stock.model.visual = nn.Identity()
    _cleanup_cuda()
    _cuda_mem("after discarding unused donor vision tower")

    raw_inputs, query_len = _raw_inputs(processor, query, args.suffix)
    if query_len != int(x_projected.shape[1]):
        raise RuntimeError(
            f"Captured query has {x_projected.shape[1]} H3 rows but stock tokenizer produced {query_len} tokens. "
            "Tokenizer mismatch; do not force the bridge."
        )
    raw_inputs = _move_inputs(raw_inputs, device)

    print(f"[H3 bridge] Capturing stock Qwen hidden state entering decoder layer {LOWER_LAYER_COUNT}")
    stock_h = _capture_layer_input(stock, raw_inputs, LOWER_LAYER_COUNT)
    stock_query_h = stock_h[:, :query_len]
    print(
        f"[H3 bridge] stock layer50 query RMS={_rms(stock_query_h):.5f}; "
        f"projected-inverse RMS={_rms(x_projected):.5f}; "
        f"cos={_cos(stock_query_h, x_projected):.8f}; "
        f"RMS delta={_rms(stock_query_h - x_projected):.6f}"
    )

    print("\n=== QWEN MOUTH DECODING ===")
    print(f"Query:  {query!r}")
    print(f"Suffix: {args.suffix!r}")

    results = {
        "capture": str(capture_path),
        "query": query,
        "suffix": args.suffix,
        "scale": args.scale,
        "projected_inverse_stats": projected_stats,
        "stock_vs_projected_cos": _cos(stock_query_h, x_projected),
        "stock_vs_projected_rms_delta": _rms(stock_query_h - x_projected),
        "generations": {},
    }

    text, _ = _generate(
        stock,
        processor,
        raw_inputs,
        replacement=None,
        replace_len=query_len,
        layer_idx=LOWER_LAYER_COUNT,
        max_new_tokens=args.max_new_tokens,
    )
    results["generations"]["stock_baseline"] = text
    print(f"\n[stock baseline]\n{text or '<empty>'}")

    projected_for_injection = _match_scale(x_projected, stock_query_h, args.scale)
    text, used = _generate(
        stock,
        processor,
        raw_inputs,
        replacement=projected_for_injection,
        replace_len=query_len,
        layer_idx=LOWER_LAYER_COUNT,
        max_new_tokens=args.max_new_tokens,
    )
    if not used:
        raise RuntimeError("Projected bridge hook never fired during generation")
    results["generations"]["projected_roundtrip"] = text
    print(f"\n[projected round-trip control]\n{text or '<empty>'}")

    for tap in taps:
        for control in controls:
            bridge, inv_stats = solved[tap][control]
            bridge = _match_scale(bridge, stock_query_h, args.scale)
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
                raise RuntimeError(f"Bridge hook never fired for {tap}/{control}")
            key = f"{tap}/{control}"
            results["generations"][key] = {
                "text": text,
                "inverse_stats": inv_stats,
                "injected_rms": _rms(bridge),
            }
            print(f"\n[{tap} | {control}]\n{text or '<empty>'}")

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[H3 bridge] Wrote {out}")

    print(
        "\n[H3 bridge] Interpretation gate:\n"
        "  1. projected round-trip should stay coherent and preferably resemble stock baseline; otherwise the bridge/hook is suspect.\n"
        "  2. real/zero/reversed outputs should differ systematically if the H3 media signal survives the projection into Qwen space.\n"
        "  3. source-specific facts appearing preferentially in REAL are the interesting result. Generic fluency alone proves nothing.\n"
        "  4. If the round-trip works but deep states do not decode, the next step is a learned 5376->5120 adapter/query bridge rather than declaring H3 has no caption signal."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
