#!/usr/bin/env python3
"""Single-clip overfit test for a learned MiniMax-H3 -> Qwen caption bridge.

Purpose: the calibrated direct inverse proves we can map the original H3 text-conditioning
boundary back into Qwen space, but deep H3 DiT text-row states collapse when injected
analytically.  This probe asks the next, narrower question: can a *tiny learned residual
adapter* turn one real H3 media-conditioned text state into a useful Qwen layer-50 prefix?

This is intentionally NOT a general captioner.  It trains on one capture/one target caption.
If it cannot overfit one clip, the bridge architecture is probably wrong.  If it can, the
next step is to train the same adapter across many clips/captions and evaluate held-out media.

The adapter is low-rank and bias-free.  It starts from the calibrated direct least-squares
inverse of H3's condition_proj, then learns only a residual delta from the raw H3 state.
The Qwen3-VL donor is frozen; gradients flow only through layers 50..63 into the adapter.
After training we decode REAL, ZERO, and REVERSED controls with the same adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.h3_dit_mouth_bridge import (
    _capture_layer_input,
    _generate,
    _load_condition_proj,
    _load_probe,
    _move_inputs,
    _resolve_dit,
    _rms,
)
from tools.h3_dit_mouth_bridge_direct import _direct_inverse_many
from tools.h3_mouth_probe import (
    DEFAULT_STOCK_MODEL,
    LOWER_LAYER_COUNT,
    _cleanup_cuda,
    _cuda_mem,
    _load_stock,
)


DEFAULT_TARGET = (
    "A young boy interviews three women on a sunny beach boardwalk about their workout routines. "
    "The first blonde woman in orange says, 'Skating. Falling mostly.' The second woman in a white "
    "sports bra and black leggings says, 'Planking while I doom scroll.' The third woman in lime "
    "green says, 'Beach volleyball. Once.' There is no music."
)


def _per_token_match_scale(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = x.float()
    target = target.float()
    xr = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True)).clamp_min(1e-20)
    tr = torch.sqrt(torch.mean(target.square(), dim=-1, keepdim=True)).clamp_min(1e-20)
    return x * (tr / xr)


class ResidualBridge(nn.Module):
    """Low-rank residual on top of the calibrated direct inverse.

    base_qwen is sample-specific (the nearest condition_proj inverse).  The learned path sees
    the raw H3 state after per-token RMS normalization and predicts a correction in Qwen space.
    No bias is used so a constant output is slightly harder to learn from a single example.
    """

    def __init__(self, rank: int):
        super().__init__()
        self.down = nn.Linear(5376, rank, bias=False)
        self.up = nn.Linear(rank, 5120, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, h3_state: torch.Tensor, base_qwen: torch.Tensor) -> torch.Tensor:
        h = h3_state.float()
        hr = torch.sqrt(torch.mean(h.square(), dim=-1, keepdim=True)).clamp_min(1e-20)
        h = h / hr
        delta = self.up(F.silu(self.down(h)))
        return base_qwen.float() + delta


def _tokenize_full(processor, query: str, suffix: str, target: str):
    tok = processor.tokenizer
    q_ids = tok(query, add_special_tokens=False)["input_ids"]
    s_ids = tok(suffix, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None:
        t_ids = t_ids + [tok.eos_token_id]
    ids = q_ids + s_ids + t_ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    creator = getattr(processor, "create_mm_token_type_ids", None)
    if creator is not None:
        try:
            out["mm_token_type_ids"] = torch.tensor(creator([ids]), dtype=torch.long)
        except Exception:
            pass
    return out, len(q_ids), len(q_ids) + len(s_ids), len(t_ids)


def _tokenize_prefill(processor, query: str, suffix: str):
    tok = processor.tokenizer
    q_ids = tok(query, add_special_tokens=False)["input_ids"]
    s_ids = tok(suffix, add_special_tokens=False)["input_ids"]
    ids = q_ids + s_ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    out = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    creator = getattr(processor, "create_mm_token_type_ids", None)
    if creator is not None:
        try:
            out["mm_token_type_ids"] = torch.tensor(creator([ids]), dtype=torch.long)
        except Exception:
            pass
    return out, len(q_ids)


def _train_step_forward(model, inputs: dict, replacement: torch.Tensor, replace_len: int):
    layer = model.model.language_model.layers[LOWER_LAYER_COUNT]
    used = {"value": False}

    def hook(_module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None or used["value"] or h.shape[1] < replace_len:
            return None
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
        outputs = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if not used["value"]:
        raise RuntimeError("Training bridge hook never fired")
    return outputs


def _generate_with_adapter(
    *,
    model,
    processor,
    prefill_inputs: dict,
    query_len: int,
    adapter: ResidualBridge,
    h3_state: torch.Tensor,
    base_state: torch.Tensor,
    max_new_tokens: int,
) -> str:
    with torch.no_grad():
        replacement = adapter(h3_state, base_state)
    text, used = _generate(
        model,
        processor,
        prefill_inputs,
        replacement=replacement,
        replace_len=query_len,
        layer_idx=LOWER_LAYER_COUNT,
        max_new_tokens=max_new_tokens,
    )
    if not used:
        raise RuntimeError("Generation bridge hook never fired")
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", help=".h3textprobe.pt from tools/h3_dit_text_probe.py")
    p.add_argument("--models-path", required=True)
    p.add_argument("--dit-path")
    p.add_argument("--stock-model", default=DEFAULT_STOCK_MODEL)
    p.add_argument("--tap", default="block_30")
    p.add_argument("--suffix", default="\nCaption:")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--target-file")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--early-stop-loss", type=float, default=0.03)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output", help="JSON result path")
    p.add_argument("--adapter-output", help="Optional .pt adapter checkpoint path")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rank < 1 or args.steps < 1:
        p.error("--rank and --steps must be positive")

    device = torch.device("cuda:0")
    capture_path = Path(args.capture).expanduser().resolve()
    models_path = Path(args.models_path).expanduser().resolve()
    payload = _load_probe(capture_path)
    captures = payload["captures"]
    query = str(payload["query"])
    if args.tap not in captures["real"]:
        raise KeyError(f"Tap {args.tap!r} unavailable; choices={list(captures['real'])}")

    target = args.target
    if args.target_file:
        target = Path(args.target_file).expanduser().read_text(encoding="utf-8").strip()
    if not target:
        raise ValueError("Target caption is empty")

    # Solve all three controls before loading the 32B donor.
    dit_path = _resolve_dit(models_path, str(payload["partition"]), args.dit_path)
    weight, bias = _load_condition_proj(dit_path)
    controls = ["real", "zero", "reversed"]
    states = torch.cat([captures[c][args.tap].float() for c in controls], dim=0)
    print(f"[H3 overfit] Direct-inverting {args.tap} for {controls}")
    base_all, inv_stats = _direct_inverse_many(states, weight, bias, device)
    del weight, bias
    _cleanup_cuda()

    print(f"[H3 overfit] Loading frozen Qwen donor: {args.stock_model}")
    stock, processor = _load_stock(args.stock_model)
    stock.model.visual = nn.Identity()
    stock.eval()
    for param in stock.parameters():
        param.requires_grad_(False)
    _cleanup_cuda()
    _cuda_mem("after discarding unused donor vision tower")

    train_inputs, query_len, target_start, target_len = _tokenize_full(
        processor, query, args.suffix, target
    )
    prefill_inputs, query_len2 = _tokenize_prefill(processor, query, args.suffix)
    if query_len != query_len2 or query_len != states.shape[1]:
        raise RuntimeError(
            f"Query-token mismatch: train={query_len}, prefill={query_len2}, H3={states.shape[1]}"
        )
    train_inputs = _move_inputs(train_inputs, device)
    prefill_inputs = _move_inputs(prefill_inputs, device)

    # Capture the stock layer-50 scale for the same query/suffix context.
    stock_h = _capture_layer_input(stock, prefill_inputs, LOWER_LAYER_COUNT)
    stock_query = stock_h[:, :query_len].to(device)

    bases = {}
    h3_states = {}
    for i, c in enumerate(controls):
        h3_states[c] = states[i : i + 1].to(device)
        bases[c] = _per_token_match_scale(base_all[i : i + 1].to(device), stock_query)

    adapter = ResidualBridge(args.rank).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print("\n=== SINGLE-CLIP LEARNED BRIDGE OVERFIT ===")
    print(f"tap={args.tap} rank={args.rank} steps={args.steps} lr={args.lr:g}")
    print(f"query tokens={query_len}; target tokens={target_len}")
    print(f"target: {target}")

    losses = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        replacement = adapter(h3_states["real"], bases["real"])
        outputs = _train_step_forward(stock, train_inputs, replacement, query_len)

        # Causal LM: logits at position k predict token k+1.  Score only target tokens.
        logits = outputs.logits[:, target_start - 1 : -1, :].float()
        labels = train_inputs["input_ids"][:, target_start:]
        if logits.shape[1] != labels.shape[1]:
            raise RuntimeError(
                f"Loss alignment mismatch logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
            )
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
        optimizer.step()

        lv = float(loss.detach().item())
        losses.append(lv)
        if step == 1 or step % args.print_every == 0 or lv <= args.early_stop_loss:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = float((pred == labels).float().mean().item())
                delta_rms = _rms((replacement.detach() - bases["real"]).cpu())
            print(f"step {step:4d}  loss={lv:.5f}  teacher_acc={acc:.3f}  delta_rms={delta_rms:.4f}")
        del outputs, logits, labels, loss, replacement
        if lv <= args.early_stop_loss:
            print(f"[H3 overfit] Early-stop threshold reached at step {step}")
            break

    adapter.eval()
    generations = {}
    print("\n=== POST-TRAIN GENERATION ===")
    for c in controls:
        text = _generate_with_adapter(
            model=stock,
            processor=processor,
            prefill_inputs=prefill_inputs,
            query_len=query_len,
            adapter=adapter,
            h3_state=h3_states[c],
            base_state=bases[c],
            max_new_tokens=args.max_new_tokens,
        )
        generations[c] = text
        print(f"\n[{args.tap} | {c}]\n{text or '<empty>'}")

    result = {
        "capture": str(capture_path),
        "tap": args.tap,
        "rank": args.rank,
        "steps_requested": args.steps,
        "steps_run": len(losses),
        "learning_rate": args.lr,
        "target": target,
        "losses": losses,
        "inverse_stats": inv_stats,
        "generations": generations,
        "interpretation": (
            "This is an overfit gate only. Success means a small learned readout can make the frozen "
            "Qwen tail consume H3 media-conditioned states. It does not demonstrate generalization. "
            "If REAL learns the target, the next experiment should train across many clips and test held-out media."
        ),
    }

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[H3 overfit] Wrote {out}")

    if args.adapter_output:
        out = Path(args.adapter_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "tap": args.tap,
                "rank": args.rank,
                "query": query,
                "suffix": args.suffix,
                "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            },
            out,
        )
        print(f"[H3 overfit] Wrote adapter {out}")

    print(
        "\n[H3 overfit] Read this as a gate, not a victory lap:\n"
        "  • failure to drive the one-clip loss down -> current query-prefix bridge is too weak/wrong.\n"
        "  • REAL learns the caption -> architecture is trainable; move to many clips + held-out evaluation.\n"
        "  • ZERO/REVERSED also emit the memorized caption -> expected possible one-example memorization; only multi-clip held-out tests can establish media decoding.\n"
        "  • ZERO/REVERSED differ while REAL alone matches details -> unusually encouraging, but still validate on unseen clips."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
