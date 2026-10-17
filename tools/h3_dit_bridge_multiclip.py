#!/usr/bin/env python3
"""Train one shared H3-DiT -> Qwen caption bridge across multiple clips.

Input is batch_index.json from h3_dit_text_probe_batch.py.  The script loads one
REAL H3 text-row state per clip, directly inverts condition_proj as the calibrated
base, and trains the same small ResidualBridge used by the successful single-clip
overfit.  Qwen3-VL stays frozen.  Held-out manifest indices are never used for
optimization and are generated only after training.

This is the first actual generalization gate.  A held-out clip producing semantic
facts specific to itself is evidence that the learned bridge is reading H3's
media-conditioned representation rather than merely memorizing a fixed caption.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.h3_dit_bridge_overfit import (
    ResidualBridge,
    _generate_with_adapter,
    _per_token_match_scale,
    _tokenize_full,
    _tokenize_prefill,
    _train_step_forward,
)
from tools.h3_dit_mouth_bridge import (
    _capture_layer_input,
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


def _csv_ints(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def _load_index(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("entries"), list) or not data["entries"]:
        raise ValueError(f"No entries in {path}")
    return data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("index", help="batch_index.json from h3_dit_text_probe_batch.py")
    p.add_argument("--models-path", required=True)
    p.add_argument("--dit-path")
    p.add_argument("--stock-model", default=DEFAULT_STOCK_MODEL)
    p.add_argument("--tap", default="block_30")
    p.add_argument("--holdout", default="9,10", help="Manifest indices excluded from training")
    p.add_argument("--suffix", default="\nCaption:")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--early-stop-loss", type=float, default=0.03)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output")
    p.add_argument("--adapter-output")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rank < 1 or args.epochs < 1:
        p.error("--rank and --epochs must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    index_path = Path(args.index).expanduser().resolve()
    models_path = Path(args.models_path).expanduser().resolve()
    index = _load_index(index_path)
    holdout_ids = _csv_ints(args.holdout)

    samples = []
    query = None
    partition = str(index.get("partition") or "fl2va_pruned")
    for ordinal, entry in enumerate(index["entries"], 1):
        idx = int(entry.get("index", ordinal))
        capture_path = Path(entry["capture"]).expanduser().resolve()
        payload = _load_probe(capture_path)
        this_query = str(payload["query"])
        if query is None:
            query = this_query
        elif this_query != query:
            raise RuntimeError(f"Query mismatch in {capture_path}: {this_query!r} != {query!r}")
        captures = payload["captures"]
        if "real" not in captures:
            raise KeyError(f"{capture_path} has no REAL capture; controls={list(captures)}")
        if args.tap not in captures["real"]:
            raise KeyError(f"{capture_path} lacks {args.tap}; taps={list(captures['real'])}")
        target = str(entry.get("target") or payload.get("target") or "").strip()
        if not target:
            raise ValueError(f"No target caption for manifest index {idx}")
        samples.append(
            {
                "index": idx,
                "file": entry.get("file", Path(payload["video"]).name),
                "video": payload["video"],
                "capture": str(capture_path),
                "target": target,
                "h3": captures["real"][args.tap].float(),
                "heldout": idx in holdout_ids,
            }
        )

    train_samples = [s for s in samples if not s["heldout"]]
    heldout_samples = [s for s in samples if s["heldout"]]
    if not train_samples:
        raise ValueError("Holdout selection leaves no training clips")
    if not heldout_samples:
        print("[H3 multiclip] WARNING: no held-out clips; this will not test generalization")

    token_counts = {int(s["h3"].shape[1]) for s in samples}
    if len(token_counts) != 1:
        raise RuntimeError(f"H3 query row counts differ across samples: {token_counts}")
    query_len_h3 = next(iter(token_counts))

    print("=== H3 MULTI-CLIP BRIDGE ===")
    print(f"tap={args.tap} rank={args.rank} epochs={args.epochs} lr={args.lr:g}")
    print(f"train={[s['index'] for s in train_samples]} holdout={[s['index'] for s in heldout_samples]}")
    print(f"query={query!r}; H3 query rows={query_len_h3}")

    # Direct-invert every sample in one calibrated QR solve before loading Qwen.
    stacked = torch.cat([s["h3"] for s in samples], dim=0)
    dit_path = _resolve_dit(models_path, partition, args.dit_path)
    weight, bias = _load_condition_proj(dit_path)
    print(f"[H3 multiclip] Direct-inverting {len(samples)} {args.tap} states")
    base_all, inv_stats = _direct_inverse_many(stacked, weight, bias, device)
    del weight, bias, stacked
    _cleanup_cuda()

    print(f"[H3 multiclip] Loading frozen Qwen donor: {args.stock_model}")
    stock, processor = _load_stock(args.stock_model)
    stock.model.visual = nn.Identity()
    stock.eval()
    for param in stock.parameters():
        param.requires_grad_(False)
    _cleanup_cuda()
    _cuda_mem("after discarding unused donor vision tower")

    prefill_inputs, query_len = _tokenize_prefill(processor, query, args.suffix)
    if query_len != query_len_h3:
        raise RuntimeError(f"Tokenizer query len={query_len}, H3 rows={query_len_h3}")
    prefill_inputs = _move_inputs(prefill_inputs, device)
    stock_h = _capture_layer_input(stock, prefill_inputs, LOWER_LAYER_COUNT)
    stock_query = stock_h[:, :query_len].to(device)
    print(f"[H3 multiclip] stock layer50 query RMS={_rms(stock_query.cpu()):.5f}")

    # Materialize sample-specific bases and tokenized supervision.
    for i, sample in enumerate(samples):
        sample["h3_dev"] = sample["h3"].to(device)
        sample["base"] = _per_token_match_scale(base_all[i : i + 1].to(device), stock_query)
        full, qlen, target_start, target_len = _tokenize_full(
            processor, query, args.suffix, sample["target"]
        )
        if qlen != query_len:
            raise RuntimeError(f"Tokenizer query mismatch for clip {sample['index']}")
        sample["train_inputs"] = _move_inputs(full, device)
        sample["target_start"] = target_start
        sample["target_len"] = target_len

    del base_all
    _cleanup_cuda()

    adapter = ResidualBridge(args.rank).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = []
    total_updates = 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_samples)))
        random.shuffle(order)
        epoch_losses = []
        epoch_accs = []

        for j in order:
            sample = train_samples[j]
            optimizer.zero_grad(set_to_none=True)
            replacement = adapter(sample["h3_dev"], sample["base"])
            outputs = _train_step_forward(
                stock, sample["train_inputs"], replacement, query_len
            )
            target_start = sample["target_start"]
            logits = outputs.logits[:, target_start - 1 : -1, :].float()
            labels = sample["train_inputs"]["input_ids"][:, target_start:]
            if logits.shape[1] != labels.shape[1]:
                raise RuntimeError(
                    f"Loss alignment mismatch clip={sample['index']} "
                    f"logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
                )
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                acc = float((logits.argmax(dim=-1) == labels).float().mean().item())
            epoch_losses.append(float(loss.detach().item()))
            epoch_accs.append(acc)
            total_updates += 1
            del outputs, logits, labels, loss, replacement

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        mean_acc = sum(epoch_accs) / len(epoch_accs)
        history.append({"epoch": epoch, "loss": mean_loss, "teacher_acc": mean_acc})
        if epoch == 1 or epoch % args.print_every == 0 or mean_loss <= args.early_stop_loss:
            print(
                f"epoch {epoch:3d}  updates={total_updates:4d}  "
                f"train_loss={mean_loss:.5f}  teacher_acc={mean_acc:.3f}"
            )
        if mean_loss <= args.early_stop_loss:
            print(f"[H3 multiclip] Early-stop train threshold reached at epoch {epoch}")
            break

    adapter.eval()
    print("\n=== GENERATION EVALUATION ===")
    generations = []
    for sample in samples:
        text = _generate_with_adapter(
            model=stock,
            processor=processor,
            prefill_inputs=prefill_inputs,
            query_len=query_len,
            adapter=adapter,
            h3_state=sample["h3_dev"],
            base_state=sample["base"],
            max_new_tokens=args.max_new_tokens,
        )
        split = "HOLDOUT" if sample["heldout"] else "train"
        print(f"\n[{sample['index']:02d} | {split} | {sample['file']}]\n{text or '<empty>'}")
        print(f"TARGET: {sample['target']}")
        generations.append(
            {
                "index": sample["index"],
                "file": sample["file"],
                "split": split,
                "target": sample["target"],
                "generation": text,
            }
        )

    result = {
        "index": str(index_path),
        "tap": args.tap,
        "rank": args.rank,
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "updates": total_updates,
        "learning_rate": args.lr,
        "train_indices": [s["index"] for s in train_samples],
        "holdout_indices": [s["index"] for s in heldout_samples],
        "history": history,
        "inverse_stats": inv_stats,
        "generations": generations,
    }

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[H3 multiclip] Wrote {out}")

    if args.adapter_output:
        out = Path(args.adapter_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "tap": args.tap,
                "rank": args.rank,
                "query": query,
                "suffix": args.suffix,
                "train_indices": [s["index"] for s in train_samples],
                "holdout_indices": [s["index"] for s in heldout_samples],
                "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            },
            out,
        )
        print(f"[H3 multiclip] Wrote adapter {out}")

    print(
        "\n[H3 multiclip] Interpretation:\n"
        "  • Training clips only show capacity/memorization.\n"
        "  • Held-out clips are the gate. Source-specific people/objects/settings/actions/audio facts there\n"
        "    mean the shared adapter is extracting information from H3 states it never trained on.\n"
        "  • Generic/repeated/nearest-training captions on holdout mean 8 training clips are insufficient\n"
        "    or the 8-token query-prefix readout needs a stronger architecture."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
