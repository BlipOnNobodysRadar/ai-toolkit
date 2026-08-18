#!/usr/bin/env python3
"""Bridge v2: decode media-induced MiniMax-H3 text-state residuals into Qwen.

This intentionally abandons the deep-state condition_proj pseudoinverse used by
bridge v1.  condition_proj is calibrated only at the H3 entrance boundary; after
many DiT blocks the text rows have moved into H3's own multimodal residual space.

Bridge v2 therefore uses a clean stock-Qwen layer-50 query state as the language
anchor and learns only a media-conditioned delta from H3.  By default it:

  * reads block_20 + block_30 + block_40 together;
  * subtracts the TRAINING-SET mean H3 state at each tap, removing the enormous
    shared "Describe the video and audio..." trajectory without holdout leakage;
  * normalizes each tap by its training residual RMS;
  * mixes depth and query-token positions with a small Transformer + learned
    cross-attention query reader;
  * adds the resulting 8 x 5120 delta to the clean stock Qwen layer-50 query state;
  * trains the frozen Qwen mouth with early-token-weighted caption loss;
  * adds a wrong-caption margin loss over the first caption tokens, explicitly
    teaching the media prefix to choose the correct trajectory instead of merely
    continuing teacher-forced text;
  * evaluates free generation on train and held-out clips;
  * ranks all known captions for each holdout by early-prefix NLL as a separate
    discriminative readout diagnostic.

If captures contain ZERO controls, --residual-mode auto will prefer REAL-ZERO,
which is an even cleaner causal media residual.  With the current REAL-only
10-clip batch it automatically uses TRAIN-MEAN centering.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.h3_dit_bridge_overfit import (
    _tokenize_full,
    _tokenize_prefill,
    _train_step_forward,
)
from tools.h3_dit_mouth_bridge import (
    _capture_layer_input,
    _generate,
    _load_probe,
    _move_inputs,
    _rms,
)
from tools.h3_mouth_probe import (
    DEFAULT_STOCK_MODEL,
    LOWER_LAYER_COUNT,
    _cleanup_cuda,
    _cuda_mem,
    _load_stock,
)


DEFAULT_TAPS = "block_20,block_30,block_40"


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _csv_ints(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def _load_index(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("entries"), list) or not data["entries"]:
        raise ValueError(f"No entries in {path}")
    return data


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _target_losses(outputs, inputs: dict, target_start: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-token CE, labels, logits for the target region only."""
    logits = outputs.logits[:, target_start - 1 : -1, :].float()
    labels = inputs["input_ids"][:, target_start:]
    if logits.shape[1] != labels.shape[1]:
        raise RuntimeError(
            f"Loss alignment mismatch logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    return losses, labels, logits


def _weighted_caption_loss(
    token_losses: torch.Tensor,
    *,
    early_tokens: int,
    early_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(token_losses)
    n = min(max(0, early_tokens), token_losses.shape[1])
    if n > 0 and early_weight != 1.0:
        weights[:, :n] = float(early_weight)
    return (token_losses * weights).sum() / weights.sum().clamp_min(1.0)


def _prefix_mean(token_losses: torch.Tensor, count: int) -> torch.Tensor:
    n = min(max(1, count), token_losses.shape[1])
    return token_losses[:, :n].mean()


class H3ResidualReader(nn.Module):
    """Mix multiple H3 taps/tokens and emit a Qwen layer-50 query delta.

    Input shape:  (B, num_taps, query_tokens, 5376)
    Output shape: (B, query_tokens, 5120)

    A shared H3->hidden projection avoids giving each tap a giant private matrix.
    Learned tap/token embeddings plus a small Transformer let the reader combine
    information across depth and query positions.  Learned output query slots then
    cross-attend over those source states and project to Qwen hidden space.
    """

    def __init__(
        self,
        *,
        num_taps: int,
        query_tokens: int,
        hidden: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        self.num_taps = num_taps
        self.query_tokens = query_tokens
        self.hidden = hidden

        self.input_norm = nn.LayerNorm(5376, elementwise_affine=False)
        self.input_proj = nn.Linear(5376, hidden, bias=False)
        self.tap_embed = nn.Parameter(torch.zeros(num_taps, hidden))
        self.token_embed = nn.Parameter(torch.zeros(query_tokens, hidden))
        nn.init.normal_(self.tap_embed, std=0.02)
        nn.init.normal_(self.token_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.encoder_norm = nn.LayerNorm(hidden)

        self.output_queries = nn.Parameter(torch.empty(query_tokens, hidden))
        nn.init.normal_(self.output_queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm1 = nn.LayerNorm(hidden)
        self.query_ff = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.query_norm2 = nn.LayerNorm(hidden)

        self.output_proj = nn.Linear(hidden, 5120, bias=False)
        # Start bridge v2 as a transparent clean-Qwen anchor.  Training learns
        # only the media delta from zero rather than fighting a garbage deep inverse.
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,T,L,5376), got {tuple(x.shape)}")
        b, t, l, d = x.shape
        if t != self.num_taps or l != self.query_tokens or d != 5376:
            raise ValueError(
                f"Expected (*,{self.num_taps},{self.query_tokens},5376), got {tuple(x.shape)}"
            )

        z = self.input_proj(self.input_norm(x.float()))
        z = z + self.tap_embed.view(1, t, 1, self.hidden)
        z = z + self.token_embed.view(1, 1, l, self.hidden)
        z = z.reshape(b, t * l, self.hidden)
        z = self.encoder_norm(self.encoder(z))

        q = self.output_queries.unsqueeze(0).expand(b, -1, -1)
        attn, _ = self.cross_attn(q, z, z, need_weights=False)
        q = self.query_norm1(q + attn)
        q = self.query_norm2(q + self.query_ff(q))
        return self.output_proj(q)


def _build_residuals(
    samples: list[dict],
    train_samples: list[dict],
    taps: list[str],
    mode: str,
) -> tuple[str, dict[str, torch.Tensor], dict[str, float]]:
    all_have_zero = all(
        "zero" in s["captures"] and all(tap in s["captures"]["zero"] for tap in taps)
        for s in samples
    )
    if mode == "auto":
        mode = "zero" if all_have_zero else "train-mean"
    if mode == "zero" and not all_have_zero:
        raise ValueError(
            "--residual-mode zero requested, but at least one capture lacks ZERO controls. "
            "Rerun h3_dit_text_probe_batch.py with --controls real,zero or use train-mean."
        )
    if mode not in ("zero", "train-mean"):
        raise ValueError(mode)

    centers: dict[str, torch.Tensor] = {}
    if mode == "train-mean":
        for tap in taps:
            centers[tap] = torch.stack(
                [s["captures"]["real"][tap].float() for s in train_samples], dim=0
            ).mean(dim=0)
    else:
        for tap in taps:
            centers[tap] = torch.zeros_like(samples[0]["captures"]["real"][tap].float())

    for sample in samples:
        sample["residuals"] = {}
        for tap in taps:
            real = sample["captures"]["real"][tap].float()
            if mode == "zero":
                residual = real - sample["captures"]["zero"][tap].float()
            else:
                residual = real - centers[tap]
            sample["residuals"][tap] = residual

    # One scalar training RMS per tap makes block_40 unable to dominate merely
    # because its residual stream has a much larger numerical scale.
    scales: dict[str, float] = {}
    for tap in taps:
        train_stack = torch.cat([s["residuals"][tap] for s in train_samples], dim=0)
        scale = _rms(train_stack)
        scales[tap] = max(scale, 1e-8)

    for sample in samples:
        normalized = [sample["residuals"][tap] / scales[tap] for tap in taps]
        sample["reader_input"] = torch.stack(normalized, dim=1)  # (1,T,L,D)

    return mode, centers, scales


def _replacement(
    reader: H3ResidualReader,
    reader_input: torch.Tensor,
    anchor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = reader(reader_input)
    return anchor.float() + delta, delta


def _generate_one(
    *,
    model,
    processor,
    prefill_inputs: dict,
    query_len: int,
    reader: H3ResidualReader,
    reader_input: torch.Tensor,
    anchor: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, float]:
    with torch.no_grad():
        replacement, delta = _replacement(reader, reader_input, anchor)
        delta_rms = _rms(delta.detach().cpu())
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
        raise RuntimeError("Bridge-v2 generation hook never fired")
    return text, delta_rms


def _score_candidate_prefix(
    *,
    model,
    replacement: torch.Tensor,
    query_len: int,
    candidate: dict,
    prefix_tokens: int,
) -> float:
    with torch.no_grad():
        outputs = _train_step_forward(
            model,
            candidate["train_inputs"],
            replacement,
            query_len,
        )
        losses, _labels, _logits = _target_losses(
            outputs,
            candidate["train_inputs"],
            candidate["target_start"],
        )
        value = float(_prefix_mean(losses, prefix_tokens).item())
    del outputs, losses, _labels, _logits
    return value


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("index", help="batch_index.json from h3_dit_text_probe_batch.py")
    p.add_argument("--models-path", required=True, help="Kept for CLI consistency; Qwen donor may use HF cache")
    p.add_argument("--stock-model", default=DEFAULT_STOCK_MODEL)
    p.add_argument("--taps", default=DEFAULT_TAPS)
    p.add_argument("--holdout", default="9,10")
    p.add_argument("--suffix", default="\nCaption:")
    p.add_argument(
        "--residual-mode",
        choices=("auto", "train-mean", "zero"),
        default="auto",
        help="auto prefers REAL-ZERO when all captures have ZERO, otherwise training-mean centering",
    )
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--reader-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--early-tokens", type=int, default=20)
    p.add_argument("--early-weight", type=float, default=4.0)
    p.add_argument("--contrastive-tokens", type=int, default=16)
    p.add_argument("--contrastive-margin", type=float, default=0.75)
    p.add_argument("--contrastive-weight", type=float, default=1.0)
    p.add_argument(
        "--delta-reg",
        type=float,
        default=1e-4,
        help="Penalty on delta RMS relative to clean Qwen anchor RMS",
    )
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument(
        "--rank-holdout-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, score every known caption against each held-out H3 state",
    )
    p.add_argument("--candidate-top-k", type=int, default=5)
    p.add_argument("--output")
    p.add_argument("--adapter-output")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.hidden < 1 or args.heads < 1 or args.reader_layers < 1 or args.epochs < 1:
        p.error("hidden, heads, reader-layers and epochs must be positive")
    if args.hidden % args.heads:
        p.error("--hidden must be divisible by --heads")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    index_path = Path(args.index).expanduser().resolve()
    index = _load_index(index_path)
    holdout_ids = _csv_ints(args.holdout)
    taps = _csv(args.taps)
    if not taps:
        p.error("--taps cannot be empty")

    samples: list[dict] = []
    query = None
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
            raise KeyError(f"{capture_path} has no REAL capture")
        for tap in taps:
            if tap not in captures["real"]:
                raise KeyError(f"{capture_path} lacks {tap}; available={list(captures['real'])}")
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
                "captures": captures,
                "heldout": idx in holdout_ids,
            }
        )

    train_samples = [s for s in samples if not s["heldout"]]
    heldout_samples = [s for s in samples if s["heldout"]]
    if not train_samples:
        raise ValueError("Holdout selection leaves no training clips")
    if not heldout_samples:
        print("[H3 bridge v2] WARNING: no held-out clips; this is not a generalization test")

    query_lengths = {
        int(s["captures"]["real"][taps[0]].shape[1]) for s in samples
    }
    if len(query_lengths) != 1:
        raise RuntimeError(f"H3 query row counts differ: {query_lengths}")
    query_len_h3 = next(iter(query_lengths))

    mode, centers, tap_scales = _build_residuals(
        samples, train_samples, taps, args.residual_mode
    )

    print("=== H3 BRIDGE V2 ===")
    print(f"taps={taps} residual_mode={mode}")
    print(
        f"reader hidden={args.hidden} heads={args.heads} layers={args.reader_layers} "
        f"epochs={args.epochs} lr={args.lr:g}"
    )
    print(f"train={[s['index'] for s in train_samples]} holdout={[s['index'] for s in heldout_samples]}")
    print(f"query={query!r}; H3 query rows={query_len_h3}")
    for tap in taps:
        train_rms = [_rms(s["residuals"][tap]) for s in train_samples]
        held_rms = [_rms(s["residuals"][tap]) for s in heldout_samples]
        print(
            f"  {tap:9s} train_resid_rms_mean={sum(train_rms)/len(train_rms):.5f} "
            f"scale={tap_scales[tap]:.5f} "
            f"holdout_resid_rms_mean={(sum(held_rms)/len(held_rms)) if held_rms else float('nan'):.5f}"
        )

    print(f"[H3 bridge v2] Loading frozen Qwen donor: {args.stock_model}")
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
    anchor = stock_h[:, :query_len].to(device).float()
    anchor_rms = _rms(anchor.detach().cpu())
    print(f"[H3 bridge v2] clean stock layer50 anchor RMS={anchor_rms:.5f}")

    # Materialize normalized H3 inputs and all target tokenizations.  We tokenize
    # heldouts too only for evaluation/candidate ranking; they never enter optimizer updates.
    for sample in samples:
        sample["reader_dev"] = sample["reader_input"].to(device)
        full, qlen, target_start, target_len = _tokenize_full(
            processor, query, args.suffix, sample["target"]
        )
        if qlen != query_len:
            raise RuntimeError(f"Tokenizer query mismatch for clip {sample['index']}")
        sample["train_inputs"] = _move_inputs(full, device)
        sample["target_start"] = target_start
        sample["target_len"] = target_len

    reader = H3ResidualReader(
        num_taps=len(taps),
        query_tokens=query_len,
        hidden=args.hidden,
        heads=args.heads,
        layers=args.reader_layers,
        dropout=args.dropout,
    ).to(device)
    trainable = _count_parameters(reader)
    print(f"[H3 bridge v2] trainable reader params={trainable:,}")

    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = []
    total_updates = 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_samples)))
        random.shuffle(order)
        epoch_total = []
        epoch_pos = []
        epoch_rank = []
        epoch_acc = []
        epoch_early_acc = []
        epoch_rank_success = []
        epoch_delta = []

        for j in order:
            sample = train_samples[j]
            wrong_pool = [s for s in train_samples if s["index"] != sample["index"]]
            wrong = random.choice(wrong_pool)

            optimizer.zero_grad(set_to_none=True)
            replacement, delta = _replacement(reader, sample["reader_dev"], anchor)

            pos_outputs = _train_step_forward(
                stock, sample["train_inputs"], replacement, query_len
            )
            pos_losses, pos_labels, pos_logits = _target_losses(
                pos_outputs, sample["train_inputs"], sample["target_start"]
            )
            pos_loss = _weighted_caption_loss(
                pos_losses,
                early_tokens=args.early_tokens,
                early_weight=args.early_weight,
            )
            pos_prefix = _prefix_mean(pos_losses, args.contrastive_tokens)

            neg_outputs = _train_step_forward(
                stock, wrong["train_inputs"], replacement, query_len
            )
            neg_losses, _neg_labels, _neg_logits = _target_losses(
                neg_outputs, wrong["train_inputs"], wrong["target_start"]
            )
            neg_prefix = _prefix_mean(neg_losses, args.contrastive_tokens)
            rank_loss = F.relu(float(args.contrastive_margin) + pos_prefix - neg_prefix)

            delta_reg = delta.float().square().mean() / max(anchor_rms * anchor_rms, 1e-12)
            total_loss = (
                pos_loss
                + float(args.contrastive_weight) * rank_loss
                + float(args.delta_reg) * delta_reg
            )
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(reader.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = pos_logits.argmax(dim=-1)
                token_acc = float((pred == pos_labels).float().mean().item())
                n_early = min(max(1, args.early_tokens), pos_labels.shape[1])
                early_acc = float(
                    (pred[:, :n_early] == pos_labels[:, :n_early]).float().mean().item()
                )
                rank_ok = float((neg_prefix > pos_prefix).item())
                delta_rms = _rms(delta.detach().cpu())

            epoch_total.append(float(total_loss.detach().item()))
            epoch_pos.append(float(pos_loss.detach().item()))
            epoch_rank.append(float(rank_loss.detach().item()))
            epoch_acc.append(token_acc)
            epoch_early_acc.append(early_acc)
            epoch_rank_success.append(rank_ok)
            epoch_delta.append(delta_rms)
            total_updates += 1

            del (
                replacement,
                delta,
                pos_outputs,
                pos_losses,
                pos_labels,
                pos_logits,
                pos_loss,
                pos_prefix,
                neg_outputs,
                neg_losses,
                _neg_labels,
                _neg_logits,
                neg_prefix,
                rank_loss,
                delta_reg,
                total_loss,
            )

        def mean(xs: Iterable[float]) -> float:
            xs = list(xs)
            return sum(xs) / max(1, len(xs))

        row = {
            "epoch": epoch,
            "loss": mean(epoch_total),
            "positive_loss": mean(epoch_pos),
            "rank_loss": mean(epoch_rank),
            "teacher_acc": mean(epoch_acc),
            "early_teacher_acc": mean(epoch_early_acc),
            "rank_success": mean(epoch_rank_success),
            "delta_rms": mean(epoch_delta),
        }
        history.append(row)
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch {epoch:3d} updates={total_updates:4d} "
                f"loss={row['loss']:.4f} pos={row['positive_loss']:.4f} "
                f"rank={row['rank_loss']:.4f} acc={row['teacher_acc']:.3f} "
                f"early_acc={row['early_teacher_acc']:.3f} "
                f"rank_ok={row['rank_success']:.3f} delta_rms={row['delta_rms']:.3f}"
            )

    reader.eval()
    print("\n=== BRIDGE V2 FREE GENERATION ===")
    generations = []
    for sample in samples:
        text, delta_rms = _generate_one(
            model=stock,
            processor=processor,
            prefill_inputs=prefill_inputs,
            query_len=query_len,
            reader=reader,
            reader_input=sample["reader_dev"],
            anchor=anchor,
            max_new_tokens=args.max_new_tokens,
        )
        split = "HOLDOUT" if sample["heldout"] else "train"
        print(f"\n[{sample['index']:02d} | {split} | {sample['file']}] delta_rms={delta_rms:.4f}")
        print(text or "<empty>")
        print(f"TARGET: {sample['target']}")
        generations.append(
            {
                "index": sample["index"],
                "file": sample["file"],
                "split": split,
                "target": sample["target"],
                "generation": text,
                "delta_rms": delta_rms,
            }
        )

    candidate_rankings = []
    if args.rank_holdout_candidates and heldout_samples:
        print("\n=== HOLDOUT CAPTION-CANDIDATE RANKING ===")
        for sample in heldout_samples:
            with torch.no_grad():
                replacement, _delta = _replacement(reader, sample["reader_dev"], anchor)
            scores = []
            for candidate in samples:
                nll = _score_candidate_prefix(
                    model=stock,
                    replacement=replacement,
                    query_len=query_len,
                    candidate=candidate,
                    prefix_tokens=args.contrastive_tokens,
                )
                scores.append(
                    {
                        "index": candidate["index"],
                        "file": candidate["file"],
                        "nll": nll,
                        "is_true": candidate["index"] == sample["index"],
                        "was_training_caption": not candidate["heldout"],
                    }
                )
            scores.sort(key=lambda x: x["nll"])
            true_rank = next(i + 1 for i, s in enumerate(scores) if s["is_true"])
            print(
                f"\nholdout {sample['index']:02d} | true caption rank={true_rank}/{len(scores)} "
                f"using first {args.contrastive_tokens} target tokens"
            )
            for row in scores[: max(1, args.candidate_top_k)]:
                marker = "<-- TRUE" if row["is_true"] else ""
                print(
                    f"  candidate {row['index']:02d} nll={row['nll']:.4f} "
                    f"{'train' if row['was_training_caption'] else 'holdout'} {marker}"
                )
            candidate_rankings.append(
                {
                    "holdout_index": sample["index"],
                    "true_rank": true_rank,
                    "scores": scores,
                }
            )
            del replacement, _delta

    result = {
        "index": str(index_path),
        "version": 2,
        "taps": taps,
        "residual_mode": mode,
        "tap_scales": tap_scales,
        "reader": {
            "hidden": args.hidden,
            "heads": args.heads,
            "layers": args.reader_layers,
            "dropout": args.dropout,
            "trainable_parameters": trainable,
        },
        "objective": {
            "early_tokens": args.early_tokens,
            "early_weight": args.early_weight,
            "contrastive_tokens": args.contrastive_tokens,
            "contrastive_margin": args.contrastive_margin,
            "contrastive_weight": args.contrastive_weight,
            "delta_reg": args.delta_reg,
        },
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "updates": total_updates,
        "learning_rate": args.lr,
        "train_indices": [s["index"] for s in train_samples],
        "holdout_indices": [s["index"] for s in heldout_samples],
        "history": history,
        "generations": generations,
        "candidate_rankings": candidate_rankings,
    }

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[H3 bridge v2] Wrote {out}")

    if args.adapter_output:
        out = Path(args.adapter_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 2,
                "taps": taps,
                "residual_mode": mode,
                "query": query,
                "suffix": args.suffix,
                "query_tokens": query_len,
                "reader_config": {
                    "num_taps": len(taps),
                    "query_tokens": query_len,
                    "hidden": args.hidden,
                    "heads": args.heads,
                    "layers": args.reader_layers,
                    "dropout": args.dropout,
                },
                "tap_scales": tap_scales,
                "centers": {k: v.detach().cpu() for k, v in centers.items()},
                "train_indices": [s["index"] for s in train_samples],
                "holdout_indices": [s["index"] for s in heldout_samples],
                "state_dict": {k: v.detach().cpu() for k, v in reader.state_dict().items()},
            },
            out,
        )
        print(f"[H3 bridge v2] Wrote adapter {out}")

    print(
        "\n[H3 bridge v2] Interpretation:\n"
        "  • Free generation on holdouts is still the primary gate.\n"
        "  • Candidate ranking is a second gate: a good true-caption rank with bad free generation means\n"
        "    the H3 reader has discriminative signal but autoregressive decoding is still collapsing.\n"
        "  • Bad free generation AND bad true-caption ranking means this tiny 8-example bridge still has\n"
        "    not learned a semantic H3->language coordinate system; scale data before adding much capacity.\n"
        "  • If train-mean centering helps, rerun the batch with --controls real,zero and repeat with\n"
        "    --residual-mode zero for the cleaner REAL-ZERO causal media residual."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
