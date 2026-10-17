#!/usr/bin/env python3
"""Inspect how similar H3 DiT text-row states are across a batch of clips.

This is a cheap diagnostic for the multi-clip caption bridge. It loads the
REAL captures from batch_index.json, compares clips at several H3 block taps,
and reports each requested holdout clip's nearest training clips under:

  * raw cosine similarity on the full (tokens x hidden) state;
  * per-token RMS-normalized cosine, matching the normalization used by the
    learned residual bridge more closely;
  * relative RMS distance.

No H3 or Qwen model is loaded. The point is to test whether free-generation
confusions from the learned bridge simply mirror nearest-neighbor structure in
H3 state space (for example, a held-out soccer clip collapsing to whichever
training clip is closest at block_30).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _csv_ints(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _load_probe(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(-1)


def _token_rms_normalize(x: torch.Tensor) -> torch.Tensor:
    # capture shape is (1, L, D)
    y = x.float()
    rms = torch.sqrt(torch.mean(y.square(), dim=-1, keepdim=True)).clamp_min(1e-20)
    return y / rms


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = _flatten(a)
    bf = _flatten(b)
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    return float((torch.dot(af, bf) / denom.clamp_min(1e-20)).item())


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    diff = torch.sqrt(torch.mean((a - b).square()))
    scale = 0.5 * (
        torch.sqrt(torch.mean(a.square())) + torch.sqrt(torch.mean(b.square()))
    )
    return float((diff / scale.clamp_min(1e-20)).item())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("index", help="batch_index.json from h3_dit_text_probe_batch.py")
    p.add_argument("--holdout", default="9,10")
    p.add_argument(
        "--taps",
        default="block_1,block_5,block_10,block_20,block_30,block_40,block_50",
    )
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--json-output")
    args = p.parse_args()

    index_path = Path(args.index).expanduser().resolve()
    data = json.loads(index_path.read_text(encoding="utf-8"))
    entries = data.get("entries") or []
    if not entries:
        raise ValueError(f"No entries in {index_path}")

    holdout = _csv_ints(args.holdout)
    taps = _csv(args.taps)

    samples = []
    for ordinal, entry in enumerate(entries, 1):
        idx = int(entry.get("index", ordinal))
        capture = Path(entry["capture"]).expanduser().resolve()
        payload = _load_probe(capture)
        real = payload["captures"].get("real")
        if real is None:
            raise KeyError(f"{capture} has no real capture")
        samples.append(
            {
                "index": idx,
                "file": entry.get("file") or Path(payload["video"]).name,
                "target": str(entry.get("target") or ""),
                "real": real,
                "heldout": idx in holdout,
            }
        )

    train = [s for s in samples if not s["heldout"]]
    tests = [s for s in samples if s["heldout"]]
    if not train or not tests:
        raise ValueError("Need at least one training and one holdout sample")

    report = {"index": str(index_path), "holdout": sorted(holdout), "taps": {}}

    print("=== H3 TEXT-STATE NEAREST-NEIGHBOR DIAGNOSTIC ===")
    print(f"train={[s['index'] for s in train]} holdout={[s['index'] for s in tests]}")

    for tap in taps:
        for s in samples:
            if tap not in s["real"]:
                raise KeyError(f"clip {s['index']} lacks {tap}; available={list(s['real'])}")

        print(f"\n--- {tap} ---")
        tap_out = {}
        for test in tests:
            x = test["real"][tap]
            xn = _token_rms_normalize(x)
            rows = []
            for tr in train:
                y = tr["real"][tap]
                yn = _token_rms_normalize(y)
                rows.append(
                    {
                        "index": tr["index"],
                        "file": tr["file"],
                        "raw_cos": _cos(x, y),
                        "norm_cos": _cos(xn, yn),
                        "rel_rms": _rel_rms(xn, yn),
                    }
                )

            by_norm = sorted(rows, key=lambda r: r["norm_cos"], reverse=True)
            by_raw = sorted(rows, key=lambda r: r["raw_cos"], reverse=True)
            by_rms = sorted(rows, key=lambda r: r["rel_rms"])
            tap_out[str(test["index"])] = {
                "file": test["file"],
                "target": test["target"],
                "nearest_norm_cos": by_norm,
                "nearest_raw_cos": by_raw,
                "nearest_rel_rms": by_rms,
            }

            print(f"holdout {test['index']:02d} | {test['file']}")
            print("  nearest by per-token-normalized cosine:")
            for r in by_norm[: max(1, args.top_k)]:
                print(
                    f"    train {r['index']:02d}  norm_cos={r['norm_cos']:.8f} "
                    f"raw_cos={r['raw_cos']:.8f} rel_rms={r['rel_rms']:.6f}  {r['file']}"
                )
            print("  nearest by normalized relative RMS:")
            for r in by_rms[: max(1, args.top_k)]:
                print(
                    f"    train {r['index']:02d}  rel_rms={r['rel_rms']:.6f} "
                    f"norm_cos={r['norm_cos']:.8f}  {r['file']}"
                )

        report["taps"][tap] = tap_out

    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {out}")

    print(
        "\nInterpretation:\n"
        "  • If the learned bridge's wrong caption comes from the nearest training state at the same tap,\n"
        "    the current bridge is behaving largely like a prototype/nearest-neighbor selector.\n"
        "  • If another tap gives much more semantically sensible neighbors, it is a better candidate for\n"
        "    the next learned bridge before increasing model complexity.\n"
        "  • If nearest states do not match the generation confusions, the failure is more likely in the\n"
        "    adapter/objective or Qwen decoding dynamics than in H3 state geometry itself."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
