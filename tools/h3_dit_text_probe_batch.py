#!/usr/bin/env python3
"""Batch MiniMax-H3 text-row probe for a folder/manifest of clips.

This is the multi-clip version of h3_dit_text_probe.py.  It deliberately loads H3
only once: all source videos/audio are VAE-encoded first while the DiT is parked,
then the common query is encoded once, then the DiT is moved to CUDA and every
clip is probed.  Each clip gets its own .h3textprobe.pt compatible with the bridge
experiments plus a batch_index.json for shared-adapter training.

The default captures REAL media only.  Pass --controls real,zero,reversed if you
want the heavier controls too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# toolkit.paths freezes MODELS_PATH at import time, so honor --models-path before
# importing any H3/toolkit module.
def _early_value(flag: str):
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None

_early_models = _early_value("--models-path")
if _early_models:
    os.environ["MODELS_PATH"] = str(Path(_early_models).expanduser().resolve())

import argparse
import json
import subprocess
import tempfile
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.h3_caption_lab import DEFAULT_ASSISTANT_LORA, load_video_tensor
from tools.h3_dit_text_probe import (
    DEFAULT_BLOCKS,
    DEFAULT_QUERY,
    _align_video_for_h3,
    _build_layout_and_rows,
    _capture_text_states,
    _cleanup_cuda,
    _embed_query,
    _load_audio_file,
    _make_h3,
    _park,
    _parse_blocks,
    _prepare_media_variants,
    _seed_everything,
)


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _extract_audio(video: Path, wav: Path, duration: float) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-t", f"{duration:.6f}",
        "-vn", "-ac", "2", "-ar", "32000", "-c:a", "pcm_s16le",
        str(wav),
    ]
    subprocess.run(cmd, check=True)


def _load_manifest(path: Path, folder: Path) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    query = str(data.get("query") or DEFAULT_QUERY)
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no entries: {path}")
    out = []
    for item in entries:
        if not isinstance(item, dict) or not item.get("file"):
            raise ValueError(f"Bad manifest entry: {item!r}")
        video = (folder / str(item["file"])).resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        row = dict(item)
        row["video"] = str(video)
        out.append(row)
    return query, out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--models-path", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--query", help="Override manifest query")
    p.add_argument("--h3-model", default="Comfy-Org/MiniMax-H3")
    p.add_argument(
        "--partition", default="fl2va_pruned",
        choices=("fl2va", "fl2va_pruned", "ref2va", "ref2va_pruned"),
    )
    p.add_argument("--assistant-lora", default=DEFAULT_ASSISTANT_LORA)
    p.add_argument("--use-assistant-lora", action="store_true")
    p.add_argument("--blocks", default=DEFAULT_BLOCKS)
    p.add_argument("--controls", default="real", help="Comma list: real,zero,reversed")
    p.add_argument("--t", type=float, default=0.999)
    p.add_argument("--max-seconds", type=float, default=15.0)
    p.add_argument("--max-edge", type=int, default=256)
    p.add_argument("--latent-seed", type=int, default=1701)
    p.add_argument("--noise-seed", type=int, default=1776)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0.0 <= args.t <= 1.0:
        p.error("--t must be in [0,1]")

    folder = Path(args.folder).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    query, entries = _load_manifest(manifest, folder)
    if args.query:
        query = args.query

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else folder / ".h3textprobe"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    controls = _csv(args.controls)
    valid_controls = {"real", "zero", "reversed"}
    if not controls or any(c not in valid_controls for c in controls):
        p.error("--controls must be a comma list drawn from real,zero,reversed")

    print(f"[H3 batch probe] {len(entries)} clips; query={query!r}")
    print("[H3 batch probe] Loading H3 once")
    h3 = _make_h3(args)
    transformer = h3.model
    taps = _parse_blocks(args.blocks, len(transformer.blocks))

    # Phase 1: VAE-encode every clip while the enormous DiT stays parked on CPU.
    encoded: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="h3_batch_audio_") as td:
        tmp = Path(td)
        for ordinal, entry in enumerate(entries, 1):
            video_path = Path(entry["video"])
            print(f"\n[H3 batch probe] [{ordinal}/{len(entries)}] VAE encode {video_path.name}")
            video = load_video_tensor(
                video_path, fps=24.0, max_seconds=args.max_seconds, max_edge=args.max_edge
            )
            video = _align_video_for_h3(video)
            num_frames = int(video.shape[0])
            duration = num_frames / 24.0
            print(
                f"[H3 batch probe]   {num_frames} frames ({duration:.3f}s) "
                f"at {video.shape[-1]}x{video.shape[-2]}"
            )

            _seed_everything(args.latent_seed + ordinal)
            with torch.inference_mode():
                video_latents = h3.encode_images([video]).detach().to("cpu", torch.float32)
            del video

            wav = tmp / f"{ordinal:02d}.wav"
            _extract_audio(video_path, wav, duration)
            audio_data = _load_audio_file(wav, duration)
            _seed_everything((args.latent_seed ^ 0x31A9B7) + ordinal)
            with torch.inference_mode():
                audio_rows = h3.encode_audio([audio_data]).detach().to("cpu", torch.float32)
            del audio_data

            encoded.append(
                {
                    "entry": entry,
                    "video_latents": video_latents,
                    "audio_rows": audio_rows,
                    "num_frames": num_frames,
                    "duration": duration,
                }
            )

    _park(h3.vae)
    _cleanup_cuda()

    # Same language query for every sample. Encode it once.
    print(f"\n[H3 batch probe] Encoding common H3 Qwen query")
    query_emb, text_tags, token_strings = _embed_query(h3, query)
    print(f"[H3 batch probe] query tokens={query_emb.shape[0]}; taps={taps}; controls={controls}")

    # Phase 2: park TE/VAEs and use the DiT for each pre-encoded clip.
    device = torch.device("cuda:0")
    _cleanup_cuda()
    if transformer.device == torch.device("cpu"):
        transformer.to(device)

    index_rows = []
    for ordinal, item in enumerate(encoded, 1):
        entry = item["entry"]
        video_path = Path(entry["video"])
        print(f"\n[H3 batch probe] [{ordinal}/{len(encoded)}] DiT probe {video_path.name}")
        variants, t_audio = _prepare_media_variants(
            video_latents=item["video_latents"],
            audio_rows_clean=item["audio_rows"],
            t_video=args.t,
            seed=args.noise_seed + ordinal,
            device=device,
        )

        captures = {}
        for control in controls:
            print(f"[H3 batch probe]   {control} pass")
            v_noisy, a_noisy = variants[control]
            layout, row_t, video_rows, audio_rows = _build_layout_and_rows(
                h3=h3,
                text_tags=text_tags,
                video_noisy=v_noisy,
                audio_noisy=a_noisy,
                t_video=args.t,
                t_audio=t_audio,
            )
            captures[control] = _capture_text_states(
                transformer=transformer,
                query_emb=query_emb,
                layout=layout,
                row_t=row_t,
                video_rows=video_rows,
                audio_rows=audio_rows,
                taps=taps,
                device=device,
            )
            del video_rows, audio_rows
            torch.cuda.empty_cache()

        capture_path = output_dir / f"{video_path.name}.h3textprobe.pt"
        if capture_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {capture_path}; pass --overwrite"
            )
        payload = {
            "version": 2,
            "batch_manifest": str(manifest),
            "manifest_index": entry.get("index", ordinal),
            "video": str(video_path),
            "query": query,
            "query_tokens": token_strings,
            "num_frames": item["num_frames"],
            "duration_seconds": item["duration"],
            "max_edge": args.max_edge,
            "partition": args.partition,
            "assistant_lora_active": bool(args.use_assistant_lora),
            "video_t": float(args.t),
            "audio_t": float(t_audio),
            "blocks": taps,
            "target": entry.get("target"),
            "captures": captures,
        }
        torch.save(payload, capture_path)
        print(f"[H3 batch probe]   saved {capture_path}")

        index_row = {k: v for k, v in entry.items() if k != "video"}
        index_row.update(
            {
                "video": str(video_path),
                "capture": str(capture_path),
                "num_frames": item["num_frames"],
                "duration_seconds": item["duration"],
            }
        )
        index_rows.append(index_row)

        del variants, captures
        torch.cuda.empty_cache()

    index_path = output_dir / "batch_index.json"
    index_payload = {
        "version": 1,
        "manifest": str(manifest),
        "query": query,
        "partition": args.partition,
        "blocks": taps,
        "controls": controls,
        "entries": index_rows,
    }
    index_path.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[H3 batch probe] COMPLETE: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
