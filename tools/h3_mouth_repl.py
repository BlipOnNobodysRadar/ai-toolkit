#!/usr/bin/env python3
"""Persistent interactive shell for the experimental H3 Qwen3-VL mouth graft.

Loads the hybrid model once, keeps it resident, and lets you ask repeated text
or video questions without paying the ~20 GiB model reload cost every turn.

Commands:
  /video PATH      set the active video
  /show            print the active video
  /text PROMPT     ask a text-only question
  /ask PROMPT      ask about the active video
  /h3              alias for /h3base
  /h3base          rewrite active video using MiniMax's official base prompt guide
  /h3ref           rewrite active video using MiniMax's official full-reference guide
  /tokens N        set max_new_tokens for subsequent generations
  /help            show commands
  /quit            exit

Any non-command line is treated like /ask when a video is active, otherwise
like /text.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from functools import lru_cache
from pathlib import Path

# Running `python tools/h3_mouth_repl.py` puts tools/ rather than the repository
# root on sys.path. Bootstrap the root before importing the sibling tool module.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.h3_mouth_probe import (
    DEFAULT_H3_MODEL,
    DEFAULT_STOCK_MODEL,
    _check_compatibility,
    _cleanup_cuda,
    _decode_generation,
    _discard_stock_lower,
    _graft_h3_lower,
    _load_h3_conditioner,
    _load_stock,
    _move_graft_to_cuda,
    _text_inputs,
    _video_inputs,
)


GUIDE_REPO = "MiniMaxAI/MiniMax-H3"
BASE_GUIDE_PATH = "docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md"
REF_GUIDE_PATH = "docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md"


@lru_cache(maxsize=2)
def _official_guide(kind: str) -> str:
    """Load MiniMax's official guide lazily and cache it for this REPL session."""
    from huggingface_hub import hf_hub_download

    filename = BASE_GUIDE_PATH if kind == "base" else REF_GUIDE_PATH
    path = hf_hub_download(repo_id=GUIDE_REPO, filename=filename)
    return Path(path).read_text(encoding="utf-8")


def _h3_prompt_request(kind: str) -> str:
    guide = _official_guide(kind)
    if kind == "base":
        task_note = """
The supplied asset is the target video itself. Rewrite what is OBSERVED into the T2VA-style three-core-field format from the guide. Do not add an image-alignment instruction. Preserve real shot boundaries and use the guide's shot/cut/camera terminology. The model receiving this request can see the video but cannot hear its soundtrack. Therefore do NOT fabricate dialogue, singing, ambience, sound effects, or music. Where the required audio fields cannot be determined from vision, write exactly `AUDIO_UNAVAILABLE_FROM_VISUAL_MODEL` rather than `N/A` (the guide reserves N/A for known absence/silence). This is an intermediate visual-only rewrite that will later be completed from an audio-capable model.
"""
    else:
        task_note = """
The supplied asset is the target/reference video being analyzed. Follow the full-reference guide's six-section organization exactly where applicable. Do not invent reference assets that were not supplied. The model receiving this request can see the video but cannot hear its soundtrack. Therefore do NOT fabricate dialogue, singing, ambience, sound effects, or music; mark audio-only facts as `AUDIO_UNAVAILABLE_FROM_VISUAL_MODEL`. This is an intermediate visual-only rewrite that will later be completed from an audio-capable model.
"""

    return f"""You are producing a MiniMax H3 prompt rewrite for this exact observed video.

Follow the OFFICIAL MiniMax H3 guide below, including its field names, ordering, shot notation, cut-time format, speaker/dialogue conventions when actually observable, camera-motion terminology, and reference-label rules. Preserve chronological order and concrete visual detail. Output only the final rewrite, not commentary about the guide.
{task_note}

--- OFFICIAL MINIMAX H3 GUIDE START ---
{guide}
--- OFFICIAL MINIMAX H3 GUIDE END ---
"""


def build_hybrid(models_path: str, stock_model: str, h3_model: str):
    stock, processor = _load_stock(stock_model)
    _discard_stock_lower(stock)

    holder, h3_tokenizer, h3_processor, h3_te = _load_h3_conditioner(
        models_path,
        h3_model,
    )
    _check_compatibility(stock, h3_te, processor, h3_tokenizer)
    _graft_h3_lower(stock, h3_te)

    # The transplanted modules are now owned by `stock` too. Drop the loader
    # wrappers before moving the completed hybrid onto CUDA.
    del h3_te, h3_processor, h3_tokenizer, holder
    _cleanup_cuda()
    _move_graft_to_cuda(stock)
    return stock, processor


def resolve_video(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def print_help() -> None:
    print(
        """
Commands:
  /video PATH      set active video (quote paths containing spaces)
  /show            show active video
  /text PROMPT     text-only generation
  /ask PROMPT      ask about active video
  /h3              alias for /h3base
  /h3base          official H3 T2VA/base-format visual rewrite
  /h3ref           official H3 full-reference-format visual rewrite
  /tokens N        change max_new_tokens
  /help            show this help
  /quit            exit

The /h3* commands use MiniMaxAI/MiniMax-H3's official prompt-writing guides.
Because this Qwen3-VL graft cannot hear audio, audio-only fields are explicitly
marked AUDIO_UNAVAILABLE_FROM_VISUAL_MODEL rather than hallucinated.

Plain text behaves like /ask when a video is active, otherwise /text.
""".strip()
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Keep the H3 mouth graft loaded for repeated interactive tests."
    )
    p.add_argument("--models-path", required=True)
    p.add_argument("--video", help="optional initial active video")
    p.add_argument("--stock-model", default=DEFAULT_STOCK_MODEL)
    p.add_argument("--h3-model", default=DEFAULT_H3_MODEL)
    p.add_argument("--qwen-fps", type=float, default=2.0)
    p.add_argument("--qwen-video-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=384)
    args = p.parse_args()

    video = resolve_video(args.video) if args.video else None
    max_tokens = args.max_new_tokens

    model, processor = build_hybrid(
        args.models_path,
        args.stock_model,
        args.h3_model,
    )

    print("\n[H3 mouth REPL] Hybrid loaded and staying resident.")
    if video is not None:
        print(f"[H3 mouth REPL] Active video: {video}")
    print_help()

    while True:
        try:
            raw = input("\nh3-mouth> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[H3 mouth REPL] Exiting.")
            break

        if not raw:
            continue

        if raw in ("/quit", "/exit", "/q"):
            break
        if raw == "/help":
            print_help()
            continue
        if raw == "/show":
            print(f"Active video: {video if video is not None else '<none>'}")
            print(f"max_new_tokens: {max_tokens}")
            continue
        if raw.startswith("/tokens "):
            try:
                max_tokens = int(raw.split(None, 1)[1])
                if max_tokens < 1:
                    raise ValueError
                print(f"max_new_tokens = {max_tokens}")
            except ValueError:
                print("Usage: /tokens POSITIVE_INTEGER")
            continue
        if raw.startswith("/video "):
            value = raw.split(None, 1)[1]
            try:
                # Permit shell-style quotes without requiring them.
                parsed = shlex.split(value)
                if len(parsed) != 1:
                    raise ValueError("expected exactly one path")
                video = resolve_video(parsed[0])
                print(f"Active video: {video}")
            except Exception as exc:
                print(f"Could not set video: {exc}")
            continue

        mode = None
        prompt = None
        if raw in ("/h3", "/h3base"):
            mode = "video"
            print("[H3 mouth REPL] Loading/caching official base prompt guide...")
            prompt = _h3_prompt_request("base")
        elif raw == "/h3ref":
            mode = "video"
            print("[H3 mouth REPL] Loading/caching official full-reference prompt guide...")
            prompt = _h3_prompt_request("ref")
        elif raw.startswith("/ask "):
            mode = "video"
            prompt = raw.split(None, 1)[1]
        elif raw.startswith("/text "):
            mode = "text"
            prompt = raw.split(None, 1)[1]
        elif raw.startswith("/"):
            print("Unknown command. Type /help.")
            continue
        elif video is not None:
            mode = "video"
            prompt = raw
        else:
            mode = "text"
            prompt = raw

        try:
            if mode == "video":
                if video is None:
                    print("No active video. Use /video PATH first.")
                    continue
                inputs = _video_inputs(
                    processor,
                    video,
                    prompt,
                    args.qwen_fps,
                    args.qwen_video_tokens,
                )
            else:
                inputs = _text_inputs(processor, prompt)

            answer = _decode_generation(
                model,
                processor,
                inputs,
                max_new_tokens=max_tokens,
            )
            print("\n" + (answer or "<empty output>"))
        except Exception as exc:
            print(f"Generation failed: {type(exc).__name__}: {exc}")
            _cleanup_cuda()

    del model, processor
    _cleanup_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
