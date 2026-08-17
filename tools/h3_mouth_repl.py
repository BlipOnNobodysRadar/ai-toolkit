#!/usr/bin/env python3
"""Persistent interactive shell for the experimental H3 Qwen3-VL mouth graft.

Loads the hybrid model once, keeps it resident, and lets you ask repeated text
or video questions without paying the ~20 GiB model reload cost every turn.

Commands:
  /video PATH      set the active video
  /show            print the active video
  /text PROMPT     ask a text-only question
  /ask PROMPT      ask about the active video
  /h3              produce an H3-oriented generation prompt for the active video
  /tokens N        set max_new_tokens for subsequent generations
  /help            show commands
  /quit            exit

Any non-command line is treated like /ask when a video is active, otherwise
like /text.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

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


H3_PROMPT_REQUEST = """Write a MiniMax H3 generation prompt that would reproduce this exact clip as closely as possible.

Use natural-language production directions rather than commentary about the source video. Preserve the chronological sequence of shots and actions. Include concrete subject appearance, environment, lighting, framing, shot scale, camera angle, camera movement, subject movement, transitions, visible readable text, and continuity details when they matter. Distinguish camera motion from subject motion. Do not invent anything that is not visibly supported. Do not invent dialogue, music, ambience, or sound effects because this vision-only graft cannot hear the soundtrack.

Return only the generation prompt, with enough temporal structure to recreate the clip."""


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
  /h3              write an H3-oriented generation prompt for active video
  /tokens N        change max_new_tokens
  /help            show this help
  /quit            exit

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
        if raw == "/h3":
            mode = "video"
            prompt = H3_PROMPT_REQUEST
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
