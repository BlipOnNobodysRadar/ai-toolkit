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
  /h3base          two-pass visual rewrite in MiniMax's official T2VA/base format
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
AUDIO_UNKNOWN = "AUDIO_UNAVAILABLE_FROM_VISUAL_MODEL"


@lru_cache(maxsize=2)
def _official_guide(kind: str) -> str:
    """Load MiniMax's official guide lazily and cache it for this REPL session."""
    from huggingface_hub import hf_hub_download

    filename = BASE_GUIDE_PATH if kind == "base" else REF_GUIDE_PATH
    path = hf_hub_download(repo_id=GUIDE_REPO, filename=filename)
    return Path(path).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str | None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    if end is None:
        return text[i:]
    j = text.find(end, i + len(start))
    return text[i:] if j < 0 else text[i:j]


@lru_cache(maxsize=1)
def _base_guide_excerpt() -> str:
    """Keep only the official T2VA/shared rules that matter for this task.

    Feeding the entire guide exposed the model to I2VA/FL2VA/L2VA examples and
    caused it to invent Picture alignment instructions.  The excerpt remains
    verbatim from MiniMax's guide but excludes those irrelevant task modes.
    """
    guide = _official_guide("base")
    pieces = [
        _between(
            guide,
            "### 2.2 Part Two Contains the Three Core Fields",
            "## 3. How to Incorporate Keyframes into the Multimodal Description",
        ),
        _between(
            guide,
            "## 4. How to Write the Three Shared Core Sections",
            "## 5. Cases",
        ),
        _between(guide, "### Case 1: T2VA", "### Case 2: I2VA"),
    ]
    return "\n\n".join(p.strip() for p in pieces if p.strip())


VISUAL_OBSERVATION_REQUEST = """Analyze ONLY what is visually observable in this exact video.

Produce a conservative factual shot-by-shot observation for a later formatter. Identify the real number of shots/segments and approximate cut times, subjects and stable appearance, actions, environment, lighting, composition, framing, camera motion, and genuinely legible on-screen text. Preserve chronological order.

Critical constraints:
- You cannot hear the soundtrack. Do not infer or describe dialogue, music, ambience, sound effects, voice qualities, or other audio.
- Do not invent additional people, shots, objects, text, or events merely to make the description complete.
- Do not mention reference pictures, reference videos, Picture labels, or H3 prompt formatting.
- Distinguish visible mouth movement / apparent conversation from actual spoken words, which are unknown.
- If uncertain about a visual detail, omit it rather than guess.

Return only the visual observation, not an H3 prompt."""


def _base_format_request(observation: str) -> str:
    return f"""Convert the VERIFIED VISUAL OBSERVATION below into MiniMax H3's T2VA/base prompt format.

This is a T2VA rewrite. There are NO reference images, no reference videos, and no Picture labels. The first characters of your answer MUST be exactly:
`integrated_multimodal_description:`

Output exactly these three fields in this order and no other headings or preamble:
1. integrated_multimodal_description
2. overall_soundscape
3. non_diegetic_music

Use MiniMax's official shot/cut/camera terminology in the visual field. The first shot has no timestamp; later shots use increasing cut times only when the observation supports a real cut. Do not add people, shots, actions, text, dialogue, or events absent from the observation.

This model did NOT hear the source. Therefore:
- integrated_multimodal_description must contain visual facts only; never invent spoken words, singing, voices, sound effects, or music.
- overall_soundscape must be exactly: {AUDIO_UNKNOWN}
- non_diegetic_music must be exactly: {AUDIO_UNKNOWN}
- Do not use N/A: absence/silence was not established.

Relevant verbatim sections of MiniMaxAI/MiniMax-H3's official base guide follow. Ignore any example content; use only its formatting/terminology rules.

--- OFFICIAL T2VA/SHARED GUIDE EXCERPT START ---
{_base_guide_excerpt()}
--- OFFICIAL T2VA/SHARED GUIDE EXCERPT END ---

--- VERIFIED VISUAL OBSERVATION START ---
{observation}
--- VERIFIED VISUAL OBSERVATION END ---

Return only the three H3 fields."""


def _sanitize_visual_only_base(answer: str) -> str:
    """Enforce invariants the visual model cannot truthfully fill itself."""
    marker = "integrated_multimodal_description:"
    pos = answer.find(marker)
    if pos >= 0:
        answer = answer[pos:]

    lines = answer.strip().splitlines()
    kept: list[str] = []
    saw_soundscape = False
    saw_music = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("overall_soundscape:"):
            kept.append(f"overall_soundscape: {AUDIO_UNKNOWN}")
            saw_soundscape = True
            continue
        if stripped.startswith("non_diegetic_music:"):
            kept.append(f"non_diegetic_music: {AUDIO_UNKNOWN}")
            saw_music = True
            continue
        kept.append(line)

    if not saw_soundscape:
        kept.append(f"overall_soundscape: {AUDIO_UNKNOWN}")
    if not saw_music:
        kept.append(f"non_diegetic_music: {AUDIO_UNKNOWN}")
    return "\n".join(kept).strip()


def _h3_ref_request() -> str:
    guide = _official_guide("ref")
    return f"""You are producing a MiniMax H3 full-reference-mode rewrite.

Follow the OFFICIAL MiniMax H3 full-reference guide below. Do not invent reference assets that were not supplied. The model receiving this request can see the active video but cannot hear its soundtrack, so do NOT fabricate dialogue, singing, ambience, sound effects, or music; mark audio-only facts as `{AUDIO_UNKNOWN}`. Output only the final rewrite.

--- OFFICIAL MINIMAX H3 FULL-REFERENCE GUIDE START ---
{guide}
--- OFFICIAL MINIMAX H3 FULL-REFERENCE GUIDE END ---
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
  /h3base          two-pass official H3 T2VA/base visual rewrite
  /h3ref           official H3 full-reference-format visual rewrite
  /tokens N        change max_new_tokens
  /help            show this help
  /quit            exit

/h3base first makes a conservative visual observation, then formats that text
using only the official T2VA/shared guide sections. Audio fields are forcibly
marked AUDIO_UNAVAILABLE_FROM_VISUAL_MODEL because this Qwen3-VL graft cannot
hear the source. This also prevents I2VA/FL2VA examples from contaminating a
T2VA rewrite with invented Picture alignment instructions.

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
                parsed = shlex.split(value)
                if len(parsed) != 1:
                    raise ValueError("expected exactly one path")
                video = resolve_video(parsed[0])
                print(f"Active video: {video}")
            except Exception as exc:
                print(f"Could not set video: {exc}")
            continue

        # /h3base is deliberately a two-pass operation: video -> conservative
        # observation, then text-only formatting. This keeps the detailed guide
        # out of the visual reasoning pass and sharply reduces guide-induced
        # hallucination.
        if raw in ("/h3", "/h3base"):
            if video is None:
                print("No active video. Use /video PATH first.")
                continue
            try:
                print("[H3 mouth REPL] Pass 1/2: conservative visual observation...")
                obs_inputs = _video_inputs(
                    processor,
                    video,
                    VISUAL_OBSERVATION_REQUEST,
                    args.qwen_fps,
                    args.qwen_video_tokens,
                )
                observation = _decode_generation(
                    model,
                    processor,
                    obs_inputs,
                    max_new_tokens=min(max_tokens, 768),
                )
                print("[H3 mouth REPL] Pass 2/2: official T2VA formatting...")
                fmt_inputs = _text_inputs(processor, _base_format_request(observation))
                answer = _decode_generation(
                    model,
                    processor,
                    fmt_inputs,
                    max_new_tokens=max_tokens,
                )
                print("\n" + _sanitize_visual_only_base(answer))
            except Exception as exc:
                print(f"Generation failed: {type(exc).__name__}: {exc}")
                _cleanup_cuda()
            continue

        mode = None
        prompt = None
        if raw == "/h3ref":
            mode = "video"
            print("[H3 mouth REPL] Loading/caching official full-reference prompt guide...")
            prompt = _h3_ref_request()
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
