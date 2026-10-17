#!/usr/bin/env python3
"""Experimental "mouth graft" for the MiniMax-H3 Qwen3-VL conditioner.

This is deliberately a probe, not a production captioner.

MiniMax-H3 consumes the unnormalized output after Qwen3-VL decoder layer 49,
so the H3 conditioner shipped in the Comfy repack contains the vision tower,
embeddings, and decoder layers 0..49 but omits layers 50..63, the final norm,
and the LM head.

This tool asks a narrow question: can we transplant H3's lower Qwen3-VL stack
into a complete Qwen3-VL-32B language model while retaining a stock upper tail
and recover coherent text/video generation?

The default stock donor is the already-small pre-quantized Unsloth checkpoint.
Its vision path currently fails under the pinned BnB stack, but this experiment
throws that vision tower and its lower 50 language layers away before the final
hybrid runs. Only the stock layers 50..63 + final norm + LM head survive.

If the graft produces coherent text and video captions, it shows that H3's
conditioner can still interface with a language-generation tail. It does NOT by
itself prove that the H3 DiT has been inverted; that is a later experiment.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_STOCK_MODEL = "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit"
DEFAULT_H3_MODEL = "Comfy-Org/MiniMax-H3"
LOWER_LAYER_COUNT = 50


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _cuda_mem(label: str) -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    used = total - free
    gib = 1024**3
    print(
        f"[H3 mouth] {label}: CUDA {used / gib:.2f} GiB used, "
        f"{free / gib:.2f} GiB free / {total / gib:.2f} GiB"
    )


def _load_stock(model_id: str):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    print(f"[H3 mouth] Loading stock donor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")
    model.eval()
    _cuda_mem("after stock donor load")
    return model, processor


def _load_h3_conditioner(models_path: str, h3_model: str):
    """Load ONLY H3's Qwen3-VL conditioner, not its 33B DiT or VAEs."""
    os.environ["MODELS_PATH"] = str(Path(models_path).expanduser().resolve())

    from toolkit.config_modules import ModelConfig
    from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model

    cfg = ModelConfig(
        name_or_path=h3_model,
        arch="minimax_h3",
        dtype="bf16",
        vae_dtype="bf16",
        te_dtype="bf16",
        quantize=False,
        quantize_te=True,
        qtype_te="nvfp4",
        low_vram=True,
        layer_offloading=False,
        model_kwargs={"partition": "fl2va_pruned"},
    )
    holder = MinimaxH3Model(device="cuda", model_config=cfg, dtype="bf16")
    print("[H3 mouth] Loading H3 Qwen3-VL conditioner only")
    tokenizer, processor, text_encoder = holder._load_text_encoder()
    text_encoder.eval()
    _cuda_mem("after H3 conditioner load (still CPU)")
    return holder, tokenizer, processor, text_encoder


def _field(config, name):
    return getattr(config, name, None)


def _check_compatibility(stock, h3_te, stock_processor, h3_tokenizer) -> None:
    stock_text = stock.config.text_config
    h3_text = h3_te.config.text_config

    fields = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rms_norm_eps",
    )
    mismatches = []
    for name in fields:
        a, b = _field(stock_text, name), _field(h3_text, name)
        if a != b:
            mismatches.append(f"text_config.{name}: stock={a!r}, h3={b!r}")

    stock_layers = len(stock.model.language_model.layers)
    h3_layers = len(h3_te.model.language_model.layers)
    if stock_layers < 64:
        mismatches.append(f"stock decoder has only {stock_layers} layers; expected >=64")
    if h3_layers != LOWER_LAYER_COUNT:
        mismatches.append(
            f"H3 conditioner has {h3_layers} decoder layers; expected {LOWER_LAYER_COUNT}"
        )

    stock_vocab = getattr(stock_processor.tokenizer, "vocab_size", None)
    h3_vocab = getattr(h3_tokenizer, "vocab_size", None)
    if stock_vocab != h3_vocab:
        mismatches.append(f"tokenizer vocab_size: stock={stock_vocab}, h3={h3_vocab}")

    # These are the special ids that matter most for multimodal placeholder
    # insertion and generation. Missing attrs are ignored rather than guessed.
    for name in (
        "pad_token_id",
        "eos_token_id",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        a = getattr(stock.config, name, None)
        b = getattr(h3_te.config, name, None)
        if a is not None and b is not None and a != b:
            mismatches.append(f"config.{name}: stock={a!r}, h3={b!r}")

    if mismatches:
        print("[H3 mouth] Structural compatibility check FAILED:")
        for item in mismatches:
            print(f"  - {item}")
        raise RuntimeError("H3 and stock Qwen3-VL stacks are not structurally compatible")

    print(
        "[H3 mouth] Structural compatibility check passed: "
        f"H3 layers 0..{LOWER_LAYER_COUNT - 1} -> stock layers "
        f"{LOWER_LAYER_COUNT}..{stock_layers - 1}"
    )


def _text_inputs(processor, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def _video_inputs(processor, video_path: Path, prompt: str, fps: float, video_tokens: int):
    # Same visual-budget convention used by h3_caption_lab. Qwen's processor
    # expresses these limits as pixel-area bounds rather than literal edge px.
    if getattr(processor, "video_processor", None) is not None:
        token_unit = 32 * 32 * 2
        processor.video_processor.size = {
            "longest_edge": max(256, video_tokens) * token_unit,
            "shortest_edge": min(256, video_tokens) * token_unit,
        }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": str(video_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Transformers 5.5 currently warns that fps should live in
    # processor_kwargs; this form is already known to work in the caption lab.
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    )


def _decode_generation(model, processor, inputs, max_new_tokens: int) -> str:
    device = torch.device("cuda:0")
    inputs = inputs.to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_len = int(inputs["input_ids"].shape[-1])
    return processor.decode(
        generated[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _stock_text_baseline(model, processor, prompt: str, max_new_tokens: int) -> None:
    print("\n[H3 mouth] === STOCK TEXT BASELINE ===")
    try:
        text = _decode_generation(
            model,
            processor,
            _text_inputs(processor, prompt),
            max_new_tokens=max_new_tokens,
        )
        print(text or "<empty output>")
    except Exception as exc:
        # The donor's broken lower/vision BnB modules are about to be discarded,
        # so a baseline failure is diagnostic but not automatically fatal.
        print(f"[H3 mouth] Stock baseline failed: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=8)
    _cleanup_cuda()


def _discard_stock_lower(model) -> None:
    """Free the donor pieces that H3 will replace while retaining its tail."""
    import torch.nn as nn

    print(
        f"[H3 mouth] Discarding stock vision + embeddings + decoder layers "
        f"0..{LOWER_LAYER_COUNT - 1}; retaining stock language tail/norm/lm_head"
    )
    tail = list(model.model.language_model.layers[LOWER_LAYER_COUNT:])
    model.model.language_model.layers = nn.ModuleList(
        [nn.Identity() for _ in range(LOWER_LAYER_COUNT)] + tail
    )
    model.model.visual = nn.Identity()
    model.model.language_model.embed_tokens = nn.Identity()
    _cleanup_cuda()
    _cuda_mem("after discarding donor lower stack")


def _graft_h3_lower(model, h3_te) -> None:
    print("[H3 mouth] Grafting H3 vision + embeddings + decoder layers 0..49")
    model.model.visual = h3_te.model.visual
    model.model.language_model.embed_tokens = h3_te.model.language_model.embed_tokens
    for idx in range(LOWER_LAYER_COUNT):
        model.model.language_model.layers[idx] = h3_te.model.language_model.layers[idx]

    # Do NOT transplant H3's final norm: its loader intentionally replaces that
    # norm with Identity because H3 consumes the unnormalized layer-49 state.
    # The donor's real stock final norm and lm_head are precisely the "mouth".

    # h3_te itself is now only an extra owner of the same transplanted modules.
    # Deleting the wrapper does not delete the modules because model owns them.
    _cleanup_cuda()


def _move_graft_to_cuda(model) -> None:
    print("[H3 mouth] Moving transplanted H3 lower stack to CUDA")
    try:
        model.model.visual.to("cuda")
        model.model.language_model.embed_tokens.to("cuda")
        for idx in range(LOWER_LAYER_COUNT):
            model.model.language_model.layers[idx].to("cuda")
    except torch.OutOfMemoryError:
        _cuda_mem("OOM while moving H3 graft")
        raise RuntimeError(
            "The all-resident mouth graft does not fit this GPU. "
            "Do not change packages yet; the next probe should add streamed/offloaded lower layers."
        )
    model.eval()
    _cleanup_cuda()
    _cuda_mem("after completed mouth graft")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Restore a stock Qwen3-VL language tail onto H3's truncated conditioner."
    )
    p.add_argument("--models-path", required=True, help="ComfyUI models root containing H3 weights")
    p.add_argument("--video", help="Optional local video for the video-caption half of the probe")
    p.add_argument("--stock-model", default=DEFAULT_STOCK_MODEL)
    p.add_argument("--h3-model", default=DEFAULT_H3_MODEL)
    p.add_argument(
        "--text-prompt",
        default=(
            "Answer in one short sentence: what is the capital of France, and what famous "
            "landmark is strongly associated with that city?"
        ),
    )
    p.add_argument(
        "--video-prompt",
        default=(
            "Describe this exact video as a detailed generation prompt. Preserve temporal order, "
            "subjects, actions, setting, lighting, framing, camera movement, and scene changes. "
            "Do not invent audio you cannot hear."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--text-max-new-tokens", type=int, default=48)
    p.add_argument("--qwen-fps", type=float, default=2.0)
    p.add_argument("--qwen-video-tokens", type=int, default=768)
    p.add_argument("--skip-stock-baseline", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA")

    video_path = None
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

    stock, processor = _load_stock(args.stock_model)

    if not args.skip_stock_baseline:
        _stock_text_baseline(
            stock,
            processor,
            args.text_prompt,
            max_new_tokens=args.text_max_new_tokens,
        )

    # Free the broken/unneeded donor lower stack before loading H3's replacement,
    # which keeps peak VRAM close to one quantized 32B model rather than two.
    _discard_stock_lower(stock)

    holder, h3_tokenizer, _h3_processor, h3_te = _load_h3_conditioner(
        args.models_path,
        args.h3_model,
    )
    _check_compatibility(stock, h3_te, processor, h3_tokenizer)
    _graft_h3_lower(stock, h3_te)

    # Drop wrapper-only references. The transplanted modules remain owned by stock.
    del h3_te, _h3_processor, h3_tokenizer, holder
    _cleanup_cuda()

    _move_graft_to_cuda(stock)

    print("\n[H3 mouth] === H3-GRAFT TEXT SELF-TEST ===")
    text = _decode_generation(
        stock,
        processor,
        _text_inputs(processor, args.text_prompt),
        max_new_tokens=args.text_max_new_tokens,
    )
    print(text or "<empty output>")

    if video_path is not None:
        print("\n[H3 mouth] === H3-GRAFT VIDEO CAPTION ===")
        inputs = _video_inputs(
            processor,
            video_path,
            args.video_prompt,
            fps=args.qwen_fps,
            video_tokens=args.qwen_video_tokens,
        )
        caption = _decode_generation(
            stock,
            processor,
            inputs,
            max_new_tokens=args.max_new_tokens,
        )
        print(caption or "<empty output>")

    print("\n[H3 mouth] Probe complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
