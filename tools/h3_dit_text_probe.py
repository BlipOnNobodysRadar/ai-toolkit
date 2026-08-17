#!/usr/bin/env python3
"""Probe whether MiniMax-H3's own DiT writes source-media information into text rows.

This is the first deliberately H3-native captioning experiment.  H3 packs
[text | audio | video] into one full-self-attention sequence.  The question is
whether the *text positions inside H3* become useful audiovisual representations
when the model is shown a real source clip.

The probe:
  1. decodes the real clip at 24 fps and encodes it with H3's video VAE;
  2. encodes the soundtrack with H3's audio VAE;
  3. encodes a short caption/query prompt with H3's own Qwen3-VL conditioner;
  4. runs the H3 DiT at a near-clean timestep and captures only the text-row
     hidden states after the token refiner and selected H3 blocks;
  5. repeats with (a) zero media and (b) the same media reversed in time;
  6. reports how strongly the text rows depend on the actual media and saves
     the captured states for the next experiment (bridging them back to a Qwen
     language-model mouth).

Nothing here claims the states are already decodable language.  The first gate
is simpler: prove that H3 itself writes media-dependent information into them,
and find which depth preserves the strongest/cleanest signal.

Example:

  venv/bin/python tools/h3_dit_text_probe.py \
    --models-path /home/blip/Desktop/AI_image_generators/ComfyUI/models \
    --video /home/blip/Downloads/AnythingElse/testClips/testClip14s.mp4 \
    --audio /home/blip/Downloads/AnythingElse/testClips/testClip14s.audio.wav \
    --query "Describe the video and audio in detail."
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.h3_caption_lab import (
    DEFAULT_ASSISTANT_LORA,
    _align_video_for_h3,
    _cleanup_cuda,
    _load_h3,
    _seed_everything,
    load_video_tensor,
)
from extensions_built_in.diffusion_models.minimax_h3.src import packing
from extensions_built_in.diffusion_models.minimax_h3.src.packing import (
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
)


DEFAULT_QUERY = "Describe the video and audio in detail."
DEFAULT_BLOCKS = "1,5,10,20,30,40,50"


def _parse_blocks(value: str, n_layers: int) -> List[int]:
    taps = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not taps:
        raise ValueError("--blocks must contain at least one block number")
    bad = [x for x in taps if x < 1 or x > n_layers]
    if bad:
        raise ValueError(f"invalid block tap(s) {bad}; model has {n_layers} blocks")
    return taps


def _load_audio_file(path: Path, max_seconds: float) -> dict:
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    max_samples = int(max_seconds * sample_rate)
    waveform = waveform[..., :max_samples]
    if waveform.numel() == 0:
        raise ValueError(f"No audio samples decoded from {path}")
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _make_h3(args):
    # Reuse the exact low-VRAM/quantized loader already proven by h3_caption_lab.
    loader_args = SimpleNamespace(
        models_path=args.models_path,
        h3_model=args.h3_model,
        partition=args.partition,
        no_assistant_lora=not args.use_assistant_lora,
        assistant_lora=args.assistant_lora,
    )
    return _load_h3(loader_args)


def _park(module) -> None:
    try:
        module.to("cpu")
    except Exception:
        pass


def _embed_query(h3, query: str):
    # On a 4090 the 32B conditioner and H3 DiT should not be resident together.
    _park(h3.model)
    _cleanup_cuda()
    pe = h3.get_prompt_embeds(query)
    _park(h3.text_encoder)
    _cleanup_cuda()

    emb = pe.text_embeds[0].detach().to("cpu", torch.float32)
    tags = pe.text_token_tags[0].detach().to("cpu", torch.long)

    token_ids = h3.tokenizer(query.strip(), add_special_tokens=False)["input_ids"]
    if h3.max_text_length and h3.max_text_length > 0:
        token_ids = token_ids[: h3.max_text_length]
    token_strings = h3.tokenizer.convert_ids_to_tokens(token_ids)
    # Empty-prompt fallback can make these lengths differ; ordinary queries should match.
    if len(token_strings) != emb.shape[0]:
        token_strings = [f"token_{i}" for i in range(emb.shape[0])]
    return emb, tags, token_strings


def _reverse_audio_rows(rows: torch.Tensor) -> torch.Tensor:
    """Reverse time independently in each stereo channel, preserving row layout."""
    b, n, c = rows.shape
    if n % packing.AUDIO_CHANNELS:
        raise ValueError(f"audio row count {n} is not divisible by stereo channel count")
    t = n // packing.AUDIO_CHANNELS
    return (
        rows.reshape(b, packing.AUDIO_CHANNELS, t, c)
        .flip(2)
        .reshape(b, n, c)
        .contiguous()
    )


def _matched_noise(shape, seed: int, device: torch.device) -> torch.Tensor:
    # CPU generator makes the seed stable independent of CUDA RNG state.
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device)


def _prepare_media_variants(
    *,
    video_latents: torch.Tensor,
    audio_rows_clean: torch.Tensor,
    t_video: float,
    seed: int,
    device: torch.device,
):
    """Return real/zero/reversed noisy video latents and packed audio rows.

    All three variants use identical noise.  At --t 0.999 they are essentially
    clean inputs but remain on the same flow parameterization H3 was trained on.
    """
    sigma_v = 1.0 - float(t_video)
    sigma_v_t = torch.tensor([sigma_v], device=device, dtype=torch.float32)
    sigma_a = float(packing.remap_sigma(sigma_v_t)[0].item())
    t_audio = 1.0 - sigma_a

    v = video_latents.to(device, torch.float32)
    a = audio_rows_clean.to(device, torch.float32)

    v_noise = _matched_noise(v.shape, seed, device)
    a_noise = _matched_noise(a.shape, seed ^ 0x5A17C9E3, device)

    v_rev = v.flip(2)
    a_rev = _reverse_audio_rows(a)

    variants = {
        "real": (
            t_video * v + sigma_v * v_noise,
            t_audio * a + sigma_a * a_noise,
        ),
        "zero": (
            sigma_v * v_noise,
            sigma_a * a_noise,
        ),
        "reversed": (
            t_video * v_rev + sigma_v * v_noise,
            t_audio * a_rev + sigma_a * a_noise,
        ),
    }
    return variants, t_audio


def _build_layout_and_rows(
    *,
    h3,
    text_tags: torch.Tensor,
    video_noisy: torch.Tensor,
    audio_noisy: torch.Tensor,
    t_video: float,
    t_audio: float,
):
    b, _, t_lat, h_lat, w_lat = video_noisy.shape
    if b != 1:
        raise ValueError("This research probe currently expects batch size 1")

    num_frames = (t_lat - 2) // 5 * 17 + 5 if t_lat > 1 else 1
    a_lat = packing.audio_latent_num_frames(num_frames)
    expected_audio_rows = a_lat * packing.AUDIO_CHANNELS
    if audio_noisy.shape[1] > expected_audio_rows:
        audio_noisy = audio_noisy[:, :expected_audio_rows]
    elif audio_noisy.shape[1] < expected_audio_rows:
        audio_noisy = torch.nn.functional.pad(
            audio_noisy, (0, 0, 0, expected_audio_rows - audio_noisy.shape[1])
        )

    layout = build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=t_lat,
        latent_height=h_lat,
        latent_width=w_lat,
        num_audio_latents=a_lat,
        keyframe_anchors=(),
    )
    row_t = build_row_timesteps(layout, t_video, t_audio).unsqueeze(0)
    video_rows = patchify_video_latents(video_noisy)
    return layout, row_t, video_rows, audio_noisy


def _capture_text_states(
    *,
    transformer,
    query_emb: torch.Tensor,
    layout,
    row_t: torch.Tensor,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    taps: List[int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Replicate MiniMaxH3Transformer.forward, retaining text positions."""
    text_idx = layout.text_indices.to(device)
    video_idx = layout.video_indices.to(device)
    audio_idx = layout.audio_indices.to(device)
    token_tags = layout.token_tags.unsqueeze(0).to(device)
    position_ids = layout.position_ids.unsqueeze(0).to(device)
    row_t = row_t.to(device)

    q = query_emb.unsqueeze(0).to(device, transformer.dtype)
    v = video_rows.to(device)
    a = audio_rows.to(device)

    rotary_emb = transformer.rope(position_ids)
    video_embeds = transformer.video_patch_proj(
        v.to(transformer.video_patch_proj.weight.dtype)
    )
    audio_embeds = transformer.audio_patch_proj(
        a.to(transformer.audio_patch_proj.weight.dtype)
    )
    projected_text = transformer.condition_proj(q)
    refined_text = transformer.token_refiner(projected_text, None)

    x = refined_text.new_zeros((1, layout.sequence_length, refined_text.shape[-1]))
    x = x.index_copy(1, text_idx, refined_text)
    x = x.index_copy(1, video_idx, video_embeds.to(x.dtype))
    x = x.index_copy(1, audio_idx, audio_embeds.to(x.dtype))

    unique_t, inverse = torch.unique(row_t.to(torch.float32), sorted=True, return_inverse=True)
    temb = transformer._time_embedding(unique_t)
    adaln_indices = inverse * 3 + token_tags.clamp(min=0)

    out: Dict[str, torch.Tensor] = {
        "projected": projected_text.detach().float().cpu(),
        "refiner": refined_text.detach().float().cpu(),
    }

    with torch.inference_mode():
        for i, block in enumerate(transformer.blocks, start=1):
            x = block(x, temb, adaln_indices, rotary_emb, None)
            if i in taps:
                out[f"block_{i}"] = x.index_select(1, text_idx).detach().float().cpu()
    return out


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float().square())).item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return float("nan")
    return float((torch.dot(a, b) / denom).item())


def _per_token_rms(delta: torch.Tensor) -> torch.Tensor:
    # delta (1,L,D) -> (L,)
    return torch.sqrt(torch.mean(delta[0].float().square(), dim=-1))


def _report(captures, token_strings: List[str], top_tokens: int) -> None:
    real = captures["real"]
    zero = captures["zero"]
    rev = captures["reversed"]
    ref = real["refiner"]

    print("\n=== H3 TEXT-ROW MEDIA PROBE ===")
    print(
        "tap          state_RMS   Δfrom_refiner   real-vs-zero_RMS   real-vs-zero_cos   "
        "real-vs-reversed_RMS   real-vs-reversed_cos"
    )
    print("-" * 125)
    for key in real:
        r = real[key]
        print(
            f"{key:12s} "
            f"{_rms(r):10.5f} "
            f"{_rms(r - ref):15.6f} "
            f"{_rms(r - zero[key]):17.6f} "
            f"{_cos(r, zero[key]):18.8f} "
            f"{_rms(r - rev[key]):21.6f} "
            f"{_cos(r, rev[key]):22.8f}"
        )

    # Pick the requested deepest tap and show which query-token positions H3
    # changed most when chronology/alignment was reversed.
    block_keys = [k for k in real if k.startswith("block_")]
    detail_key = block_keys[-1] if block_keys else "refiner"
    scores = _per_token_rms(real[detail_key] - rev[detail_key])
    order = torch.argsort(scores, descending=True)[: max(1, top_tokens)]
    print(f"\nLargest real-vs-reversed token-state changes at {detail_key}:")
    for idx in order.tolist():
        token = token_strings[idx] if idx < len(token_strings) else f"token_{idx}"
        print(f"  {idx:>3}  rms={scores[idx].item():.6f}  {token!r}")

    print(
        "\nWhat matters first:\n"
        "  • projected and refiner should be ~identical across controls; they have not seen media yet.\n"
        "  • block_1+ should diverge if full self-attention writes audiovisual information into text rows.\n"
        "  • real-vs-reversed is the stronger control: it preserves the same media content/distribution but\n"
        "    destroys temporal order and A/V alignment. A nontrivial gap means the H3 text states care about\n"
        "    more than merely 'some video/audio tensors are present'.\n"
        "  • If that signal survives at one or more depths, the next experiment is to learn/derive a bridge\n"
        "    from those H3 text states back into the restored Qwen mouth and see whether words come out.\n"
    )


def _default_output(video: Path) -> Path:
    return video.with_suffix(video.suffix + ".h3textprobe.pt")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-path", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--audio", required=True, help="Extracted soundtrack (WAV is ideal).")
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--h3-model", default="Comfy-Org/MiniMax-H3")
    p.add_argument(
        "--partition",
        default="fl2va_pruned",
        choices=("fl2va", "fl2va_pruned", "ref2va", "ref2va_pruned"),
    )
    p.add_argument("--assistant-lora", default=DEFAULT_ASSISTANT_LORA)
    p.add_argument(
        "--use-assistant-lora",
        action="store_true",
        help="Probe with the training/de-distillation assistant active. Default is released base H3 only.",
    )
    p.add_argument("--blocks", default=DEFAULT_BLOCKS)
    p.add_argument(
        "--t",
        type=float,
        default=0.999,
        help="H3 t in [0,1], where 1 is clean. Default mirrors near-clean reference conditioning.",
    )
    p.add_argument("--max-seconds", type=float, default=15.0)
    p.add_argument(
        "--max-edge",
        type=int,
        default=256,
        help="Downscale source for this probe; 256 keeps the 50-block experiment manageable on 24 GB.",
    )
    p.add_argument("--latent-seed", type=int, default=1701)
    p.add_argument("--noise-seed", type=int, default=1776)
    p.add_argument("--top-tokens", type=int, default=12)
    p.add_argument("--output")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not 0.0 <= args.t <= 1.0:
        p.error("--t must be in [0,1]")
    if args.max_edge < 32:
        p.error("--max-edge must be >= 32")

    video_path = Path(args.video).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    if not video_path.is_file():
        p.error(f"video not found: {video_path}")
    if not audio_path.is_file():
        p.error(f"audio not found: {audio_path}")

    output = Path(args.output).expanduser().resolve() if args.output else _default_output(video_path)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite or choose --output")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    print("[H3 text probe] Decoding source clip")
    video = load_video_tensor(
        video_path,
        fps=24.0,
        max_seconds=args.max_seconds,
        max_edge=args.max_edge,
    )
    video = _align_video_for_h3(video)
    num_frames = int(video.shape[0])
    duration = num_frames / 24.0
    print(
        f"[H3 text probe] Using {num_frames} aligned frames "
        f"({duration:.3f}s) at {video.shape[-1]}x{video.shape[-2]}"
    )

    print("[H3 text probe] Loading H3")
    h3 = _make_h3(args)
    transformer = h3.model
    taps = _parse_blocks(args.blocks, len(transformer.blocks))

    print("[H3 text probe] Encoding video with H3 VAE")
    _seed_everything(args.latent_seed)
    with torch.inference_mode():
        video_latents = h3.encode_images([video]).detach().to("cpu", torch.float32)
    del video

    print(f"[H3 text probe] Encoding soundtrack with H3 audio VAE: {audio_path}")
    audio_data = _load_audio_file(audio_path, duration)
    _seed_everything(args.latent_seed ^ 0x31A9B7)
    with torch.inference_mode():
        audio_rows_clean = h3.encode_audio([audio_data]).detach().to("cpu", torch.float32)
    del audio_data

    _park(h3.vae)
    _cleanup_cuda()

    print(f"[H3 text probe] Encoding query with H3 Qwen conditioner: {args.query!r}")
    query_emb, text_tags, token_strings = _embed_query(h3, args.query)

    print(
        f"[H3 text probe] video latents={tuple(video_latents.shape)}; "
        f"audio rows={tuple(audio_rows_clean.shape)}; text tokens={query_emb.shape[0]}"
    )

    # DiT phase: text encoder and VAEs are parked; bring the H3 transformer in.
    _cleanup_cuda()
    if transformer.device == torch.device("cpu"):
        transformer.to(device)

    variants, t_audio = _prepare_media_variants(
        video_latents=video_latents,
        audio_rows_clean=audio_rows_clean,
        t_video=args.t,
        seed=args.noise_seed,
        device=device,
    )
    print(
        f"[H3 text probe] Running taps {taps} at video t={args.t:.6f}, "
        f"audio t={t_audio:.6f}"
    )

    captures: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in ("real", "zero", "reversed"):
        print(f"[H3 text probe]   {name} media pass")
        v_noisy, a_noisy = variants[name]
        layout, row_t, video_rows, audio_rows = _build_layout_and_rows(
            h3=h3,
            text_tags=text_tags,
            video_noisy=v_noisy,
            audio_noisy=a_noisy,
            t_video=args.t,
            t_audio=t_audio,
        )
        captures[name] = _capture_text_states(
            transformer=transformer,
            query_emb=query_emb,
            layout=layout,
            row_t=row_t,
            video_rows=video_rows,
            audio_rows=audio_rows,
            taps=taps,
            device=device,
        )
        del v_noisy, a_noisy, video_rows, audio_rows
        torch.cuda.empty_cache()

    _report(captures, token_strings, args.top_tokens)

    payload = {
        "version": 1,
        "video": str(video_path),
        "audio": str(audio_path),
        "query": args.query,
        "query_tokens": token_strings,
        "num_frames": num_frames,
        "duration_seconds": duration,
        "max_edge": args.max_edge,
        "partition": args.partition,
        "assistant_lora_active": bool(args.use_assistant_lora),
        "video_t": float(args.t),
        "audio_t": float(t_audio),
        "blocks": taps,
        "captures": captures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"[H3 text probe] Saved captures for the Qwen-mouth bridge: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
