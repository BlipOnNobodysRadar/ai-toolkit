#!/usr/bin/env python3
"""Probe whether MiniMax-H3's own multimodal DiT writes source information into text rows.

This is deliberately a *research probe*, not a production captioner.

Experiment:
  1. Encode a real source video (+ optional audio) with H3's VAEs.
  2. Encode a short text query with H3's Qwen3-VL conditioner.
  3. Run the H3 DiT on the clean/near-clean source latents.
  4. Capture the internal hidden states at the text-token positions after the
     token refiner and after selected DiT blocks.
  5. Map those 5376-d H3 states back toward the 5120-d Qwen conditioner space
     using a least-squares pseudoinverse of H3's condition_proj.
  6. Report diagnostics showing whether the recovered states are media-dependent
     and how far they remain from the condition-projection subspace.

A later step can feed promising recovered Qwen-space states into the restored
Qwen3-VL upper layers / LM head.  This first probe intentionally stops before
that decoder graft so we can establish whether H3 itself is putting useful
media-dependent signal into its text rows.

Run from the ai-toolkit repo root, for example:

  venv/bin/python tools/h3_dit_text_probe.py \
    --models-path /home/blip/Desktop/AI_image_generators/ComfyUI/models \
    --video /home/blip/Downloads/AnythingElse/testClips/testClip14s.mp4 \
    --audio /home/blip/Downloads/AnythingElse/testClips/testClip14s.audio.wav \
    --query "Describe the video and audio in detail." \
    --blocks 1,5,10,20,30,40,50

The script prints, per tap:
  - change from the initial projected query state
  - reconstruction error after projecting back through pinv(condition_proj)
  - cosine similarity between reconstructed and observed H3 text states
  - norm of the recovered 5120-d Qwen-space state

It also runs a zero-media control with the same query and reports the cosine /
RMS difference between real-media and zero-media text states.  A robust media
signal at intermediate H3 depths is the key result we are looking for.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.config_modules import ModelConfig
from toolkit.paths import MODELS_PATH
from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.src import packing
from extensions_built_in.diffusion_models.minimax_h3.src.packing import (
    AUDIO_LATENTS_PER_SECOND,
    AUDIO_SAMPLE_RATE,
    FPS,
    build_packed_sequence,
    build_row_timesteps,
    pack_audio_latents,
    patchify_video_latents,
    remap_sigma,
)
from extensions_built_in.diffusion_models.minimax_h3.src.text_encoder import (
    encode_minimax_h3_prompt,
)


@dataclass
class ProbeInputs:
    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    text_states: torch.Tensor
    text_token_tags: torch.Tensor
    layout: object
    row_timesteps: torch.Tensor


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")
    return torch.device("cuda")


def _parse_blocks(value: str, n_layers: int) -> List[int]:
    out: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        i = int(part)
        if i < 1 or i > n_layers:
            raise ValueError(f"block tap {i} must be in 1..{n_layers}")
        out.append(i)
    return sorted(set(out))


def _find_model_loader(models_path: str, partition: str) -> MinimaxH3Model:
    # ai-toolkit's H3 loader resolves weights underneath MODELS_PATH.  Override
    # the module-level value for this standalone tool before constructing it.
    import extensions_built_in.diffusion_models.minimax_h3.minimax_h3 as h3mod

    h3mod.MODELS_PATH = models_path

    cfg = ModelConfig(
        name_or_path="Comfy-Org/MiniMax-H3",
        arch="minimax_h3",
        model_kwargs={
            "partition": partition,
            # Keep this probe explicit; the caller can still point MODELS_PATH
            # at the normal ComfyUI model tree.
            "max_text_length": 512,
        },
    )
    model = MinimaxH3Model(device=str(_device()), model_config=cfg, dtype="bf16")
    model.load_model()
    model.eval()
    return model


def _load_video_tensor(path: Path, max_frames: Optional[int] = None) -> torch.Tensor:
    """Return uint8-ish video tensor as (T,H,W,C) via torchvision.io.read_video.

    We use torchvision here because ai-toolkit already depends on torch/vision,
    and this is only a probe.  The model's VAE preprocessing remains responsible
    for the exact resize/normalization expected by H3.
    """
    try:
        from torchvision.io import read_video
    except Exception as e:  # pragma: no cover - environment-dependent
        raise RuntimeError("torchvision video IO is required for this probe") from e

    frames, _, info = read_video(str(path), pts_unit="sec", output_format="THWC")
    if frames.numel() == 0:
        raise RuntimeError(f"no frames decoded from {path}")

    # H3 uses 24 fps.  If the source differs, nearest-neighbour temporal sample
    # to approximately 24 fps, then snap DOWN to the VAE's 17n+5 grid.
    src_fps = float(info.get("video_fps", FPS) or FPS)
    if src_fps > 0 and abs(src_fps - FPS) > 1e-3:
        target_n = max(1, round(frames.shape[0] * FPS / src_fps))
        idx = torch.linspace(0, frames.shape[0] - 1, target_n).round().long()
        frames = frames.index_select(0, idx)

    if max_frames is not None:
        frames = frames[:max_frames]

    n = packing.align_num_frames_down(int(frames.shape[0]))
    frames = frames[:n]
    return frames


def _load_audio_wave(path: Path) -> Tuple[torch.Tensor, int]:
    try:
        import torchaudio
    except Exception as e:  # pragma: no cover - environment-dependent
        raise RuntimeError("torchaudio is required when --audio is supplied") from e

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != AUDIO_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, AUDIO_SAMPLE_RATE)
        sr = AUDIO_SAMPLE_RATE
    return wav, sr


def _first_callable(obj, names: Iterable[str]):
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    return None


def _encode_video_latents(model: MinimaxH3Model, frames: torch.Tensor) -> torch.Tensor:
    """Best-effort adapter around the H3 video VAE's encode API."""
    vae = model.video_vae
    # Most ai-toolkit video VAEs expose encode_video or encode.  Keep the probe
    # tolerant to the exact wrapper signature used by the local checkout.
    fn = _first_callable(vae, ("encode_video", "encode"))
    if fn is None:
        raise RuntimeError("could not find video VAE encode method")

    # Convert THWC uint8 -> BCTHW float [0,1].  The H3 VAE wrapper is expected
    # to perform its own [-1,1] normalization if required.
    x = frames.to(torch.float32).permute(0, 3, 1, 2) / 255.0
    x = x.permute(1, 0, 2, 3).unsqueeze(0).to(_device())

    with torch.inference_mode():
        out = fn(x)
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("latents", "latent", "sample"):
        value = getattr(out, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise RuntimeError(f"unrecognized video VAE encode return type: {type(out)}")


def _encode_audio_latents(model: MinimaxH3Model, wav: torch.Tensor) -> torch.Tensor:
    vae = model.audio_vae
    fn = _first_callable(vae, ("encode_audio", "encode"))
    if fn is None:
        raise RuntimeError("could not find audio VAE encode method")
    x = wav.unsqueeze(0).to(_device())  # (B,2,T)
    with torch.inference_mode():
        out = fn(x)
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("latents", "latent", "sample"):
        value = getattr(out, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise RuntimeError(f"unrecognized audio VAE encode return type: {type(out)}")


def _encode_text(model: MinimaxH3Model, query: str):
    # Use the same helper the H3 generation/training path uses.  Depending on
    # local helper version this returns either (embeds, mask/tags) or an object.
    out = encode_minimax_h3_prompt(
        tokenizer=model.tokenizer,
        processor=model.processor,
        text_encoder=model.text_encoder,
        prompt=query,
        images=None,
        max_text_length=model.max_text_length,
        device=_device(),
    )

    if isinstance(out, tuple):
        embeds = out[0]
        tags = None
        for x in out[1:]:
            if isinstance(x, torch.Tensor) and x.ndim <= 2 and x.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.bool,
            ):
                tags = x
                break
        if tags is None:
            tags = torch.full((embeds.shape[1],), packing.TEXT_TAG, dtype=torch.long)
        if tags.ndim == 2:
            tags = tags[0]
        return embeds, tags.to(torch.long)

    embeds = getattr(out, "prompt_embeds", None) or getattr(out, "embeds", None)
    if embeds is None:
        raise RuntimeError(f"unrecognized text encoder return type: {type(out)}")
    tags = getattr(out, "token_tags", None)
    if tags is None:
        tags = torch.full((embeds.shape[1],), packing.TEXT_TAG, dtype=torch.long)
    if tags.ndim == 2:
        tags = tags[0]
    return embeds, tags.to(torch.long)


def _prepare_inputs(
    model: MinimaxH3Model,
    query: str,
    video_latents: torch.Tensor,
    audio_latents: Optional[torch.Tensor],
    t_video: float,
) -> ProbeInputs:
    text_states, text_tags = _encode_text(model, query)
    text_states = text_states.to(_device())

    video_rows = patchify_video_latents(video_latents).to(_device())

    if audio_latents is None:
        # Build a correctly shaped, silent placeholder.  The actual count must
        # track video duration because the packed RoPE clock aligns them.
        n_video_lat = int(video_latents.shape[2])
        n_audio_lat = round(n_video_lat / max(1, n_video_lat) * 1)  # replaced below
        # Derive from original pixel-frame count using inverse 17n+5 relation.
        n = (n_video_lat - 2) // 5
        pixel_frames = 17 * n + 5
        n_audio_lat = packing.audio_latent_num_frames(pixel_frames)
        audio_rows = torch.zeros(
            (video_rows.shape[0], 2 * n_audio_lat, 32),
            device=_device(),
            dtype=video_rows.dtype,
        )
    else:
        audio_rows = pack_audio_latents(audio_latents).to(_device())
        n_audio_lat = int(audio_latents.shape[-1])

    n_video_lat = int(video_latents.shape[2])
    latent_h = int(video_latents.shape[3])
    latent_w = int(video_latents.shape[4])

    layout = build_packed_sequence(
        text_tags.detach().cpu(),
        num_latent_frames=n_video_lat,
        latent_height=latent_h,
        latent_width=latent_w,
        num_audio_latents=n_audio_lat,
        patch_size=model.transformer.params.patch_size,
        keyframe_anchors=(),
    )

    sigma_v = 1.0 - float(t_video)
    sigma_a = remap_sigma(sigma_v, packing.VIDEO_SIGMA_SHIFT, packing.AUDIO_SIGMA_SHIFT)
    t_audio = 1.0 - float(sigma_a)
    row_t = build_row_timesteps(layout, t_video, t_audio).unsqueeze(0).to(_device())

    return ProbeInputs(
        video_rows=video_rows,
        audio_rows=audio_rows,
        text_states=text_states,
        text_token_tags=text_tags,
        layout=layout,
        row_timesteps=row_t,
    )


def _run_capture(
    transformer,
    inp: ProbeInputs,
    taps: List[int],
) -> Dict[str, torch.Tensor]:
    """Replicate transformer.forward while retaining text-row states."""
    layout = inp.layout
    text_idx = layout.text_indices.to(_device())
    video_idx = layout.video_indices.to(_device())
    audio_idx = layout.audio_indices.to(_device())
    token_tags = layout.token_tags.unsqueeze(0).to(_device())
    pos = layout.position_ids.unsqueeze(0).to(_device())

    rotary_emb = transformer.rope(pos)
    video_embeds = transformer.video_patch_proj(
        inp.video_rows.to(transformer.video_patch_proj.weight.dtype)
    )
    audio_embeds = transformer.audio_patch_proj(
        inp.audio_rows.to(transformer.audio_patch_proj.weight.dtype)
    )
    projected_text = transformer.condition_proj(inp.text_states.to(transformer.dtype))

    text_embeds = transformer.token_refiner(projected_text, None)
    x = text_embeds.new_zeros((1, layout.sequence_length, text_embeds.shape[-1]))
    x = x.index_copy(1, text_idx, text_embeds)
    x = x.index_copy(1, video_idx, video_embeds.to(x.dtype))
    x = x.index_copy(1, audio_idx, audio_embeds.to(x.dtype))

    unique_t, inverse = torch.unique(
        inp.row_timesteps.to(torch.float32), sorted=True, return_inverse=True
    )
    temb = transformer._time_embedding(unique_t)
    adaln_indices = inverse * 3 + token_tags.clamp(min=0)

    captures: Dict[str, torch.Tensor] = {
        "projected": projected_text.detach().float().cpu(),
        "refiner": x.index_select(1, text_idx).detach().float().cpu(),
    }

    with torch.inference_mode():
        for i, block in enumerate(transformer.blocks, start=1):
            x = block(x, temb, adaln_indices, rotary_emb, None)
            if i in taps:
                captures[f"block_{i}"] = (
                    x.index_select(1, text_idx).detach().float().cpu()
                )

    return captures


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float() ** 2)).item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(den) == 0.0:
        return float("nan")
    return float(torch.dot(a, b).div(den).item())


def _condition_pinv(transformer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (W_pinv, bias) for y = x @ W.T + b.

    condition_proj.weight is (5376,5120).  pinv(W.T) maps centered 5376-d
    outputs back to the minimum-norm 5120-d least-squares input.
    """
    layer = transformer.condition_proj
    Wt = layer.weight.detach().float().cpu().T  # (5120,5376)
    bias = layer.bias.detach().float().cpu() if layer.bias is not None else torch.zeros(layer.out_features)
    print("Computing float32 pseudoinverse of condition_proj (one-time CPU cost)...")
    pinv = torch.linalg.pinv(Wt)  # (5376,5120)
    return pinv, bias


def _inverse_project(y: torch.Tensor, pinv: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # y (B,L,5376), pinv (5376,5120)
    return (y.float() - bias.view(1, 1, -1)) @ pinv


def _forward_project(q: torch.Tensor, transformer) -> torch.Tensor:
    layer = transformer.condition_proj
    W = layer.weight.detach().float().cpu()
    b = layer.bias.detach().float().cpu() if layer.bias is not None else torch.zeros(layer.out_features)
    return q @ W.T + b.view(1, 1, -1)


def _report(
    real: Dict[str, torch.Tensor],
    zero: Dict[str, torch.Tensor],
    transformer,
) -> None:
    pinv, bias = _condition_pinv(transformer)
    initial = real["projected"]

    print("\n=== H3 text-row probe ===")
    print(
        "tap          Δfrom_query_RMS   media_vs_zero_RMS   media_vs_zero_cos   "
        "pinv_recon_RMS   pinv_recon_cos   recovered_qwen_RMS"
    )
    print("-" * 126)

    for key in real.keys():
        r = real[key]
        z = zero[key]
        q = _inverse_project(r, pinv, bias)
        recon = _forward_project(q, transformer)
        print(
            f"{key:12s} "
            f"{_rms(r - initial):16.6f} "
            f"{_rms(r - z):17.6f} "
            f"{_cos(r, z):19.8f} "
            f"{_rms(recon - r):16.6f} "
            f"{_cos(recon, r):16.8f} "
            f"{_rms(q):18.6f}"
        )

    print(
        "\nInterpretation:\n"
        "  * media_vs_zero_* is the main first-pass signal. projected should be identical;\n"
        "    deeper taps should diverge if H3 writes media information into text rows.\n"
        "  * pinv_recon_* says how close that H3 state remains to the affine subspace reachable\n"
        "    directly from Qwen hidden_states[50] through condition_proj. Lower RMS / higher\n"
        "    cosine is friendlier to the later Qwen-mouth decoding experiment.\n"
        "  * A promising tap has both clear media-vs-zero separation AND tolerable pinv\n"
        "    reconstruction error. Intermediate blocks may beat block 50.\n"
    )


def _zero_media(inp: ProbeInputs) -> ProbeInputs:
    return ProbeInputs(
        video_rows=torch.zeros_like(inp.video_rows),
        audio_rows=torch.zeros_like(inp.audio_rows),
        text_states=inp.text_states,
        text_token_tags=inp.text_token_tags,
        layout=inp.layout,
        row_timesteps=inp.row_timesteps,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-path", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--audio")
    p.add_argument("--query", default="Describe the video and audio in detail.")
    p.add_argument("--partition", default="fl2va_pruned")
    p.add_argument("--blocks", default="1,5,10,20,30,40,50")
    p.add_argument(
        "--t",
        type=float,
        default=0.999,
        help="H3 t in [0,1], where 1 is clean. 0.999 mirrors clean keyframe conditioning.",
    )
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    if not (0.0 <= args.t <= 1.0):
        p.error("--t must be in [0,1]")

    video_path = Path(args.video).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve() if args.audio else None
    if not video_path.is_file():
        p.error(f"video not found: {video_path}")
    if audio_path is not None and not audio_path.is_file():
        p.error(f"audio not found: {audio_path}")

    print("Loading H3...")
    model = _find_model_loader(args.models_path, args.partition)
    transformer = model.transformer
    taps = _parse_blocks(args.blocks, len(transformer.blocks))

    print(f"Decoding source video: {video_path}")
    frames = _load_video_tensor(video_path, args.max_frames)
    print(f"Using {frames.shape[0]} frames after 24-fps resample / H3 grid snap")

    print("Encoding source video with H3 VAE...")
    video_latents = _encode_video_latents(model, frames)
    print(f"video latents: {tuple(video_latents.shape)} {video_latents.dtype}")

    audio_latents = None
    if audio_path is not None:
        print(f"Encoding source audio: {audio_path}")
        wav, _ = _load_audio_wave(audio_path)
        audio_latents = _encode_audio_latents(model, wav)
        print(f"audio latents: {tuple(audio_latents.shape)} {audio_latents.dtype}")

    inp = _prepare_inputs(model, args.query, video_latents, audio_latents, args.t)
    print(
        f"packed rows: total={inp.layout.sequence_length} text={len(inp.layout.text_indices)} "
        f"video={len(inp.layout.video_indices)} audio={len(inp.layout.audio_indices)}"
    )

    print(f"Running real-media capture at blocks {taps}...")
    real = _run_capture(transformer, inp, taps)

    print("Running zero-media control with identical query/layout/timesteps...")
    zero = _run_capture(transformer, _zero_media(inp), taps)

    _report(real, zero, transformer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
