#!/usr/bin/env python3
"""Experimental MiniMax-H3 inverse captioning lab.

Two-stage idea:
  1. Qwen3-VL generates several dense video-caption candidates.
  2. MiniMax-H3 scores each caption against the actual clip by measuring
     matched-noise flow-prediction error at several timesteps.

The H3 score is experimental. It is useful as a relative ranking signal, not a
calibrated log-likelihood.

Designed for a single 24 GB CUDA GPU by never keeping Qwen3-VL and H3 resident
at the same time.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

# Running `python tools/h3_caption_lab.py` normally puts tools/ rather than the
# repository root on sys.path. Add the root so ai-toolkit internals import
# without requiring callers to remember PYTHONPATH=.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F


# The original 32B pre-quantized BnB proposer currently fails in its vision
# stack under the pinned Transformers/BitsAndBytes environment. The official
# 8B checkpoint, quantized to NF4 at load time below, is the proven-good 24 GB
# default. --qwen-model can still override it.
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ASSISTANT_LORA = (
    "ostris/minimax_h3_training_adapter/"
    "minimax_h3_training_adapter_alpha.safetensors"
)

DEFAULT_SYSTEM_PROMPT = r"""
You are writing a training caption for MiniMax-H3, a joint video-and-audio
generation model. Describe the supplied clip as a generation prompt that could
reproduce the observed clip.

Be concrete and factual. Preserve temporal order. Describe:
- the main subjects and stable identifying visual details;
- the environment, composition, lighting, and visual style;
- actions and object interactions in the order they occur;
- shot scale and camera angle;
- camera motion separately from subject motion;
- important transitions or changes over time;
- visible text only when it can actually be read.

Prefer explicit temporal language ("first", "then", "as", "while", "finally")
when it resolves ambiguity. Do not invent events outside the clip. Do not
invent dialogue, sound effects, music, or ambience: the video-language model
does not hear the soundtrack. If an external guide is supplied below, follow
its MiniMax-H3-specific conventions.

Output only the final caption. No analysis, headings, JSON, markdown, or
explanation.
""".strip()


@dataclass
class Candidate:
    text: str
    source: str = "qwen3-vl"
    seed: Optional[int] = None


@dataclass
class PassScore:
    timestep: int
    seed: int
    video_loss: float
    audio_loss: Optional[float]


@dataclass
class CaptionScore:
    text: str
    passes: list[PassScore]
    video_loss: float
    audio_loss: Optional[float]
    relative_gain: float


class _DatasetConfig:
    def __init__(self, do_audio: bool):
        self.do_i2v = False
        self.do_audio = do_audio


class _ScoreBatch:
    """Minimal batch surface consumed by MinimaxH3Model.get_noise_prediction."""

    def __init__(self, num_frames: int, audio_latents: Optional[torch.Tensor]):
        self.dataset_config = _DatasetConfig(audio_latents is not None)
        self.num_frames = num_frames
        self.audio_pred_slot = None
        self.audio_latents = audio_latents
        self.audio_data = None
        self.audio_noise = None
        self.audio_target = None
        self.audio_noisy = None
        self.audio_sigma = None
        self.audio_pred = None
        self.first_frame_latents = None
        self.tensor = None

    def set_secondary_audio_pred(self, value):
        self.audio_pred = value


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _seed_everything(seed: int) -> None:
    """Seed Python and Torch immediately before a stochastic operation."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_optional_guide(path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path).expanduser().resolve()
    text = p.read_text(encoding="utf-8")
    if len(text) > 40_000:
        text = text[:40_000]
    return text.strip()


def _candidate_instruction(guide: str, variant: int) -> str:
    focus = [
        "Prioritize precise temporal order and object interactions.",
        "Prioritize camera behavior, framing changes, and subject-versus-camera motion.",
        "Prioritize stable subject appearance, environment, composition, and lighting.",
        "Write the most complete balanced caption you can without speculative details.",
    ][variant % 4]
    out = f"{DEFAULT_SYSTEM_PROMPT}\n\nFor this candidate: {focus}"
    if guide:
        out += "\n\nMiniMax-H3 prompt-writing guide:\n---\n" + guide + "\n---"
    return out


def generate_candidates(
    video_path: Path,
    *,
    model_id: str,
    count: int,
    fps: float,
    max_video_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    guide: str,
) -> list[Candidate]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"[H3 caption lab] Loading caption proposer: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    if getattr(processor, "video_processor", None) is not None:
        token_unit = 32 * 32 * 2
        processor.video_processor.size = {
            "longest_edge": max(256, max_video_tokens) * token_unit,
            "shortest_edge": min(256, max_video_tokens) * token_unit,
        }

    load_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    lowered = model_id.lower()
    if not any(tag in lowered for tag in ("4bit", "4-bit", "awq", "gptq", "fp8")):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()

    candidates: list[Candidate] = []
    video_local_path = str(video_path.resolve())

    for i in range(count):
        this_seed = seed + i
        _seed_everything(this_seed)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": _candidate_instruction(guide, i)}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_local_path},
                    {
                        "type": "text",
                        "text": (
                            "Write one MiniMax-H3 training caption for this exact clip. "
                            "Do not describe anything you cannot observe."
                        ),
                    },
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"videos_kwargs": {"fps": fps}},
        )
        input_device = next(model.parameters()).device
        inputs = inputs.to(input_device)

        do_sample = count > 1
        generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        text = processor.decode(
            generated[0][prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if text:
            candidates.append(Candidate(text=text, seed=this_seed))
            print(f"[H3 caption lab] Candidate {len(candidates)}/{count}: {len(text)} chars")
        del inputs, generated

    del model, processor
    _cleanup_cuda()
    return candidates


def _target_hw(height: int, width: int, max_edge: int) -> tuple[int, int]:
    scale = min(1.0, float(max_edge) / float(max(height, width)))
    h = max(32, int(math.floor(height * scale / 32.0) * 32))
    w = max(32, int(math.floor(width * scale / 32.0) * 32))
    return h, w


def load_video_tensor(video_path: Path, *, fps: float, max_seconds: float, max_edge: int) -> torch.Tensor:
    """Decode and time-sample a local video to (T,C,H,W) in [-1,1]."""
    import av
    import numpy as np

    frames: list[torch.Tensor] = []
    next_time = 0.0
    target_hw = None

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if frame.time is not None:
                t = float(frame.time)
            elif frame.pts is not None and stream.time_base is not None:
                t = float(frame.pts * stream.time_base)
            else:
                guessed = float(stream.average_rate) if stream.average_rate else fps
                t = idx / max(guessed, 1e-6)
            if t > max_seconds:
                break
            if t + 1e-6 < next_time:
                continue
            next_time += 1.0 / fps

            arr = frame.to_ndarray(format="rgb24")
            if target_hw is None:
                target_hw = _target_hw(arr.shape[0], arr.shape[1], max_edge)
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
            tensor = tensor.float().div_(127.5).sub_(1.0).unsqueeze(0)
            if tensor.shape[-2:] != target_hw:
                tensor = F.interpolate(
                    tensor,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            frames.append(tensor.squeeze(0))

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return torch.stack(frames, dim=0)


def load_audio_data(video_path: Path, *, max_seconds: float) -> Optional[dict]:
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(video_path))
        max_samples = int(max_seconds * sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
        if waveform.numel() == 0:
            return None
        return {"waveform": waveform, "sample_rate": int(sample_rate)}
    except Exception as exc:
        print(f"[H3 caption lab] Audio decode unavailable; video-only scoring: {exc}")
        return None


def _load_h3(args):
    if args.models_path:
        os.environ["MODELS_PATH"] = str(Path(args.models_path).expanduser().resolve())

    from toolkit.config_modules import ModelConfig
    from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Model

    assistant = None if args.no_assistant_lora else args.assistant_lora
    model_config = ModelConfig(
        name_or_path=args.h3_model,
        arch="minimax_h3",
        dtype="bf16",
        vae_dtype="bf16",
        te_dtype="bf16",
        quantize=True,
        qtype="convrot8",
        quantize_te=True,
        qtype_te="nvfp4",
        low_vram=True,
        layer_offloading=True,
        layer_offloading_text_encoder_percent=1.0,
        layer_offloading_transformer_percent=1.0,
        assistant_lora_path=assistant,
        model_kwargs={"partition": args.partition},
    )
    model = MinimaxH3Model(device="cuda", model_config=model_config, dtype="bf16")
    model.load_model()
    return model


def _align_video_for_h3(video: torch.Tensor) -> torch.Tensor:
    from extensions_built_in.diffusion_models.minimax_h3.src import packing

    aligned = packing.align_num_frames_down(video.shape[0])
    if aligned < 1:
        raise ValueError("Clip is too short for MiniMax-H3 frame geometry.")
    if aligned != video.shape[0]:
        print(f"[H3 caption lab] H3 frame alignment: {video.shape[0]} -> {aligned} frames")
    return video[:aligned]


def _make_noise(shape, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def _embed_caption(h3, text: str):
    # H3 and its 32B Qwen conditioner do not coexist comfortably on 24 GB.
    # Park the DiT before bringing the text encoder onto CUDA.
    try:
        h3.model.to("cpu")
    except Exception:
        pass
    _cleanup_cuda()

    embeds = h3.get_prompt_embeds(text)
    try:
        h3.text_encoder.to("cpu")
    except Exception:
        pass
    _cleanup_cuda()
    return embeds


def _score_one_pass(
    h3,
    latents: torch.Tensor,
    audio_latents: Optional[torch.Tensor],
    embeds,
    *,
    num_frames: int,
    timestep: int,
    seed: int,
) -> PassScore:
    device = h3.device_torch
    clean = latents.to(device, torch.float32)
    noise = _make_noise(clean.shape, seed, device)
    sigma = float(timestep) / 1000.0
    noisy = (1.0 - sigma) * clean + sigma * noise
    target = noise - clean

    batch = _ScoreBatch(num_frames=num_frames, audio_latents=audio_latents)
    if audio_latents is not None:
        batch.audio_noise = _make_noise(audio_latents.shape, seed ^ 0x5A17C9E3, device)

    with torch.inference_mode():
        pred = h3.get_noise_prediction(
            noisy.to(h3.torch_dtype),
            torch.tensor([float(timestep)], device=device),
            embeds,
            batch=batch,
        )
        video_loss = F.mse_loss(pred.float(), target.float()).item()
        audio_loss = None
        if batch.audio_pred is not None and batch.audio_target is not None:
            audio_loss = F.mse_loss(batch.audio_pred.float(), batch.audio_target.float()).item()

    return PassScore(
        timestep=int(timestep),
        seed=int(seed),
        video_loss=float(video_loss),
        audio_loss=None if audio_loss is None else float(audio_loss),
    )


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(1, len(vals))


def _aggregate(text: str, passes: list[PassScore], blank: list[PassScore], audio_weight: float):
    video = _mean(p.video_loss for p in passes)
    audio_vals = [p.audio_loss for p in passes if p.audio_loss is not None]
    audio = _mean(audio_vals) if audio_vals else None

    matched_gains = []
    for p, b in zip(passes, blank):
        vg = (b.video_loss - p.video_loss) / max(abs(b.video_loss), 1e-8)
        ag = 0.0
        if p.audio_loss is not None and b.audio_loss is not None:
            ag = (b.audio_loss - p.audio_loss) / max(abs(b.audio_loss), 1e-8)
        matched_gains.append(vg + audio_weight * ag)

    return CaptionScore(
        text=text,
        passes=passes,
        video_loss=video,
        audio_loss=audio,
        relative_gain=_mean(matched_gains),
    )


def score_candidates(video_path: Path, candidates: list[Candidate], args) -> list[CaptionScore]:
    if not torch.cuda.is_available():
        raise RuntimeError("H3 scoring requires CUDA.")

    print("[H3 caption lab] Decoding clip for H3 scoring")
    video = load_video_tensor(
        video_path,
        fps=24.0,
        max_seconds=args.max_seconds,
        max_edge=args.score_max_edge,
    )

    h3 = _load_h3(args)
    video = _align_video_for_h3(video)
    num_frames = int(video.shape[0])

    print(f"[H3 caption lab] Encoding {num_frames} frames at {video.shape[-1]}x{video.shape[-2]}")
    # H3's video VAE samples from a posterior. Seed immediately before encode
    # so repeated score runs use the same clean latent rather than comparing
    # different posterior samples.
    _seed_everything(args.latent_seed)
    with torch.inference_mode():
        latents = h3.encode_images([video]).detach()

    audio_latents = None
    if not args.no_audio_score and num_frames > 1:
        audio_data = load_audio_data(
            video_path,
            max_seconds=min(args.max_seconds, num_frames / 24.0),
        )
        if audio_data is not None:
            _seed_everything(args.latent_seed ^ 0x31A9B7)
            with torch.inference_mode():
                audio_latents = h3.encode_audio([audio_data]).detach()
            del audio_data

    try:
        h3.vae.to("cpu")
    except Exception:
        pass
    del video
    _cleanup_cuda()

    pass_grid = []
    for repeat in range(args.score_repeats):
        repeat_seed = int(args.score_seed + repeat * 10007)
        for offset, timestep in enumerate(args.timesteps):
            pass_grid.append((int(timestep), repeat_seed + offset))

    print(
        f"[H3 caption lab] Scoring {len(pass_grid)} matched pass(es): "
        f"{len(args.timesteps)} timestep(s) x {args.score_repeats} noise seed repeat(s)"
    )
    print("[H3 caption lab] Scoring blank-caption baseline")
    blank_embeds = _embed_caption(h3, "")
    blank_passes = [
        _score_one_pass(
            h3,
            latents,
            audio_latents,
            blank_embeds,
            num_frames=num_frames,
            timestep=t,
            seed=s,
        )
        for t, s in pass_grid
    ]
    del blank_embeds
    _cleanup_cuda()

    scores: list[CaptionScore] = []
    for idx, candidate in enumerate(candidates, 1):
        print(f"[H3 caption lab] H3 scoring candidate {idx}/{len(candidates)}")
        embeds = _embed_caption(h3, candidate.text)
        passes = [
            _score_one_pass(
                h3,
                latents,
                audio_latents,
                embeds,
                num_frames=num_frames,
                timestep=t,
                seed=s,
            )
            for t, s in pass_grid
        ]
        scores.append(_aggregate(candidate.text, passes, blank_passes, args.audio_weight))
        del embeds
        _cleanup_cuda()

    scores.sort(key=lambda x: x.relative_gain, reverse=True)
    return scores


def _load_candidate_file(path: str) -> list[Candidate]:
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("candidates", data.get("captions", []))
        out = []
        for item in data:
            if isinstance(item, str):
                out.append(Candidate(item, source=str(p)))
            elif isinstance(item, dict) and item.get("text"):
                out.append(
                    Candidate(
                        text=str(item["text"]),
                        source=str(item.get("source") or p),
                        seed=item.get("seed"),
                    )
                )
        return out

    text = p.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return [Candidate(c, source=str(p)) for c in chunks]


def _default_output_path(video_path: Path, command: str) -> Path:
    suffix = {
        "generate": ".qwen.json",
        "score": ".h3score.json",
        "caption": ".h3caption.json",
    }[command]
    return video_path.with_suffix(video_path.suffix + suffix)


def _write_results(video_path: Path, candidates: list[Candidate], scores: list[CaptionScore], args):
    result_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output_path(video_path, args.command)
    )
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing results {result_path}; "
            "pass --overwrite or choose --output."
        )

    payload = {
        "video": str(video_path),
        "experimental": True,
        "command": args.command,
        "qwen_model": getattr(args, "qwen_model", None),
        "h3_model": getattr(args, "h3_model", None),
        "settings": {
            "score_max_edge": getattr(args, "score_max_edge", None),
            "max_seconds": getattr(args, "max_seconds", None),
            "timesteps": getattr(args, "timesteps", None),
            "score_seed": getattr(args, "score_seed", None),
            "score_repeats": getattr(args, "score_repeats", None),
            "latent_seed": getattr(args, "latent_seed", None),
            "audio_weight": getattr(args, "audio_weight", None),
            "audio_enabled": not getattr(args, "no_audio_score", True),
        },
        "candidates": [asdict(c) for c in candidates],
        "scores": [asdict(s) for s in scores],
        "best_caption": scores[0].text if scores else (candidates[0].text if candidates else None),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[H3 caption lab] Wrote {result_path}")

    best = payload["best_caption"]
    if args.write_sidecar and best:
        sidecar = video_path.with_suffix(".txt")
        if sidecar.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing caption {sidecar}; pass --overwrite.")
        sidecar.write_text(best.strip() + "\n", encoding="utf-8")
        print(f"[H3 caption lab] Wrote training sidecar {sidecar}")


def _add_common_score_args(p):
    p.add_argument("--models-path", help="Root containing diffusion_models/, text_encoders/, vae/, loras/")
    p.add_argument("--h3-model", default="Comfy-Org/MiniMax-H3")
    p.add_argument(
        "--partition",
        default="fl2va_pruned",
        choices=("fl2va", "fl2va_pruned", "ref2va", "ref2va_pruned"),
    )
    p.add_argument("--assistant-lora", default=DEFAULT_ASSISTANT_LORA)
    p.add_argument("--no-assistant-lora", action="store_true")
    p.add_argument(
        "--score-max-edge",
        type=int,
        default=256,
        help="Downscale H3 scoring so the longest edge is at most this many pixels.",
    )
    p.add_argument("--max-seconds", type=float, default=15.0)
    p.add_argument("--timesteps", type=int, nargs="+", default=[250, 500, 750])
    p.add_argument("--score-seed", type=int, default=1776)
    p.add_argument(
        "--score-repeats",
        type=int,
        default=1,
        help="Repeat each timestep with independent matched noise seeds and average the gains.",
    )
    p.add_argument(
        "--latent-seed",
        type=int,
        default=1701,
        help="Seed the H3 VAE posterior sample so repeated score runs use the same clean latent.",
    )
    p.add_argument("--no-audio-score", action="store_true")
    p.add_argument(
        "--audio-weight",
        type=float,
        default=1.0,
        help="Weight of normalized audio improvement in the relative ranking.",
    )


def _add_output_args(p):
    p.add_argument("--output")
    p.add_argument("--write-sidecar", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing result JSON and/or training sidecar.",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Experimental MiniMax-H3 inverse captioning / caption scoring."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("caption", help="Generate Qwen candidates, then rank them with H3.")
    cap.add_argument("video")
    cap.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    cap.add_argument("--candidates", type=int, default=4)
    cap.add_argument("--qwen-fps", type=float, default=2.0)
    cap.add_argument("--qwen-video-tokens", type=int, default=1536)
    cap.add_argument("--max-new-tokens", type=int, default=768)
    cap.add_argument("--temperature", type=float, default=0.7)
    cap.add_argument("--top-p", type=float, default=0.9)
    cap.add_argument("--candidate-seed", type=int, default=4242)
    cap.add_argument("--guide", help="Optional H3 prompt-writing guide to inject into Qwen's instruction.")
    _add_common_score_args(cap)
    _add_output_args(cap)

    gen = sub.add_parser("generate", help="Generate Qwen caption candidates only.")
    gen.add_argument("video")
    gen.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    gen.add_argument("--candidates", type=int, default=4)
    gen.add_argument("--qwen-fps", type=float, default=2.0)
    gen.add_argument("--qwen-video-tokens", type=int, default=1536)
    gen.add_argument("--max-new-tokens", type=int, default=768)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--candidate-seed", type=int, default=4242)
    gen.add_argument("--guide")
    _add_output_args(gen)

    score = sub.add_parser("score", help="Rank supplied captions with H3.")
    score.add_argument("video")
    score.add_argument("--candidate", action="append", default=[])
    score.add_argument("--candidate-file")
    _add_common_score_args(score)
    _add_output_args(score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if getattr(args, "score_repeats", 1) < 1:
        raise ValueError("--score-repeats must be >= 1")

    candidates: list[Candidate] = []
    if args.command in ("caption", "generate"):
        guide = _read_optional_guide(args.guide)
        candidates.extend(
            generate_candidates(
                video_path,
                model_id=args.qwen_model,
                count=args.candidates,
                fps=args.qwen_fps,
                max_video_tokens=args.qwen_video_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.candidate_seed,
                guide=guide,
            )
        )

    if args.command == "score":
        candidates.extend(Candidate(text=x, source="cli") for x in args.candidate)
        if args.candidate_file:
            candidates.extend(_load_candidate_file(args.candidate_file))

    deduped = []
    seen = set()
    for candidate in candidates:
        key = candidate.text.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(candidate)
    candidates = deduped

    if not candidates:
        raise ValueError("No caption candidates were produced or supplied.")

    if args.command == "generate":
        _write_results(video_path, candidates, [], args)
        return 0

    scores = score_candidates(video_path, candidates, args)
    print("\n[H3 caption lab] Ranking:")
    for i, score in enumerate(scores, 1):
        audio = "n/a" if score.audio_loss is None else f"{score.audio_loss:.6f}"
        print(
            f"  {i:>2}. gain={score.relative_gain:+.5f} "
            f"video={score.video_loss:.6f} audio={audio}"
        )
    _write_results(video_path, candidates, scores, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
