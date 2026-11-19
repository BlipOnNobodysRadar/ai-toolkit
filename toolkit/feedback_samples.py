"""Candidate image-pair capture for hybrid MOSS text-to-image preference training.

The ordinary sample path stays untouched.  At an eligible ordinary sampling
event this module reuses the first sample for every configured prompt, asks the
model for one extra independently-seeded image with the same generation
settings, and publishes one immutable preference candidate directory per
prompt.

Only the fixed files in :data:`PAIR_FILES` are ever exposed.  A pair is built
inside a private staging directory and the directory is renamed into place only
after every file exists and validates.  UI-readable paths never come from model
output or user metadata.
"""

from __future__ import annotations

import copy
import json
import secrets
import os
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.print import print_acc

PAIR_VERSION = 1
PAIR_FILES = (
    "pair.json",
    "a.png",
    "b.png",
    "embeddings.safetensors",
    "latents.safetensors",
)
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_REFERENCE_RE = re.compile(r"^[0-9a-f]{64}$")
_STAGING_PREFIX = ".staging-"


def _ordinary_image_path(config: Any, index: int) -> Optional[str]:
    try:
        path = config.get_image_path(index)
    except Exception:
        path = None
    if path and os.path.isfile(path):
        return os.path.abspath(path)
    fallback = getattr(config, "output_path", None)
    if fallback and os.path.isfile(fallback):
        return os.path.abspath(fallback)
    return None


def _independent_seed(seed: int) -> int:
    # Sample independently of the training RNG. The archive records the draw
    # so the pair can be reproduced even if the gallery reuses a fixed seed.
    return (int(seed) + 1 + secrets.randbelow(2**32 - 1)) % 2**32


def _clone_for_second_sample(config: Any, output_path: str, seed: int) -> Any:
    second = copy.copy(config)
    second.seed = int(seed)
    second.feedback_conditioning = None
    second.output_path = output_path
    second.output_folder = os.path.dirname(output_path)
    second.output_filename_no_ext = os.path.splitext(os.path.basename(output_path))[0]
    second.logger = None  # temporary second-arm paths do not belong in the ordinary sample log
    second.gen_time = int(time.time() * 1000)
    # Keep mutable settings independent even though they are not changed.
    if hasattr(second, "extra_values"):
        second.extra_values = list(second.extra_values or [])
    if hasattr(second, "extra_kwargs"):
        second.extra_kwargs = dict(second.extra_kwargs or {})
    return second


def _write_png(source: Any, destination: str) -> None:
    """Normalize a rendered view to an RGB PNG and fsync it before publish.

    The real sampler path hands us the ordinary sample's saved filename.  A
    test/native sampler may instead return the rendered image objects, so an
    in-memory PIL image is accepted as well.
    """

    opened = None
    owned = False
    if isinstance(source, Image.Image):
        opened = source
    else:
        opened = Image.open(os.fspath(source))
        owned = True
    normalized = None
    try:
        opened.load()
        normalized = opened.convert("RGB")
        normalized.save(destination, format="PNG", optimize=False, compress_level=6)
    finally:
        if normalized is not None:
            normalized.close()
        if owned and opened is not None:
            opened.close()
    with open(destination, "r+b") as handle:
        os.fsync(handle.fileno())


def _png_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as opened:
        opened.load()
        rgb = opened.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor / 127.5 - 1.0


def _latents_for_pair(sd: Any, a_path: str, b_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    images = [_png_tensor(a_path), _png_tensor(b_path)]
    vae = getattr(sd, "vae", None)
    entry_device = None
    entry_training = None
    if vae is not None:
        entry_training = bool(getattr(vae, "training", False))
        try:
            entry_device = next(vae.parameters()).device
        except (StopIteration, AttributeError, TypeError):
            entry_device = getattr(vae, "device", None)
    try:
        with torch.no_grad():
            encoded = sd.encode_images(images)
        if hasattr(encoded, "tensor"):
            encoded = encoded.tensor
        if not torch.is_tensor(encoded):
            raise TypeError("VAE encode did not return a tensor")
        if encoded.ndim != 4 or encoded.shape[0] != 2:
            raise ValueError(f"VAE encode returned {tuple(encoded.shape)}, expected a batch of two latents")
        # The dataset latent cache stores the VAE output and the loader hands
        # it to training in the model dtype.  Do the same conversion here so
        # the pair payload is the canonical MOSS training x0, not merely the
        # VAE's working dtype.
        latent_dtype = getattr(sd, "torch_dtype", None)
        a_latent = encoded[0].unsqueeze(0).detach().to("cpu").contiguous()
        b_latent = encoded[1].unsqueeze(0).detach().to("cpu").contiguous()
        if latent_dtype is not None:
            a_latent = a_latent.to(dtype=latent_dtype).contiguous()
            b_latent = b_latent.to(dtype=latent_dtype).contiguous()
        if a_latent.ndim != 4 or a_latent.shape[0] != 1 or b_latent.shape != a_latent.shape:
            raise ValueError("VAE latents do not have matching [1,C,H,W] shapes")
        if not torch.isfinite(a_latent).all() or not torch.isfinite(b_latent).all():
            raise FloatingPointError("VAE produced non-finite training latents")
    finally:
        # MOSS's encode_images opportunistically moves a CPU VAE to its encode
        # device.  Put the shared trainer state back even when encoding failed;
        # capture must never leave an extra model resident on the GPU.
        if vae is not None and entry_device is not None:
            try:
                current_device = next(vae.parameters()).device
            except (StopIteration, AttributeError, TypeError):
                current_device = getattr(vae, "device", None)
            if current_device != entry_device:
                vae.to(entry_device)
        if vae is not None and entry_training is not None:
            vae.train(entry_training)
    return a_latent, b_latent


def _reference_id(trainer: Any) -> str:
    from toolkit.feedback_training import feedback_reference_id

    reference_id = feedback_reference_id(trainer)
    if not isinstance(reference_id, str) or _REFERENCE_RE.fullmatch(reference_id) is None:
        raise ValueError("feedback_reference_id(trainer) did not return canonical lowercase 64-hex")
    return reference_id


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _publish_pair(
    parent: str,
    pair_id: str,
    *,
    step: int,
    provenance: dict,
    prompt: str,
    seeds: Sequence[int],
    reference_id: str,
    created_at: str,
    a_source: str,
    b_source: str,
    embedding: AdvancedPromptEmbeds,
    sd: Any,
) -> str:
    pair_stage = os.path.join(parent, pair_id)
    os.makedirs(pair_stage, exist_ok=False)
    a_path = os.path.join(pair_stage, "a.png")
    b_path = os.path.join(pair_stage, "b.png")
    _write_png(a_source, a_path)
    _write_png(b_source, b_path)

    with Image.open(a_path) as a_image:
        width, height = a_image.size
    with Image.open(b_path) as b_image:
        if b_image.size != (width, height):
            raise ValueError("second sample dimensions differ from the first sample")

    embedding_path = os.path.join(pair_stage, "embeddings.safetensors")
    embedding.save(embedding_path)

    a_latent, b_latent = _latents_for_pair(sd, a_path, b_path)
    latent_path = os.path.join(pair_stage, "latents.safetensors")
    save_file({"a": a_latent, "b": b_latent}, latent_path)

    pair_path = os.path.join(pair_stage, "pair.json")
    _write_json(pair_path, {
        "version": PAIR_VERSION,
        "id": pair_id,
        "step": int(step),
        **provenance,
        "prompt": str(prompt),
        "seeds": [int(seeds[0]), int(seeds[1])],
        "width": int(width),
        "height": int(height),
        "reference_id": reference_id,
        "created_at": created_at,
    })

    # Fixed names and a complete staging directory only: pair.json is written
    # last, then the single rename exposes all five files at once.
    missing = [name for name in PAIR_FILES if not os.path.isfile(os.path.join(pair_stage, name))]
    if missing:
        raise RuntimeError(f"staged preference pair is incomplete: {missing}")
    if os.path.exists(os.path.join(pair_stage, "vote.json")):
        raise RuntimeError("trainer must not create vote.json")
    return pair_stage


def capture_feedback_samples(
    trainer: Any,
    sample_config: Any,
    image_configs: Sequence[Any],
    *,
    step: Optional[int] = None,
) -> list[str]:
    """Capture one independent second sample per configured positive prompt.

    Returns the published pair directories.  Any capture-side error is
    reported and swallowed so ordinary rendering and the native sampling
    cleanup path keep their original behavior.
    """

    if not trainer.accelerator.is_main_process or not image_configs:
        return []
    sd = trainer.sd
    schedule = trainer._feedback_schedule
    completed_step = getattr(trainer, "_checkpoint_completed_step", None)
    if completed_step is None:
        completed_step = trainer.step_num if step is None else step

    preferences_root = os.path.abspath(os.path.join(str(trainer.save_root), "preferences"))
    parent = None
    published: list[str] = []
    try:
        # The first image for every prompt must already be a finished ordinary
        # sample. Validate that before paying for the second sampling pass.
        first_sources: list[Any] = []
        for index, config in enumerate(image_configs):
            first_source = _ordinary_image_path(config, index)
            if first_source is None:
                raise FileNotFoundError(f"ordinary sample {index} was not written")
            first_sources.append(first_source)

        os.makedirs(preferences_root, exist_ok=True)
        parent = tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=preferences_root)
        work_dir = os.path.join(parent, ".work")
        os.makedirs(work_dir, exist_ok=False)

        second_configs = []
        second_seeds = []
        for index, config in enumerate(image_configs):
            first_seed = config.seed
            if first_seed is None:
                raise ValueError(f"ordinary sample {index} has no explicit integer seed")
            second_seed = _independent_seed(first_seed)
            extension = config.output_ext.lstrip(".")
            output_path = os.path.join(work_dir, f"b_{index}.{extension}")
            second_configs.append(_clone_for_second_sample(config, output_path, second_seed))
            second_seeds.append(second_seed)

        # One native pass, with the clone order identical to the ordinary
        # config order.  This keeps sample_prompts_cache indices correct even
        # though only the second image is generated here.
        sd.generate_images(second_configs, sampler=getattr(sample_config, "sampler", None))

        second_paths = []
        for index, config in enumerate(second_configs):
            second_path = _ordinary_image_path(config, index)
            if second_path is None:
                raise FileNotFoundError(f"second sample {index} was not written")
            second_paths.append(second_path)

        reference_id = _reference_id(trainer)
        created_at = _utc_now()
        staged_pairs: list[tuple[str, str]] = []
        for index, (first_config, second_config) in enumerate(zip(image_configs, second_configs)):
            pair_id = uuid.uuid4().hex
            if _ID_RE.fullmatch(pair_id) is None:
                raise RuntimeError("uuid4 produced an unexpected id")
            embedding = second_config.feedback_conditioning
            if not isinstance(embedding, AdvancedPromptEmbeds):
                raise RuntimeError(f"missing MOSS sample conditioning for pair {index}")
            prompt = getattr(first_config, "prompt", None)
            if prompt is None:
                prompt = sample_config.prompts[index]
            stage = _publish_pair(
                parent,
                pair_id,
                step=completed_step,
                provenance={
                    "optimizer_updates": schedule.updates,
                    "ema": trainer.ema is not None,
                    "network_multiplier": first_config.network_multiplier,
                    "sampler": sample_config.sampler,
                    "sampling_steps": first_config.num_inference_steps,
                    "guidance_scale": first_config.guidance_scale,
                    "negative_prompt": first_config.negative_prompt,
                },
                prompt=prompt,
                seeds=(int(first_config.seed), int(second_seeds[index])),
                reference_id=reference_id,
                created_at=created_at,
                a_source=first_sources[index],
                b_source=second_paths[index],
                embedding=embedding,
                sd=sd,
            )
            staged_pairs.append((stage, os.path.join(preferences_root, pair_id)))

        # Each rename commits one complete pair. A vote can arrive immediately
        # afterwards; never roll a published pair back if a later rename fails.
        for stage, final_path in staged_pairs:
            os.rename(stage, final_path)
            published.append(final_path)
        return published
    except Exception as exc:
        print_acc(f"Feedback sample capture skipped: {type(exc).__name__}: {exc}")
        return published
    finally:
        if parent is not None and os.path.isdir(parent):
            try:
                shutil.rmtree(parent)
            except Exception as exc:
                print_acc(f"Could not remove feedback staging folder {parent}: {exc}")
