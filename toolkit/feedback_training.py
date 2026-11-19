"""Consume immutable GUI preference pairs as native DPO training batches.

The GUI writes one directory per sampled image pair under
``<save_root>/preferences/<uuid4-hex>``.  This module is the trusted boundary
between that untrusted on-disk data and the trainer:

* only fixed filenames below the trainer's bound save root are read;
* symlinks and Windows reparse points are rejected;
* polling validates lightweight metadata and never opens tensor payloads;
* a frozen reference identity is cached once on the trainer and is derived only
  from the actual base/frozen model, encoder identity, and optional resolved
  component sources -- never from datasets, optimizer, LoRA, or train knobs;
* loaded tensors are collated through the real ``FileItemDTO`` and
  ``DataLoaderBatchDTO`` constructors and the native ``DTO.stack`` path.

The public interface used by the sampling/scheduling integration is
:func:`feedback_reference_id`, :func:`voted_feedback_ids`, and
:func:`load_feedback_batch`.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import json
import logging
import os
import re
import stat
import uuid
from typing import Any, Optional

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load as load_safetensors

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.cache_identity import (
    cache_key_digest,
    local_path_identity,
)
from toolkit.config_modules import DatasetConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO

LOGGER = logging.getLogger(__name__)

PAIR_ID_RE = re.compile(r"^[0-9a-f]{32}$")
PAIR_ID_V4_RE = re.compile(r"^[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}$")
REFERENCE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
PAIR_VERSION = 1
VOTE_VERSION = 1
VOTE_CHOICES = frozenset({"a", "b", "skip"})
FIXED_MATERIAL = ("a.png", "b.png", "embeddings.safetensors", "latents.safetensors")
_JSON_MAX_BYTES = 1024 * 1024
_IMAGE_MAX_BYTES = 256 * 1024 * 1024
_TENSOR_MAX_BYTES = 2 * 1024 * 1024 * 1024
_MAX_IMAGE_EDGE = 32768
_MAX_IMAGE_PIXELS = 268_435_456
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

# Polling caches parsed JSON and the one-time fixed-material checks behind file
# generation tokens.  Tensor payloads are never opened from this cache.
_SCAN_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_SCAN_CACHE_LIMIT = 50_000
_UNSET = object()

class FeedbackError(RuntimeError):
    """Base class for feedback-pair boundary errors."""


class FeedbackPathError(FeedbackError):
    """A path or identifier was not safe to resolve."""


class FeedbackCorruptError(FeedbackError):
    """A referenced feedback record or tensor payload is malformed."""


class FeedbackNotEligibleError(FeedbackError):
    """The pair exists but is not eligible for this reference."""


class FeedbackPendingError(FeedbackNotEligibleError):
    """The pair has no immutable vote yet."""


class FeedbackSkippedError(FeedbackNotEligibleError):
    """The user explicitly skipped this pair."""


class FeedbackForeignReferenceError(FeedbackNotEligibleError):
    """The pair belongs to a different frozen reference/config identity."""


@dataclasses.dataclass(frozen=True)
class _PairRecord:
    pair_id: str
    step: int
    prompt: str
    seeds: tuple[int, int]
    width: int
    height: int
    reference_id: str
    created_at: _datetime.datetime


@dataclasses.dataclass(frozen=True)
class _VoteRecord:
    choice: str
    voted_at: _datetime.datetime


def _is_reparse_point(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode) or bool(
        int(getattr(st, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE
    )


def _require_safe_directory(path: str, label: str) -> None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FeedbackCorruptError(
            f"{label} cannot be inspected: {type(exc).__name__}: {exc}"
        ) from exc
    if _is_reparse_point(st):
        raise FeedbackPathError(f"{label} is a symlink/reparse point and is not allowed")
    if not stat.S_ISDIR(st.st_mode):
        raise FeedbackCorruptError(f"{label} is not a directory")


def _require_safe_regular_file(
    path: str, label: str, *, max_bytes: Optional[int] = None
) -> os.stat_result:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FeedbackCorruptError(
            f"{label} cannot be inspected: {type(exc).__name__}: {exc}"
        ) from exc
    if _is_reparse_point(st):
        raise FeedbackPathError(f"{label} is a symlink/reparse point and is not allowed")
    if not stat.S_ISREG(st.st_mode):
        raise FeedbackCorruptError(f"{label} is not a regular file")
    if st.st_size <= 0:
        raise FeedbackCorruptError(f"{label} is empty")
    if max_bytes is not None and st.st_size > max_bytes:
        raise FeedbackCorruptError(
            f"{label} has an unreasonable size ({st.st_size} bytes)"
        )
    return st


def _read_closed_bytes(path: str, label: str, max_bytes: int) -> bytes:
    # lstat first so a Windows path that lacks O_NOFOLLOW cannot redirect us to
    # a symlink/reparse target.  fstat below rechecks the opened descriptor.
    _require_safe_regular_file(path, label, max_bytes=max_bytes)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FeedbackCorruptError(
            f"{label} cannot be opened: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        st = os.fstat(fd)
        if _is_reparse_point(st):
            raise FeedbackPathError(
                f"{label} is a symlink/reparse point and is not allowed"
            )
        if not stat.S_ISREG(st.st_mode):
            raise FeedbackCorruptError(f"{label} is not a regular file")
        if st.st_size < 0 or st.st_size > max_bytes:
            raise FeedbackCorruptError(
                f"{label} has an unreasonable size ({st.st_size} bytes)"
            )
        chunks: list[bytes] = []
        total = 0
        while total < st.st_size:
            chunk = os.read(fd, min(1024 * 1024, st.st_size - total))
            if not chunk:
                raise FeedbackCorruptError(f"{label} changed size while being read")
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise FeedbackCorruptError(
            f"{label} cannot be read: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        os.close(fd)


def _read_json_object(path: str, label: str) -> dict[str, Any]:
    raw = _read_closed_bytes(path, label, _JSON_MAX_BYTES)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeedbackCorruptError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise FeedbackCorruptError(f"{label} must contain one JSON object")
    return value


def _parse_utc(value: Any, field: str, context: str) -> _datetime.datetime:
    if not isinstance(value, str):
        raise FeedbackCorruptError(f"{context}: {field} must be an ISO-8601 UTC string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = _datetime.datetime.fromisoformat(text)
    except ValueError as exc:
        raise FeedbackCorruptError(f"{context}: {field} is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != _datetime.timedelta(0):
        raise FeedbackCorruptError(f"{context}: {field} must carry a UTC offset")
    return parsed.astimezone(_datetime.timezone.utc)


def _require_int(
    value: Any, field: str, context: str, *, minimum: Optional[int] = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeedbackCorruptError(f"{context}: {field} must be an integer")
    if minimum is not None and value < minimum:
        raise FeedbackCorruptError(f"{context}: {field} must be >= {minimum}")
    return value


def _validate_pair_record(data: dict[str, Any], pair_id: str) -> _PairRecord:
    context = f"pair {pair_id}/pair.json"
    if data.get("version") != PAIR_VERSION or isinstance(data.get("version"), bool):
        raise FeedbackCorruptError(f"{context}: unsupported version {data.get('version')!r}")
    if data.get("id") != pair_id:
        raise FeedbackCorruptError(f"{context}: id does not match its directory")
    prompt = data.get("prompt")
    if not isinstance(prompt, str):
        raise FeedbackCorruptError(f"{context}: prompt must be a string")
    seeds = data.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 2 or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise FeedbackCorruptError(f"{context}: seeds must be a two-integer list")
    width = _require_int(data.get("width"), "width", context, minimum=1)
    height = _require_int(data.get("height"), "height", context, minimum=1)
    if width > _MAX_IMAGE_EDGE or height > _MAX_IMAGE_EDGE:
        raise FeedbackCorruptError(f"{context}: image dimensions are unreasonably large")
    if width * height > _MAX_IMAGE_PIXELS:
        raise FeedbackCorruptError(f"{context}: image pixel count is unreasonably large")
    reference_id = data.get("reference_id")
    if not isinstance(reference_id, str) or not REFERENCE_ID_RE.fullmatch(reference_id):
        raise FeedbackCorruptError(
            f"{context}: reference_id must be a lowercase 64-hex string"
        )
    step = _require_int(data.get("step"), "step", context, minimum=0)
    created_at = _parse_utc(data.get("created_at"), "created_at", context)
    return _PairRecord(
        pair_id=pair_id,
        step=step,
        prompt=prompt,
        seeds=(int(seeds[0]), int(seeds[1])),
        width=width,
        height=height,
        reference_id=reference_id,
        created_at=created_at,
    )


def _validate_vote_data(data: dict[str, Any], pair_id: str) -> _VoteRecord:
    context = f"pair {pair_id}/vote.json"
    if data.get("version") != VOTE_VERSION or isinstance(data.get("version"), bool):
        raise FeedbackCorruptError(f"{context}: unsupported version {data.get('version')!r}")
    if data.get("id") != pair_id:
        raise FeedbackCorruptError(f"{context}: id does not match its directory")
    choice = data.get("choice")
    if not isinstance(choice, str) or choice not in VOTE_CHOICES:
        raise FeedbackCorruptError(f"{context}: choice must be one of a, b, skip")
    voted_at = _parse_utc(data.get("voted_at"), "voted_at", context)
    return _VoteRecord(choice=choice, voted_at=voted_at)


def _source_identity(source):
    """Keep hub IDs independent of the launch directory and local IDs of drive letters."""
    if not source:
        return None
    source = os.fspath(source)
    if not os.path.exists(source):
        return source
    manifest = local_path_identity(source)
    # File identities retain the volume/file IDs and generation; directory entries
    # retain their relative paths. Remounting the same volume is safe.
    return {key: value for key, value in manifest.items() if key != "path"}


def _feedback_reference_identity(trainer):
    """Frozen weights and the preprocessing that produced the archived tensors."""
    sd = trainer.sd
    config = trainer.model_config
    sources = dict(getattr(sd, "feedback_component_sources", {}) or {})
    # Native model loaders record the exact component sources. Alternative loaders can
    # still identify the base and encoders from their configured sources.
    if not any(name == "transformer" or name.startswith("transformer/") for name in sources):
        sources["transformer"] = config.name_or_path
    kw = config.model_kwargs or {}
    sources.setdefault("text_encoder", getattr(config, "te_name_or_path", None) or kw.get("text_encoder_path"))
    sources.setdefault("vae", getattr(config, "vae_path", None))
    for name in ("lora_path", "assistant_lora_path", "accuracy_recovery_adapter", "unconditional_lora_path"):
        value = getattr(config, name, None)
        if name == "accuracy_recovery_adapter":
            adapter = getattr(sd, name, None)
            value = getattr(adapter, "_aitk_weight_source", None) or value
        if isinstance(value, str) and value:
            sources[name] = value
    precision = {key: getattr(config, key, None) for key in (
        "dtype", "vae_dtype", "te_dtype", "quantize", "qtype", "quantize_te", "qtype_te",
        "text_encoder_bits", "quantize_kwargs", "accuracy_recovery_adapter",
    )}
    # Adapter source was identified above; the option can also be a boolean switch.
    if "accuracy_recovery_adapter" in sources:
        precision.pop("accuracy_recovery_adapter")
    identity = {
        "schema": "moss-feedback-reference-v3", "architecture": config.arch,
        "sources": {key: _source_identity(source) for key, source in sources.items() if source},
        "precision": precision,
        "max_text_length": int(getattr(sd, "max_text_length", kw.get("max_text_length", 4096))),
        "edit": bool(kw.get("edit", False)), "kv_cache": bool(kw.get("kv_cache", False)),
    }
    if config.arch == "krea2":
        identity["schema"] = "krea2-feedback-reference-v1"
        # A Krea checkpoint supplies weights; the configured architecture and
        # conditioning also determine the frozen prediction for a given pair.
        identity["conditioning"] = getattr(sd, "feedback_reference_metadata", {"mmdit_config": kw.get("mmdit_config") or {}})
    return identity


def feedback_reference_id(trainer: Any) -> str:
    """Return the stable 64-lowercase-hex frozen reference for this loaded job.

    The value is computed once and cached on the trainer.  Dataset choices,
    optimizer state, LoRA settings, train knobs (step/cadence/lr/beta), and
    transient loaded/moved component state are intentionally not part of it.
    """
    cached = getattr(trainer, "_feedback_reference_id", None)
    if isinstance(cached, str) and REFERENCE_ID_RE.fullmatch(cached):
        return cached

    identity = _feedback_reference_identity(trainer)
    digest = cache_key_digest(identity)
    if not isinstance(digest, str) or not REFERENCE_ID_RE.fullmatch(digest):
        raise FeedbackError("could not derive a valid feedback reference identity")
    setattr(trainer, "_feedback_reference_id", digest)
    setattr(trainer, "_feedback_reference_identity", identity)
    return digest


def _bound_job_root(trainer: Any) -> str:
    root = getattr(trainer, "save_root", None)
    if not isinstance(root, (str, os.PathLike)):
        raise FeedbackPathError("feedback requires a bound job save_root on the trainer")
    raw = os.fspath(root)
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise FeedbackPathError("feedback save_root is not a usable path")
    return os.path.abspath(os.path.expanduser(raw))


def _preferences_dir(root: str) -> str:
    path = os.path.join(root, "preferences")
    if not os.path.lexists(path):
        return path
    _require_safe_directory(path, "feedback preferences directory")
    root_real = os.path.realpath(root)
    pref_real = os.path.realpath(path)
    try:
        if os.path.commonpath((root_real, pref_real)) != root_real:
            raise FeedbackPathError(
                "feedback preferences directory escapes the bound save root"
            )
    except ValueError as exc:
        raise FeedbackPathError(
            "feedback preferences directory has an invalid path"
        ) from exc
    return path


def _validate_pair_id(pair_id: Any) -> str:
    if not isinstance(pair_id, str) or not PAIR_ID_RE.fullmatch(pair_id):
        raise FeedbackPathError("pair_id must be a lowercase uuid4 hex string")
    try:
        parsed = uuid.UUID(hex=pair_id)
    except (ValueError, AttributeError) as exc:
        raise FeedbackPathError("pair_id must be a lowercase uuid4 hex string") from exc
    if parsed.version != 4:
        raise FeedbackPathError("pair_id must be a lowercase uuid4 hex string")
    return pair_id


def _pair_dir(root: str, pair_id: str) -> str:
    pair_id = _validate_pair_id(pair_id)
    preferences = _preferences_dir(root)
    pair_dir = os.path.join(preferences, pair_id)
    try:
        _require_safe_directory(pair_dir, f"feedback pair {pair_id}")
    except FileNotFoundError as exc:
        raise FeedbackCorruptError(f"feedback pair {pair_id} does not exist") from exc
    preferences_real = os.path.realpath(preferences)
    pair_real = os.path.realpath(pair_dir)
    try:
        if os.path.commonpath((preferences_real, pair_real)) != preferences_real:
            raise FeedbackPathError(
                f"feedback pair {pair_id} escapes the preferences directory"
            )
    except ValueError as exc:
        raise FeedbackPathError(
            f"feedback pair {pair_id} has an invalid path"
        ) from exc
    return pair_dir


def _fixed_child(pair_dir: str, name: str) -> str:
    allowed = ("pair.json", "vote.json") + FIXED_MATERIAL
    if name not in allowed:
        raise FeedbackPathError(f"unexpected feedback filename {name!r}")
    path = os.path.join(pair_dir, name)
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(pair_dir):
        raise FeedbackPathError("feedback material path escapes its pair directory")
    return path


def _require_fixed_material(pair_dir: str, pair_id: str) -> None:
    for name in FIXED_MATERIAL:
        path = _fixed_child(pair_dir, name)
        try:
            _require_safe_regular_file(path, f"pair {pair_id}/{name}")
        except FileNotFoundError as exc:
            raise FeedbackCorruptError(f"pair {pair_id} is missing {name}") from exc


def _file_token(path: str, label: str, *, required: bool) -> Optional[tuple[Any, ...]]:
    try:
        st = os.lstat(path)
    except FileNotFoundError as exc:
        if required:
            raise FeedbackCorruptError(f"{label} is missing") from exc
        return None
    except OSError as exc:
        raise FeedbackCorruptError(f"{label} cannot be inspected: {exc}") from exc
    if _is_reparse_point(st):
        raise FeedbackPathError(f"{label} is a symlink/reparse point and is not allowed")
    if not stat.S_ISREG(st.st_mode):
        raise FeedbackCorruptError(f"{label} is not a regular file")
    return (
        int(st.st_size),
        int(st.st_mtime_ns),
        int(getattr(st, "st_ctime_ns", 0)),
        int(getattr(st, "st_ino", 0)),
        int(getattr(st, "st_dev", 0)),
    )


def _scan_cache_entry(preferences_real: str, pair_id: str) -> dict[str, Any]:
    key = (preferences_real, pair_id)
    cache = _SCAN_CACHE.get(key)
    if cache is None:
        if len(_SCAN_CACHE) >= _SCAN_CACHE_LIMIT:
            _SCAN_CACHE.clear()
        cache = {}
        _SCAN_CACHE[key] = cache
    return cache


def _vote_path(pair_dir):
    receipt = _fixed_child(pair_dir, "vote.json")
    if os.path.lexists(receipt):
        return receipt
    # The GUI atomically publishes a nonempty directory before projecting the
    # receipt. A crash in between still leaves a complete, irrevocable decision.
    committed = os.path.join(pair_dir, ".decision")
    if os.path.lexists(committed):
        _require_safe_directory(committed, "committed feedback decision")
        return os.path.join(committed, "vote.json")
    return receipt


def _scan_cached_pair(
    preferences_real: str, pair_dir: str, pair_id: str
) -> tuple[_PairRecord, Optional[_VoteRecord], dict[str, Any]]:
    cache = _scan_cache_entry(preferences_real, pair_id)

    pair_path = _fixed_child(pair_dir, "pair.json")
    pair_token = _file_token(pair_path, f"pair {pair_id}/pair.json", required=True)
    if cache.get("pair_token") != pair_token or "record" not in cache:
        data = _read_json_object(pair_path, f"pair {pair_id}/pair.json")
        cache["record"] = _validate_pair_record(data, pair_id)
        cache["pair_token"] = pair_token
    record: _PairRecord = cache["record"]

    vote_path = _vote_path(pair_dir)
    vote_token = _file_token(vote_path, f"pair {pair_id}/vote.json", required=False)
    if cache.get("vote_token") != vote_token or "vote" not in cache:
        vote = None
        if vote_token is not None:
            data = _read_json_object(vote_path, f"pair {pair_id}/vote.json")
            vote = _validate_vote_data(data, pair_id)
        prior = cache.get("vote", _UNSET)
        if prior is not _UNSET and prior is not None and prior != vote:
            # vote.json is an immutable atomic decision.  A changed decision is
            # tampering, not a delayed-first vote (None -> decision is allowed).
            raise FeedbackCorruptError(
                f"pair {pair_id}/vote.json changed after its first observation"
            )
        cache["vote"] = vote
        cache["vote_token"] = vote_token

    return record, cache["vote"], cache


def _ensure_cached_material(
    pair_dir: str, pair_id: str, cache: dict[str, Any]
) -> None:
    if not cache.get("material_ok"):
        _require_fixed_material(pair_dir, pair_id)
        cache["material_ok"] = True


def voted_feedback_ids(trainer: Any) -> list[str]:
    """Return sorted immutable pair ids with an eligible vote for this reference.

    Pending, skipped, foreign-reference, malformed, old-version, changed-vote
    and symlinked records are excluded.  This poll path validates metadata and
    fixed-file presence only; tensor payloads are loaded lazily by
    :func:`load_feedback_batch`.
    """
    root = _bound_job_root(trainer)
    preferences = _preferences_dir(root)
    logs = getattr(trainer, "additional_logs", None)
    if logs is not None:
        logs.update({"feedback/eligible_pairs": 0, "feedback/foreign_votes": 0,
                     "feedback/skipped_pairs": 0, "feedback/invalid_pairs": 0})
    if not os.path.isdir(preferences):
        return []
    preferences_real = os.path.realpath(preferences)
    reference_id = feedback_reference_id(trainer)

    try:
        with os.scandir(preferences) as scanning:
            entries = list(scanning)
    except OSError as exc:
        raise FeedbackCorruptError(
            f"cannot scan feedback preferences: {type(exc).__name__}: {exc}"
        ) from exc

    eligible: list[str] = []
    foreign = skipped = invalid = 0
    for entry in entries:
        pair_id = entry.name
        if not PAIR_ID_V4_RE.fullmatch(pair_id):
            continue
        try:
            st = entry.stat(follow_symlinks=False)
            if _is_reparse_point(st) or not stat.S_ISDIR(st.st_mode):
                continue
            pair_dir = os.path.join(preferences, pair_id)
            record, vote, _cache = _scan_cached_pair(
                preferences_real, pair_dir, pair_id
            )
            if vote is None:
                continue
            if vote.choice == "skip":
                skipped += 1
                continue
            if record.reference_id != reference_id:
                foreign += 1
                continue
            _ensure_cached_material(pair_dir, pair_id, _cache)
            eligible.append(pair_id)
        except FileNotFoundError:
            # The publisher may rename a complete directory into place between
            # enumeration and inspection.  It is not eligible this poll.
            continue
        except (FeedbackError, OSError) as exc:
            invalid += 1
            LOGGER.warning("Skipping feedback pair %s: %s", pair_id, exc)
            continue
    if foreign:
        LOGGER.warning("%d voted feedback pair(s) use a different frozen reference and were excluded", foreign)
    if logs is not None:
        logs.update({"feedback/eligible_pairs": len(eligible), "feedback/foreign_votes": foreign,
                     "feedback/skipped_pairs": skipped, "feedback/invalid_pairs": invalid})
    return sorted(eligible)


def _load_latents(pair_dir: str, pair_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    path = _fixed_child(pair_dir, "latents.safetensors")
    try:
        raw = _read_closed_bytes(
            path, f"pair {pair_id}/latents.safetensors", _TENSOR_MAX_BYTES
        )
    except FileNotFoundError as exc:
        raise FeedbackCorruptError(
            f"pair {pair_id}/latents.safetensors is missing"
        ) from exc
    try:
        tensors = load_safetensors(raw)
    except Exception as exc:
        raise FeedbackCorruptError(
            f"pair {pair_id}/latents.safetensors cannot be loaded: {exc}"
        ) from exc
    if not isinstance(tensors, dict) or set(tensors) != {"a", "b"}:
        raise FeedbackCorruptError(
            f"pair {pair_id}/latents.safetensors must contain exactly keys a and b"
        )
    out: list[torch.Tensor] = []
    for key in ("a", "b"):
        tensor = tensors[key]
        if not torch.is_tensor(tensor):
            raise FeedbackCorruptError(
                f"pair {pair_id}/latents.safetensors:{key} is not a tensor"
            )
        if not tensor.is_floating_point():
            raise FeedbackCorruptError(
                f"pair {pair_id}/latents.safetensors:{key} must be floating point"
            )
        if tensor.ndim != 4 or tensor.shape[0] != 1 or any(
            dim <= 0 for dim in tensor.shape[1:]
        ):
            raise FeedbackCorruptError(
                f"pair {pair_id}/latents.safetensors:{key} must have shape "
                f"[1,C,H,W], got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise FeedbackCorruptError(
                f"pair {pair_id}/latents.safetensors:{key} contains non-finite values"
            )
        out.append(tensor.detach().to("cpu").clone())
    if out[0].shape != out[1].shape:
        raise FeedbackCorruptError(
            f"pair {pair_id} latent shapes differ: "
            f"a={tuple(out[0].shape)} b={tuple(out[1].shape)}"
        )
    return out[0], out[1]


def _load_embeddings(pair_dir: str, pair_id: str) -> AdvancedPromptEmbeds:
    path = _fixed_child(pair_dir, "embeddings.safetensors")
    label = f"pair {pair_id}/embeddings.safetensors"
    try:
        _require_safe_regular_file(path, label, max_bytes=_TENSOR_MAX_BYTES)
    except FileNotFoundError as exc:
        raise FeedbackCorruptError(f"{label} is missing") from exc
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = set(handle.keys())
    except Exception as exc:
        raise FeedbackCorruptError(f"{label} cannot be inspected: {exc}") from exc
    if metadata.get("class_name") != "AdvancedPromptEmbeds":
        raise FeedbackCorruptError(
            f"{label} is not a single AdvancedPromptEmbeds.save payload"
        )
    try:
        embeds = AdvancedPromptEmbeds.load(path)
    except Exception as exc:
        raise FeedbackCorruptError(f"{label} cannot be loaded: {exc}") from exc
    if set(embeds.keys()) != keys or not keys:
        raise FeedbackCorruptError(f"{label} has inconsistent embedding keys")
    for key in embeds.keys():
        values = embeds[key]
        if not isinstance(values, list) or len(values) != 1 or not torch.is_tensor(values[0]):
            raise FeedbackCorruptError(f"{label}:{key} must hold exactly one tensor")
        tensor = values[0]
        if tensor.numel() <= 0:
            raise FeedbackCorruptError(f"{label}:{key} is empty")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise FeedbackCorruptError(f"{label}:{key} contains non-finite values")
    return embeds


def _validate_png(path: str, label: str, width: int, height: int) -> None:
    _require_safe_regular_file(path, label, max_bytes=_IMAGE_MAX_BYTES)
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise FeedbackCorruptError(f"{label} is not a PNG image")
            image.verify()
        with Image.open(path) as image:
            image.load()
            size = image.size
    except FeedbackError:
        raise
    except Exception as exc:
        raise FeedbackCorruptError(f"{label} is a malformed PNG: {exc}") from exc
    if size != (width, height):
        raise FeedbackCorruptError(
            f"{label} is {size[0]}x{size[1]}, pair.json declares {width}x{height}"
        )


def _feedback_dataset_config(pair_dir: str, width: int, height: int) -> DatasetConfig:
    """Neutral per-pair dataset config; never reuse a training DatasetConfig."""
    return DatasetConfig(
        folder_path=pair_dir,
        resolution=max(width, height),
        scale=1.0,
        buckets=False,
        random_scale=False,
        random_crop=False,
        random_crop_percent=0.0,
        square_crop=False,
        standardize_images=False,
        is_reg=False,
        prior_reg=False,
        network_weight=1.0,
        loss_multiplier=1.0,
        token_dropout_rate=0.0,
        caption_dropout_rate=0.0,
        shuffle_tokens=False,
        keep_tokens=0,
        random_triggers=[],
        augments=[],
        augmentations=None,
        clip_image_path=None,
        control_path=None,
        inpaint_path=None,
        mask_path=None,
        unconditional_path=None,
        cache_latents=True,
        cache_latents_to_disk=False,
        cache_text_embeddings=True,
        cache_tensors_to_disk=False,
        load_image_when_caching_latents=False,
        num_frames=1,
        auto_frame_count=False,
        do_i2v=False,
        do_audio=False,
    )


def _feedback_file_item(
    *,
    path: str,
    pair_dir: str,
    dataset_config: DatasetConfig,
    architecture: str,
    prompt: str,
    pair_id: str,
    role: str,
    latent: torch.Tensor,
    width: int,
    height: int,
    embeddings: AdvancedPromptEmbeds,
) -> FileItemDTO:
    label = f"pair {pair_id}/{os.path.basename(path)}"
    _validate_png(path, label, width, height)
    try:
        item = FileItemDTO(
            path=path,
            dataset_config=dataset_config,
            dataset_root=pair_dir,
            raw_caption=prompt,
            latent_cache_role=role,
            dpo_pair_signature=pair_id,
            scale_to_width=width,
            scale_to_height=height,
            crop_width=width,
            crop_height=height,
            latent_space_version=architecture,
            text_embedding_space_version=architecture,
        )
        item.raw_caption = prompt
        item.raw_caption_short = prompt
        item.load_caption()
        # ``get_latent`` and DTO.stack expect the per-item [C,H,W] axis.  The
        # contract file stores a single-image [1,C,H,W] payload, so remove the
        # file-only leading axis exactly once here.
        item._encoded_latent = latent.squeeze(0)
        item.is_latent_cached = True
        item.latent_load_device = "cpu"
        item.prompt_embeds = embeddings
        item.is_text_embedding_cached = True
        item.caption = prompt
        item.caption_short = prompt
        item.network_weight = 1.0
        item.loss_multiplier = 1.0
        item.is_reg = False
        item.prior_reg = False
        item.extra_values = []
    except FeedbackError:
        raise
    except Exception as exc:
        raise FeedbackCorruptError(f"{label} cannot be represented as FileItemDTO: {exc}") from exc
    if item.get_latent().shape != latent.shape[1:]:
        raise FeedbackCorruptError(f"{label} did not preserve the per-item latent shape")
    return item


def load_feedback_batch(trainer: Any, pair_id: str) -> DataLoaderBatchDTO:
    """Load one voted pair as a native B=1 DPO ``DataLoaderBatchDTO``.

    ``batch.latents`` is the chosen arm and ``batch.rejected_latents`` is the
    rejected arm.  The single ``file_items`` entry is a real chosen
    ``FileItemDTO`` with its rejected sibling, and the native constructor
    performs latent and cached prompt-embedding collation.
    """
    pair_id = _validate_pair_id(pair_id)
    root = _bound_job_root(trainer)
    reference_id = feedback_reference_id(trainer)
    pair_dir = _pair_dir(root, pair_id)
    preferences_real = os.path.realpath(_preferences_dir(root))
    record, vote, _cache = _scan_cached_pair(preferences_real, pair_dir, pair_id)

    if record.reference_id != reference_id:
        raise FeedbackForeignReferenceError(
            f"feedback pair {pair_id} belongs to reference "
            f"{record.reference_id}, not {reference_id}"
        )
    if vote is None:
        raise FeedbackPendingError(f"feedback pair {pair_id} has no vote yet")
    if vote.choice == "skip":
        raise FeedbackSkippedError(f"feedback pair {pair_id} was skipped")

    _ensure_cached_material(pair_dir, pair_id, _cache)
    latents_a, latents_b = _load_latents(pair_dir, pair_id)
    embeddings = _load_embeddings(pair_dir, pair_id)
    chosen_key = vote.choice
    rejected_key = "b" if chosen_key == "a" else "a"
    chosen_latent = latents_a if chosen_key == "a" else latents_b
    rejected_latent = latents_b if chosen_key == "a" else latents_a
    chosen_path = _fixed_child(pair_dir, f"{chosen_key}.png")
    rejected_path = _fixed_child(pair_dir, f"{rejected_key}.png")

    dataset_config = _feedback_dataset_config(
        pair_dir, record.width, record.height
    )
    chosen = _feedback_file_item(
        path=chosen_path,
        pair_dir=pair_dir,
        dataset_config=dataset_config,
        architecture=trainer.model_config.arch,
        prompt=record.prompt,
        pair_id=pair_id,
        role="chosen",
        latent=chosen_latent,
        width=record.width,
        height=record.height,
        embeddings=embeddings,
    )
    rejected = _feedback_file_item(
        path=rejected_path,
        pair_dir=pair_dir,
        dataset_config=dataset_config,
        architecture=trainer.model_config.arch,
        prompt=record.prompt,
        pair_id=pair_id,
        role="rejected",
        latent=rejected_latent,
        width=record.width,
        height=record.height,
        embeddings=embeddings,
    )
    chosen.rejected_item = rejected
    chosen.rejected_path = rejected_path

    try:
        batch = DataLoaderBatchDTO(file_items=[chosen])
    except FeedbackError:
        raise
    except Exception as exc:
        raise FeedbackCorruptError(
            f"pair {pair_id} cannot be collated as a feedback batch: {exc}"
        ) from exc
    if batch.latents is None or batch.rejected_latents is None:
        raise FeedbackCorruptError(
            f"pair {pair_id} did not produce chosen and rejected latents"
        )
    if tuple(batch.latents.shape) != tuple(chosen_latent.shape) or tuple(
        batch.rejected_latents.shape
    ) != tuple(rejected_latent.shape):
        raise FeedbackCorruptError(f"pair {pair_id} produced an invalid batch latent axis")

    batch.feedback_pair_id = pair_id
    batch.feedback_reference_id = reference_id
    batch.feedback_choice = vote.choice
    batch.feedback_prompt = record.prompt
    batch.feedback_seeds = record.seeds
    batch.feedback_step = record.step
    batch.feedback_voted_at = vote.voted_at
    return batch
