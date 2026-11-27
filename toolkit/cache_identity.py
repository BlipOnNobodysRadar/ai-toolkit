"""Deterministic identities for derived training caches.

The cache-key helpers here are deliberately small and standard-library only so
the dataloader can use them without importing model stacks. Cache keys avoid
reading or hashing media/reference/model files on disk; path and lightweight
stat metadata are used instead.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_key_digest(info: Any) -> str:
    """Stable digest used as a second check on a latent cache file's metadata."""
    return hashlib.sha256(canonical_json(info).encode("utf-8")).hexdigest()


def encoder_identity_fingerprint(identity: Any) -> str:
    """Compact per-preparation digest of a full encoder identity.

    A full model-path identity can be large (it contains one metadata entry per
    local file).  Callers compute it once per prepared dataset and carry this
    digest into every per-item key instead of serializing the manifest for each
    item.
    """
    return cache_key_digest(identity)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    # torch tensors and other config objects are not imported here.  Their
    # class/shape/dtype labels are enough to distinguish common architecture
    # configs without hashing weights.
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        return {"class": type(value).__name__, "shape": _jsonable(shape), "dtype": str(dtype)}
    return {"class": type(value).__name__, "value": str(value)}


def sha256_file(path: os.PathLike | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: str) -> dict:
    st = os.stat(path)
    return {
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(getattr(st, "st_ctime_ns", 0)),
        "ino": int(getattr(st, "st_ino", 0)),
        "dev": int(getattr(st, "st_dev", 0)),
    }


def file_generation(path: Optional[os.PathLike | str]) -> Optional[dict]:
    """Cheap admission-time identity for a single file.

    This is metadata, not a content snapshot.  It catches the normal rewrite and
    atomic-replacement cases while avoiding a full hash on every request/step.
    A deliberate in-place rewrite that preserves size, mtime and the same inode
    cannot be distinguished from an unchanged file; this is an explicit limit,
    not a content hash.
    """
    if not path:
        return None
    real = os.path.realpath(os.fspath(path))
    if not os.path.isfile(real):
        return {"path": real, "exists": False}
    meta = _file_metadata(real)
    meta["path"] = real
    meta["exists"] = True
    return meta


def local_path_identity(path: Optional[os.PathLike | str]) -> Optional[dict]:
    """Lightweight stat/path identity for a local file or directory.

    Avoids walking directories or hashing file bytes on disk while still
    preserving file name + stat metadata for callers (such as feedback
    reference tracking) that drop the drive/absolute `path` key.
    """
    if path is None:
        return None
    raw = os.fspath(path)
    if not raw:
        return None
    real = os.path.realpath(raw)
    if not os.path.exists(real):
        return {"path": real, "exists": False}
    info = {
        "path": real,
        "exists": True,
        "kind": "directory" if os.path.isdir(real) else "file",
        "name": os.path.basename(real),
    }
    info.update(_file_metadata(real))
    return info


def _cfg(model_config: Any, *names: str, default=None):
    for name in names:
        if model_config is not None and hasattr(model_config, name):
            value = getattr(model_config, name)
            if value is not None:
                return value
    return default


def _component_local_path(base: Any, subdir: str) -> Optional[str]:
    if base is None:
        return None
    base_s = os.fspath(base)
    if not base_s:
        return None
    if os.path.isfile(base_s):
        return base_s
    if os.path.isdir(base_s):
        candidate = os.path.join(base_s, subdir)
        if os.path.exists(candidate):
            return candidate
        return base_s
    return None


def _config_identity(model: Any) -> Any:
    config = getattr(model, "config", None)
    return _jsonable(config)


def _loaded_text_encoder_identities(sd: Any) -> list:
    encoders = getattr(sd, "text_encoder", None)
    if encoders is None:
        return []
    if not isinstance(encoders, (list, tuple)):
        encoders = [encoders]
    out = []
    for i, enc in enumerate(encoders):
        if enc is None:
            out.append({"index": i, "present": False})
            continue
        out.append({
            "index": i,
            "class": type(enc).__name__,
            "config": _config_identity(enc),
        })
    return out


def build_latent_encoder_identity(sd: Any) -> dict:
    """Effective VAE/encoding identity for latent cache keys.

    Deliberately excludes DiT `name_or_path` and DiT quantization settings so
    switching merged checkpoints or DiT quantizers reuses identical VAE latents
    without disk bloat.
    """
    cfg = getattr(sd, "model_config", None)
    vae_path = _cfg(cfg, "vae_path")
    return {
        "kind": "latent",
        "requested_vae_path": None if vae_path is None else os.path.realpath(os.fspath(vae_path)),
        "vae_class": None if getattr(sd, "vae", None) is None else type(sd.vae).__name__,
        "vae_config": None if getattr(sd, "vae", None) is None else _config_identity(sd.vae),
        "vae_dtype": str(_cfg(cfg, "vae_dtype", default=getattr(sd, "vae_torch_dtype", None))),
        "cache_latents_as_uint8": bool(getattr(sd, "cache_latents_as_uint8", False)),
        "load_rgba": bool(getattr(sd, "load_rgba", False)),
        "is_audio_model": bool(getattr(sd, "is_audio_model", False)),
        "sample_rate": getattr(sd, "sample_rate", None),
    }


def build_text_encoder_identity(sd: Any) -> dict:
    """Effective text-encoder/prompt-pipeline identity for TE cache keys.

    Tracks TE precision (`quantize_te`/`qtype_te`/`te_torch_dtype`) and explicit
    TE overrides while excluding DiT `name_or_path` so merged/sibling DiT runs
    reuse identical cached text embeddings.
    """
    cfg = getattr(sd, "model_config", None)
    te_path = _cfg(cfg, "te_name_or_path", "text_encoder_path")
    qtype_te = _cfg(cfg, "qtype_te")
    if isinstance(qtype_te, str) and "|" in qtype_te:
        qtype_te = qtype_te.split("|", 1)[0]
    return {
        "kind": "text",
        "requested_te_path": None if te_path is None else os.path.realpath(os.fspath(te_path)),
        "loaded_text_encoders": _loaded_text_encoder_identities(sd),
        "quantize_te": _cfg(cfg, "quantize_te", "quantize"),
        "qtype_te": qtype_te,
        "text_encoder_bits": _cfg(cfg, "text_encoder_bits"),
        "te_padding_side": getattr(sd, "te_padding_side", "right"),
        "te_torch_dtype": str(getattr(sd, "te_torch_dtype", None)),
    }


def ordered_reference_identity(paths: Iterable[os.PathLike | str] | None) -> list:
    """Ordered reference paths without hashing file contents on disk."""
    if not paths:
        return []
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return [os.path.realpath(os.fspath(path)) for path in paths]
