"""MOSS-Image checks that need no weights: checkpoint naming."""

import os

import huggingface_hub
import pytest

import toolkit.models.v2.resolver as resolver
from extensions_built_in.diffusion_models.moss_image.src import checkpoints


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "MODELS_PATH", str(tmp_path))

    def offline(*args, **kwargs):
        raise AssertionError("hub download attempted")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    return tmp_path


def _resolve(repo_id, qtype):
    return os.path.normpath(checkpoints.comfy_transformer_file(repo_id, qtype=qtype))


def _touch(root, rel):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return os.path.normpath(path)


@pytest.mark.parametrize(
    "repo_id, variant",
    [(checkpoints.BASE_REPO, "base"), (checkpoints.SFT_REPO, "sft"), (checkpoints.COMFY_REPO, "sft")],
)
def test_repack_int8_names_resolve_locally(models_dir, repo_id, variant):
    # the repack's int8 files lost their "_attn" suffix on 2026-10-22
    path = _touch(models_dir, f"diffusion_models/moss_image_{variant}_int8_convrot.safetensors")
    assert _resolve(repo_id, "convrot8") == path


def test_old_attn_names_still_resolve(models_dir):
    path = _touch(models_dir, "diffusion_models/moss_image_base_int8_convrot_attn.safetensors")
    assert _resolve(checkpoints.BASE_REPO, "convrot8") == path


def test_bf16_base_wins_without_convrot(models_dir):
    _touch(models_dir, "diffusion_models/moss_image_base_int8_convrot.safetensors")
    bf16 = _touch(models_dir, "diffusion_models/moss_image_base_bf16.safetensors")
    # int8 only stands in for bf16 when it attaches as-is (convrot requested)
    assert _resolve(checkpoints.BASE_REPO, None) == bf16
    assert _resolve(checkpoints.BASE_REPO, "qfloat8") == bf16


def test_repack_listing_names():
    # Comfy-Org/MOSS-Image files as listed by the hub API on 2026-10-22
    listed = {
        "diffusion_models/moss_image_base_bf16.safetensors",
        "diffusion_models/moss_image_base_int8_convrot.safetensors",
        "diffusion_models/moss_image_sft_bf16.safetensors",
        "diffusion_models/moss_image_sft_int8_convrot.safetensors",
        "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
    }
    for repo_files, _ in checkpoints.COMFY_TRANSFORMER_FILES.values():
        assert set(repo_files) <= listed
    assert set(checkpoints.VENDOR_MIRROR.values()) <= listed
    assert checkpoints.COMFY_TEXT_ENCODER_FILES[0][1] in listed
