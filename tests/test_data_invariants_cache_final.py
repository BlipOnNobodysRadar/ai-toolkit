"""Focused cache-identity, retry, and cached-caption regressions (CPU, tiny files).

These tests exercise the real DTO key/path methods and the real cache admission
loops.  There is no model load and no GPU work.
"""
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from toolkit import cache_identity as ci
from toolkit import data_loader as dl
from toolkit import dataloader_mixins as dm
from toolkit.config_modules import DatasetConfig


def _cfg(**overrides):
    values = dict(
        folder_path="fixture", buckets=False, standardize_images=False,
        resolution=64, scale=1.0, random_crop=False, random_crop_percent=0,
        square_crop=False, auto_frame_count=False, trim_auto_frame_count_tail=False,
        num_frames=1, fps=24, do_i2v=False, do_audio=False,
        audio_normalize=False, audio_preserve_pitch=False,
        cache_tensors_to_disk=False, cache_latents_num_workers=1,
        diff_output_preservation=False, caption_dropout_rate=0.0,
        shuffle_tokens=False, random_triggers=[], random_triggers_max=1,
        token_dropout_rate=0.0, keep_tokens=0, augments=[],
    )
    values.update(overrides)
    return DatasetConfig(**values)


class LatentItem(dm.LatentCachingFileItemDTOMixin):
    def __init__(self, path, cfg, latent_space_version="latent-v1", encoder_identity=None):
        super().__init__()
        self.latent_cache_encoder_identity = encoder_identity
        self.path = str(path)
        self.dataset_config = cfg
        self.scale_to_width = self.scale_to_height = 16
        self.crop_x = self.crop_y = 0
        self.crop_width = self.crop_height = 16
        self.flip_x = self.flip_y = False
        self.latent_space_version = latent_space_version
        self.is_video = False
        self.is_audio_model = False
        self.load_rgba = False
        self.sample_rate = 48000
        self.audio_data = None
        self.tensor = None
        self.dataloader_transforms = None
        self.loads = 0

    def load_and_process_image(self, transform, only_load_latents=False):
        self.loads += 1
        pixels = np.asarray(Image.open(self.path).convert("RGB"), dtype=np.float32) / 127.5 - 1
        self.tensor = torch.from_numpy(pixels).permute(2, 0, 1)

    def cleanup(self):
        self.tensor = None


class TextItem(dm.TextEmbeddingFileItemDTOMixin):
    def __init__(self, path, cfg, caption="cap", text_embedding_space_version="text-v1", **kwargs):
        super().__init__()
        self.path = str(path)
        self.dataset_config = cfg
        self.caption = caption
        self.text_embedding_space_version = text_embedding_space_version
        self.te_padding_side = "right"
        self.text_embedding_uses_target_size = kwargs.get("text_embedding_uses_target_size", False)
        self.encode_control_in_text_embeddings = kwargs.get("encode_control_in_text_embeddings", False)
        self.encode_first_frame_in_text_embeddings = kwargs.get("encode_first_frame_in_text_embeddings", False)
        self.control_path = kwargs.get("control_path")
        self.control_video_paths = kwargs.get("control_video_paths")
        self.is_video = kwargs.get("is_video", False)
        self.crop_x = kwargs.get("crop_x", 0)
        self.crop_y = kwargs.get("crop_y", 0)
        self.crop_width = kwargs.get("crop_width", 16)
        self.crop_height = kwargs.get("crop_height", 16)
        self.scale_to_width = kwargs.get("scale_to_width", 16)
        self.scale_to_height = kwargs.get("scale_to_height", 16)
        self.flip_x = kwargs.get("flip_x", False)
        self.flip_y = kwargs.get("flip_y", False)
        self.dataloader_transforms = None


class LatentDataset(dm.LatentCachingMixin):
    def __init__(self, items):
        self.latent_cache = {}
        self.file_list = list(items)
        self.dataset_path = "fixture"
        self.is_caching_latents_to_disk = True
        self.is_caching_latents_to_memory = False
        self.dataset_config = SimpleNamespace(cache_latents_num_workers=1, buckets=False)
        self.transform = None
        self.sd = SimpleNamespace(
            torch_dtype=torch.float32,
            device_torch=torch.device("cpu"),
            device="cpu",
            encode_images=lambda imgs: torch.nn.functional.avg_pool2d(imgs, 4),
            save_device_state=lambda: None,
            set_device_state_preset=lambda name: None,
            restore_device_state=lambda: None,
        )


def _write_same_length_restore_mtime(path, data):
    st = path.stat()
    path.write_bytes(data)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


# ---------------------------------------------------------------- latent/text source and policy keys

def test_latent_key_does_not_hash_source_files_or_bloat_on_dataset_resolution_change(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"AAAA")

    def forbid_hash(path, *args, **kwargs):
        raise AssertionError(f"sha256_file should not be called for {path}")

    monkeypatch.setattr(ci, "sha256_file", forbid_hash)

    first = LatentItem(source, _cfg(resolution=[768, 1024], random_crop_percent=0))
    second = LatentItem(source, _cfg(resolution=[1024], random_crop_percent=10))
    info = first.get_latent_info_dict()
    assert "source_content_sha256" not in info
    assert "cache_key_schema_version" not in info
    assert "preprocessing_identity" not in info
    # Same crop/scale geometry must reuse the exact same cache path (zero disk bloat).
    assert first.get_latent_path(recalculate=True) == second.get_latent_path(recalculate=True)


def test_latent_key_keeps_model_space_override_and_encoder_digest(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"same")
    cfg = _cfg()
    base = LatentItem(source, cfg, latent_space_version="space-a", encoder_identity="encoder-a")
    same = LatentItem(source, cfg, latent_space_version="space-a", encoder_identity="encoder-a")
    override = LatentItem(source, cfg, latent_space_version="space-b", encoder_identity="encoder-a")
    encoder = LatentItem(source, cfg, latent_space_version="space-a", encoder_identity="encoder-b")
    assert base.get_latent_path() == same.get_latent_path()
    assert base.get_latent_path() != override.get_latent_path()
    assert base.get_latent_path() != encoder.get_latent_path()


def test_text_and_control_keys_never_hash_source_files(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    control = tmp_path / "ctrl.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(source)
    Image.new("RGB", (4, 4), (4, 5, 6)).save(control)

    def forbid_hash(path, *args, **kwargs):
        raise AssertionError(f"sha256_file should not be called for {path}")

    monkeypatch.setattr(ci, "sha256_file", forbid_hash)
    cfg = _cfg(do_i2v=True, num_frames=5)
    item = TextItem(source, cfg, encode_control_in_text_embeddings=True, control_path=[str(control)])
    info = item.get_text_embedding_info_dict(force_source_identity=True)
    assert "self_reference_content_sha256" not in info
    assert "cache_key_schema_version" not in info
    dopsd = item.get_text_embedding_info_dict(dopsd_self_ref=True, force_source_identity=True)
    assert "self_reference_content_sha256" not in dopsd


def test_latent_encoder_identity_tracks_vae_not_dit_checkpoint_or_quant(tmp_path):
    vae_a = tmp_path / "vae_a.safetensors"
    vae_b = tmp_path / "vae_b.safetensors"
    vae_a.write_bytes(b"A" * 32)
    vae_b.write_bytes(b"B" * 32)
    sd = SimpleNamespace(
        model_config=SimpleNamespace(vae_path=str(vae_a), name_or_path="dit-v1", qtype="qfloat8",
                                     vae_dtype="bf16", quantize=True),
        vae=SimpleNamespace(config={"latent_channels": 4}),
        cache_latents_as_uint8=False, load_rgba=False, is_audio_model=False, sample_rate=None,
    )
    base = ci.cache_key_digest(ci.build_latent_encoder_identity(sd))
    # Switching merged DiT checkpoint or DiT quantizer must NOT invalidate VAE latent cache.
    sd.model_config.name_or_path = "dit-v2-merged"
    sd.model_config.quantize = False
    sd.model_config.qtype = None
    assert ci.cache_key_digest(ci.build_latent_encoder_identity(sd)) == base

    # Changing actual VAE path or VAE config DOES change latent encoder identity.
    sd.model_config.vae_path = str(vae_b)
    diff_path = ci.cache_key_digest(ci.build_latent_encoder_identity(sd))
    sd.vae = SimpleNamespace(config={"latent_channels": 8})
    diff_cfg = ci.cache_key_digest(ci.build_latent_encoder_identity(sd))
    assert len({base, diff_path, diff_cfg}) == 3


def test_text_encoder_identity_tracks_te_config_and_precision_not_dit(tmp_path):
    te_a = tmp_path / "te_a.safetensors"
    te_b = tmp_path / "te_b.safetensors"
    te_a.write_bytes(b"A" * 32)
    te_b.write_bytes(b"B" * 32)
    sd = SimpleNamespace(
        model_config=SimpleNamespace(te_name_or_path=str(te_a), name_or_path="dit-v1", qtype_te="qfloat8",
                                     quantize_te=True, text_encoder_bits=8),
        text_encoder=[SimpleNamespace(config={"hidden_size": 1024})],
        te_padding_side="right", te_torch_dtype="bf16",
    )
    base = ci.cache_key_digest(ci.build_text_encoder_identity(sd))
    # Switching DiT checkpoint reuses identical TE embeddings.
    sd.model_config.name_or_path = "dit-v2-merged"
    assert ci.cache_key_digest(ci.build_text_encoder_identity(sd)) == base

    # Changing TE path, TE config, or TE quantizer isolates caches safely.
    sd.model_config.te_name_or_path = str(te_b)
    diff_path = ci.cache_key_digest(ci.build_text_encoder_identity(sd))
    sd.text_encoder[0].config = {"hidden_size": 2048}
    diff_cfg = ci.cache_key_digest(ci.build_text_encoder_identity(sd))
    sd.model_config.qtype_te = "int4"
    diff_quant = ci.cache_key_digest(ci.build_text_encoder_identity(sd))
    assert len({base, diff_path, diff_cfg, diff_quant}) == 4


def test_local_path_identity_is_lightweight_stat_metadata_without_walking_or_hashing(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    for i in range(10):
        (root / f"small_{i:03}.json").write_text("x", encoding="utf-8")

    def forbid_hash(path, *args, **kwargs):
        raise AssertionError(f"sha256_file should not be called for {path}")

    monkeypatch.setattr(ci, "sha256_file", forbid_hash)
    identity = ci.local_path_identity(root)
    assert identity["exists"] is True
    assert identity["kind"] == "directory"
    assert identity["name"] == "model"
    assert "files" not in identity
    assert "small_files_hashed" not in identity

    single = root / "weights.safetensors"
    single.write_bytes(b"first")
    first_id = ci.local_path_identity(single)
    assert first_id["kind"] == "file" and first_id["size"] == 5
    single.write_bytes(b"second-bytes")
    second_id = ci.local_path_identity(single)
    assert second_id["size"] == 12 and second_id != first_id


def test_control_reference_order_changes_text_key_without_hashing(tmp_path, monkeypatch):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(a)
    Image.new("RGB", (4, 4), (4, 5, 6)).save(b)

    def forbid_hash(path, *args, **kwargs):
        raise AssertionError(f"sha256_file should not be called for {path}")

    monkeypatch.setattr(ci, "sha256_file", forbid_hash)
    cfg = _cfg()
    one = TextItem(tmp_path / "item.png", cfg, encode_control_in_text_embeddings=True,
                   control_path=[str(a), str(b)])
    first = ci.cache_key_digest(one.get_text_embedding_info_dict())
    second = ci.cache_key_digest(one.get_text_embedding_info_dict())
    assert first == second

    two = TextItem(tmp_path / "item.png", cfg, encode_control_in_text_embeddings=True,
                   control_path=[str(b), str(a)])
    assert ci.cache_key_digest(two.get_text_embedding_info_dict()) != first


def test_explicit_control_key_does_not_invent_dto_flip_or_crop(tmp_path):
    control = tmp_path / "control.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(control)
    cfg = _cfg()
    base = TextItem(tmp_path / "item.png", cfg, encode_control_in_text_embeddings=True,
                    control_path=str(control), crop_x=0, flip_x=False)
    changed = TextItem(tmp_path / "item.png", cfg, encode_control_in_text_embeddings=True,
                       control_path=str(control), crop_x=3, flip_x=True)
    assert ci.cache_key_digest(base.get_text_embedding_info_dict()) == \
        ci.cache_key_digest(changed.get_text_embedding_info_dict())


def test_dopsd_image_and_i2v_first_frame_keys_include_consumed_pixel_policy(tmp_path):
    source = tmp_path / "media.bin"
    source.write_bytes(b"same-bytes")
    cfg = _cfg()
    image_a = TextItem(source, cfg, crop_x=0, flip_x=False)
    image_b = TextItem(source, cfg, crop_x=2, flip_x=False)
    image_c = TextItem(source, cfg, crop_x=0, flip_x=True)
    a = ci.cache_key_digest(image_a.get_text_embedding_info_dict(dopsd_self_ref=True))
    b = ci.cache_key_digest(image_b.get_text_embedding_info_dict(dopsd_self_ref=True))
    c = ci.cache_key_digest(image_c.get_text_embedding_info_dict(dopsd_self_ref=True))
    assert len({a, b, c}) == 3
    policy = image_a.get_text_embedding_info_dict(dopsd_self_ref=True)["self_reference_identity"]
    assert policy["mode"] == "own_image_pixels"
    assert policy["crop"]["crop_x"] == 0 and policy["flip_x"] is False

    i2v_cfg = _cfg(do_i2v=True, num_frames=5)
    frame_a = TextItem(source, i2v_cfg, is_video=True, crop_x=0,
                       encode_first_frame_in_text_embeddings=True)
    frame_b = TextItem(source, i2v_cfg, is_video=True, crop_x=2,
                       encode_first_frame_in_text_embeddings=True)
    fa = ci.cache_key_digest(frame_a.get_text_embedding_info_dict())
    fb = ci.cache_key_digest(frame_b.get_text_embedding_info_dict())
    assert fa != fb
    frame_policy = frame_a.get_text_embedding_info_dict()["self_reference_identity"]
    assert frame_policy["mode"] == "own_video_first_frame_pixels"
    assert frame_policy["video_frame_policy"]["num_frames"] == 5


def test_dopsd_video_key_uses_raw_reference_policy_not_loader_pixels(tmp_path):
    source = tmp_path / "clip.bin"
    source.write_bytes(b"video-bytes")
    cfg = _cfg(num_frames=5, fps=24)
    first = TextItem(source, cfg, is_video=True, crop_x=0)
    crop_changed = TextItem(source, cfg, is_video=True, crop_x=9)
    longer = TextItem(source, _cfg(num_frames=7, fps=24), is_video=True, crop_x=0)
    a = ci.cache_key_digest(first.get_text_embedding_info_dict(dopsd_self_ref=True))
    b = ci.cache_key_digest(crop_changed.get_text_embedding_info_dict(dopsd_self_ref=True))
    c = ci.cache_key_digest(longer.get_text_embedding_info_dict(dopsd_self_ref=True))
    assert a == b
    assert a != c
    policy = first.get_text_embedding_info_dict(dopsd_self_ref=True)["self_reference_identity"]
    assert policy["mode"] == "raw_video_reference_path"
    assert "crop" not in policy and "flip_x" not in policy
    assert policy["dataset_temporal_policy"]["num_frames"] == 5


def test_text_key_keeps_model_space_override_and_encoder_digest(tmp_path):
    source = tmp_path / "item.png"
    source.write_bytes(b"x")
    cfg = _cfg()
    a = TextItem(source, cfg, text_embedding_space_version="space-a")
    b = TextItem(source, cfg, text_embedding_space_version="space-b")
    a.text_encoder_cache_identity = "encoder-a"
    c = TextItem(source, cfg, text_embedding_space_version="space-a")
    c.text_encoder_cache_identity = "encoder-b"
    assert ci.cache_key_digest(a.get_text_embedding_info_dict()) != \
        ci.cache_key_digest(b.get_text_embedding_info_dict())
    assert ci.cache_key_digest(a.get_text_embedding_info_dict()) != \
        ci.cache_key_digest(c.get_text_embedding_info_dict())


# ---------------------------------------------------------------- old schema lazy miss / no deletion

def test_legacy_unkeyed_latent_cache_is_reused_and_mismatched_key_misses(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(source)
    item = LatentItem(source, _cfg())
    expected_path = Path(item.get_latent_path(recalculate=True))
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"latent": torch.zeros(1)}, str(expected_path))  # legacy file without aitk_cache_key
    legacy_sibling = expected_path.parent / "old_unrelated.safetensors"
    legacy_sibling.write_bytes(b"legacy-do-not-delete")
    expected_digest = ci.cache_key_digest(item.get_latent_info_dict())

    # Legacy cache files missing aitk_cache_key metadata are accepted instead of forcing rebuild.
    ok, state = dm._read_latent_cache(str(expected_path), load=True, expected_cache_digest=expected_digest)
    assert ok is True and state is not None and "latent" in state

    # An explicit mismatched aitk_cache_key is still rejected and rebuilt cleanly.
    save_file({"latent": torch.zeros(1)}, str(expected_path), metadata={"aitk_cache_key": "stale-digest"})
    assert dm._read_latent_cache(str(expected_path), load=True, expected_cache_digest=expected_digest) == (False, None)
    assert legacy_sibling.read_bytes() == b"legacy-do-not-delete"

    LatentDataset([item]).cache_latents_all_latents()
    assert item.is_latent_cached
    with safe_open(str(expected_path), framework="pt") as f:
        assert f.metadata()["aitk_cache_key"] == expected_digest
    assert legacy_sibling.read_bytes() == b"legacy-do-not-delete"


# ---------------------------------------------------------------- one manifest digest per enabled cache kind

class _DummyFileItem:
    def __init__(self, **kwargs):
        self.path = kwargs["path"]
        self.dataset_config = kwargs["dataset_config"]
        self.kwargs = kwargs


def _init_dataset_for_identity_calls(tmp_path, monkeypatch, latent=False, text=False):
    root = tmp_path / "data"
    root.mkdir(parents=True)
    (root / "a.png").write_bytes(b"not decoded by this fixture")
    cfg = DatasetConfig(folder_path=str(root), cache_latents_to_disk=latent,
                        cache_text_embeddings=text, bucket_tolerance=8)
    calls = []

    def latent_builder(sd):
        calls.append("latent")
        return {"kind": "latent", "vae": "fixture"}

    def text_builder(sd):
        calls.append("text")
        return {"kind": "text", "te": "fixture"}

    monkeypatch.setattr(dl, "build_latent_encoder_identity", latent_builder)
    monkeypatch.setattr(dl, "build_text_encoder_identity", text_builder)
    monkeypatch.setattr(dl, "FileItemDTO", _DummyFileItem)
    monkeypatch.setattr(dl.AiToolkitDataset, "setup_epoch", lambda self: None)
    sd = SimpleNamespace(
        is_audio_model=False, is_multimodal_llm=False, is_xl=False, is_vega=False, is_ssd=False,
        encode_control_in_text_embeddings=False, encode_first_frame_in_text_embeddings=False,
        dopsd_self_ref=False, text_embedding_uses_target_size=False, te_padding_side="right",
        vae=SimpleNamespace(config=SimpleNamespace()), unet=SimpleNamespace(config=SimpleNamespace()),
        get_bucket_divisibility=lambda: 8,
        get_latent_space_version=lambda: "latent-v1",
        get_text_embedding_space_version=lambda: "text-v1",
        sample_rate=48000,
    )
    dataset = dl.AiToolkitDataset(cfg, batch_size=1, sd=sd)
    return dataset, calls


def test_encoder_identities_built_once_and_only_for_enabled_cache_kinds(tmp_path, monkeypatch):
    dataset, calls = _init_dataset_for_identity_calls(tmp_path, monkeypatch, latent=False, text=False)
    assert calls == [] and dataset.cache_encoder_fingerprints == {}
    dataset, calls = _init_dataset_for_identity_calls(tmp_path / "latent", monkeypatch, latent=True, text=False)
    assert calls == ["latent"] and set(dataset.cache_encoder_fingerprints) == {"latent"}
    item_kwargs = dataset.file_list[0].kwargs
    assert item_kwargs["latent_cache_encoder_identity"] == dataset.cache_encoder_fingerprints["latent"]
    assert isinstance(item_kwargs["latent_cache_encoder_identity"], str)
    assert item_kwargs["text_encoder_cache_identity"] is None
    dataset, calls = _init_dataset_for_identity_calls(tmp_path / "text", monkeypatch, latent=False, text=True)
    assert calls == ["text"] and set(dataset.cache_encoder_fingerprints) == {"text"}
    item_kwargs = dataset.file_list[0].kwargs
    assert item_kwargs["text_encoder_cache_identity"] == dataset.cache_encoder_fingerprints["text"]
    assert isinstance(item_kwargs["text_encoder_cache_identity"], str)
    assert item_kwargs["latent_cache_encoder_identity"] is None


# ---------------------------------------------------------------- DATA-02 same-bucket bounded retry

class _Bucket:
    def __init__(self, indices):
        self.file_list_idx = list(indices)


class _ReplacementDataset:
    def __init__(self, buckets, n=4):
        self.dataset_config = SimpleNamespace(buckets=True)
        self.file_list = list(range(n))
        self.buckets = buckets


def test_replacement_index_stays_in_the_source_bucket():
    ds = _ReplacementDataset({"a": _Bucket([0, 1, 2]), "b": _Bucket([3])})
    random.seed(7)
    pairs = [(index, dl.AiToolkitDataset._get_replacement_index(ds, index))
             for index in (0, 1, 2) for _ in range(20)]
    assert {pick for _, pick in pairs} == {0, 1, 2}
    assert all(pick != index for index, pick in pairs)


def test_singleton_and_missing_bucket_fail_clearly():
    ds = _ReplacementDataset({"only": _Bucket([2])})
    with pytest.raises(RuntimeError, match="no same-bucket replacement"):
        dl.AiToolkitDataset._get_replacement_index(ds, 2)
    ds = _ReplacementDataset({"a": _Bucket([0])})
    with pytest.raises(RuntimeError, match="No latent bucket contains index"):
        dl.AiToolkitDataset._get_replacement_index(ds, 9)


class _BrokenItem:
    calls = []

    def __init__(self):
        self.path = "broken.png"
        self.dataloader_transforms = None

    def load_and_process_image(self, transform):
        type(self).calls.append(self.path)
        raise OSError("broken fixture")


class _RetryDataset:
    _get_single_item = dl.AiToolkitDataset._get_single_item
    _get_replacement_index = dl.AiToolkitDataset._get_replacement_index
    _roll_crop_on_access = dl.AiToolkitDataset._roll_crop_on_access

    def __init__(self):
        self.dataset_config = SimpleNamespace(buckets=False)
        self.file_list = [_BrokenItem()]
        self.caption_dict = None
        self.transform = None
        self.is_caching_latents = False


def test_failed_item_retry_stays_bounded():
    ds = _RetryDataset()
    with pytest.raises(OSError, match="broken fixture"):
        ds._get_single_item(0)
    assert len(_BrokenItem.calls) == 11


# ---------------------------------------------------------------- CAP-01 dynamic captions vs cached embeds

class _CaptionItem:
    get_caption = dm.CaptionProcessingDTOMixin.get_caption

    def __init__(self, cfg):
        self.dataset_config = cfg
        self.raw_caption = "red, green, blue"
        self.trigger_word = None
        self.is_reg = False


@pytest.mark.parametrize("overrides", [
    {"shuffle_tokens": True},
    {"random_triggers": ["alpha", "beta"], "random_triggers_max": 2},
])
def test_cached_embeddings_refuse_dynamic_caption_redraws(overrides):
    dynamic = _CaptionItem(_cfg(**overrides))
    random.seed(11)
    draws = {dynamic.get_caption() for _ in range(30)}
    assert len(draws) > 1
    guard = SimpleNamespace(
        dataset_path="fixture",
        dataset_config=dynamic.dataset_config,
        file_list=[SimpleNamespace(path="a.png", dataset_config=dynamic.dataset_config)],
    )
    with pytest.raises(ValueError, match="shuffle_tokens or random_triggers"):
        dm.TextEmbeddingCachingMixin.cache_text_embeddings(guard)


class _HitTextItem:
    def __init__(self, folder, cfg):
        self.path = str(folder / "item.png")
        self.dataset_config = cfg
        self.caption = "static caption"
        self.encode_control_in_text_embeddings = False
        self.control_path = None
        self.control_video_paths = []
        self.is_video = False
        self.is_text_embedding_cached = False
        self._path = folder / "_t_e_cache" / "item.safetensors"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"text_embed": torch.zeros(1)}, str(self._path))

    def get_text_embedding_path(self, recalculate=False):
        return str(self._path)

    def get_blank_text_embedding_path(self, recalculate=False):
        return str(self._path)


def test_static_and_dropout_caption_cache_paths_remain_accepted(tmp_path):
    for caption_dropout_rate in (0.0, 0.25):
        cfg = _cfg(cache_text_embeddings=True, caption_dropout_rate=caption_dropout_rate)
        ds = SimpleNamespace(dataset_path="fixture", dataset_config=cfg,
                             file_list=[_HitTextItem(tmp_path / f"drop{caption_dropout_rate}", cfg)],
                             sd=SimpleNamespace(device="cpu",
                                                set_device_state_preset=lambda name: None))
        dm.TextEmbeddingCachingMixin.cache_text_embeddings(ds)
        assert ds.file_list[0].is_text_embedding_cached
