import os

import pytest
import torch
from PIL import Image

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset, get_dataloader_from_datasets, trigger_dataloader_setup_epoch
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


class FakeSd:
    """Small dependency surface needed by the image dataset and latent cacher."""

    is_audio_model = False
    is_multimodal_llm = False
    is_xl = False
    is_vega = False
    is_ssd = False
    load_rgba = False
    cache_latents_as_uint8 = False
    torch_dtype = torch.float32
    device_torch = torch.device("cpu")
    device = "cpu"
    te_padding_side = "right"
    encode_control_in_text_embeddings = False
    encode_first_frame_in_text_embeddings = False
    text_embedding_uses_target_size = False
    dopsd_self_ref = False
    use_raw_control_images = False
    supports_video_control_images = False
    has_multiple_control_images = False
    vae = None
    unet = None

    def get_bucket_divisibility(self):
        return 8

    def get_latent_space_version(self):
        return "fake-vae"

    def get_text_embedding_space_version(self):
        return "fake-te"

    def save_device_state(self):
        pass

    def set_device_state_preset(self, *args, **kwargs):
        pass

    def restore_device_state(self):
        pass

    def encode_images(self, imgs):
        # Keep the chosen/rejected arms distinguishable while preserving H/W.
        return imgs[:, :1].repeat(1, 4, 1, 1)


def make_tree(tmp_path, names, size=(96, 64), chosen_color=(255, 20, 10), rejected_color=(10, 20, 255)):
    chosen = tmp_path / "chosen"
    rejected = tmp_path / "rejected"
    chosen.mkdir(parents=True)
    rejected.mkdir(parents=True)
    for name in names:
        chosen_path = chosen / name
        rejected_path = rejected / name
        chosen_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, chosen_color).save(chosen_path)
        Image.new("RGB", size, rejected_color).save(rejected_path)
        chosen_path.with_suffix(".txt").write_text("chosen caption")
        rejected_path.with_suffix(".txt").write_text("rejected caption must be ignored")
    return chosen, rejected


def config_for(chosen, rejected, **kwargs):
    options = dict(
        folder_path=str(chosen),
        rejected_path=str(rejected),
        resolution=64,
        buckets=True,
        caption_ext="txt",
    )
    options.update(kwargs)
    return DatasetConfig(**options)


def test_rejected_path_is_optional_and_validated_as_separate_tree(tmp_path):
    assert DatasetConfig(folder_path="x").rejected_path is None
    assert DatasetConfig(folder_path="x", rejected_path="").rejected_path is None

    chosen, rejected = make_tree(tmp_path, ["a.png"])
    with pytest.raises(ValueError, match="must not be the same"):
        AiToolkitDataset(config_for(chosen, chosen), batch_size=1, sd=None)
    with pytest.raises(ValueError, match="separate sibling tree"):
        AiToolkitDataset(config_for(chosen, chosen / "nested"), batch_size=1, sd=None)
    with pytest.raises(ValueError, match="separate sibling tree"):
        AiToolkitDataset(config_for(chosen, tmp_path), batch_size=1, sd=None)


def test_missing_or_extra_mates_abort_the_whole_scan_with_paths(tmp_path):
    chosen, rejected = make_tree(tmp_path, ["a.png", "sub/b.png"])
    (rejected / "a.png").unlink()
    with pytest.raises(ValueError, match=r"Missing rejected mate.*a\.png"):
        AiToolkitDataset(config_for(chosen, rejected), batch_size=1, sd=None)

    chosen2, rejected2 = make_tree(tmp_path / "extra", ["a.png"])
    Image.new("RGB", (96, 64), (0, 255, 0)).save(rejected2 / "orphan.png")
    with pytest.raises(ValueError, match=r"no chosen mate.*orphan\.png"):
        AiToolkitDataset(config_for(chosen2, rejected2), batch_size=1, sd=None)


def test_matching_requires_same_extension_not_just_stem(tmp_path):
    chosen, rejected = make_tree(tmp_path, ["a.png"])
    (rejected / "a.png").unlink()
    Image.new("RGB", (96, 64), (0, 255, 0)).save(rejected / "a.jpg")
    with pytest.raises(ValueError, match=r"Missing rejected mate.*a\.png"):
        AiToolkitDataset(config_for(chosen, rejected), batch_size=1, sd=None)


@pytest.mark.parametrize("contents", [
    "{", "[]", '{"version":2,"preference_probability":0.8}',
    '{"version":true,"preference_probability":1}', '{"version":1}',
    '{"version":1,"preference_probability":true}',
    '{"version":1,"preference_probability":"0.7"}',
    '{"version":1,"preference_probability":NaN}',
    '{"version":1,"preference_probability":0.49}',
    '{"version":1,"preference_probability":1.01}',
])
def test_invalid_preference_sidecar_names_the_bad_file(tmp_path, contents):
    chosen, rejected = make_tree(tmp_path, ["sub/a.png"])
    (chosen / "sub/a.dpo.json").write_text(contents)
    with pytest.raises(ValueError, match=r"a\.dpo\.json"):
        AiToolkitDataset(config_for(chosen, rejected), batch_size=1, sd=None)


def test_exif_oriented_dimensions_are_compared_and_mismatch_is_rejected(tmp_path):
    chosen = tmp_path / "chosen"
    rejected = tmp_path / "rejected"
    chosen.mkdir()
    rejected.mkdir()

    # Chosen raw 100x60 with orientation 6 becomes oriented 60x100.
    img = Image.new("RGB", (100, 60), (255, 0, 0))
    exif = Image.Exif()
    exif[274] = 6
    img.save(chosen / "a.png", exif=exif)
    # Rejected raw 60x100, no orientation, is also oriented 60x100.
    Image.new("RGB", (60, 100), (0, 0, 255)).save(rejected / "a.png")

    ds = AiToolkitDataset(config_for(chosen, rejected, fast_image_size=True), batch_size=1, sd=None)
    item = ds.file_list[0]
    assert (item.width, item.height) == (60, 100)
    assert (item.rejected_item.width, item.rejected_item.height) == (60, 100)

    # Same raw dimensions, but one image is rotated by EXIF. Oriented sizes
    # differ, so this is not a trainable preference pair.
    broken = tmp_path / "broken"
    broken_chosen = broken / "chosen"
    broken_rejected = broken / "rejected"
    broken_chosen.mkdir(parents=True)
    broken_rejected.mkdir(parents=True)
    img = Image.new("RGB", (100, 60), (255, 0, 0))
    img.save(broken_chosen / "a.png", exif=exif)
    Image.new("RGB", (100, 60), (0, 0, 255)).save(broken_rejected / "a.png")
    with pytest.raises(ValueError, match=r"Invalid DPO pair.*EXIF-oriented dimensions differ"):
        AiToolkitDataset(config_for(broken_chosen, broken_rejected), batch_size=1, sd=None)


def test_corrupt_pair_scan_fails_with_useful_path(tmp_path):
    chosen, rejected = make_tree(tmp_path, ["a.png", "b.png"])
    (rejected / "b.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match=r"Invalid DPO pair.*chosen=.*b\.png.*rejected=.*b\.png"):
        AiToolkitDataset(config_for(chosen, rejected), batch_size=1, sd=None)


def test_bucket_geometry_is_symmetric_and_caption_is_chosen_only(tmp_path):
    chosen, rejected = make_tree(tmp_path, ["a.png", "sub/b.png"])
    (chosen / "sub/b.dpo.json").write_text('{"version":1,"preference_probability":0.75}')
    ds = AiToolkitDataset(
        config_for(chosen, rejected, resolution=64, buckets=True, random_crop=True, flip_x=True),
        batch_size=2,
        sd=None,
    )
    # flip_x creates a second item for every pair.
    assert len(ds.file_list) == 4
    geometry_attrs = (
        "scale_to_width", "scale_to_height", "base_scale_to_width",
        "base_scale_to_height", "crop_x", "crop_y", "crop_width",
        "crop_height", "flip_x", "flip_y",
    )
    for item in ds.file_list:
        assert item.preference_probability == (0.75 if item.path.endswith("b.png") else 1.0)
        mate = item.rejected_item
        assert mate is not None
        assert mate.path != item.path
        for attr in geometry_attrs:
            assert getattr(mate, attr) == getattr(item, attr), attr
        assert item.tensor is None

    loaded = ds._get_single_item(0)
    assert loaded.caption == "chosen caption"
    assert loaded.rejected_item.raw_caption is None
    assert loaded.tensor.shape == loaded.rejected_item.tensor.shape
    assert not torch.equal(loaded.tensor, loaded.rejected_item.tensor)

    batch = DataLoaderBatchDTO(file_items=[loaded])
    assert batch.tensor.shape == batch.rejected_tensor.shape
    assert not torch.equal(batch.tensor, batch.rejected_tensor)
    assert batch.latents is None
    assert batch.rejected_latents is None


def test_per_access_random_crop_rolls_both_arms_together(tmp_path):
    chosen, rejected = make_tree(tmp_path, ["a.png"], size=(160, 96))
    ds = AiToolkitDataset(
        config_for(chosen, rejected, resolution=64, random_crop_percent=0.2),
        batch_size=1,
        sd=None,
    )
    geometry_attrs = (
        "scale_to_width", "scale_to_height", "base_scale_to_width",
        "base_scale_to_height", "crop_x", "crop_y", "crop_width",
        "crop_height", "flip_x", "flip_y",
    )
    seen = set()
    for _ in range(8):
        item = ds._get_single_item(0)
        mate = item.rejected_item
        for attr in geometry_attrs:
            assert getattr(mate, attr) == getattr(item, attr), attr
        seen.add((item.scale_to_width, item.scale_to_height, item.crop_x, item.crop_y))
    assert len(seen) > 1  # the crop is actually re-rolled and still mirrored


def test_load_never_substitutes_a_corrupt_pair(tmp_path, monkeypatch):
    chosen, rejected = make_tree(tmp_path, ["a.png", "b.png"])
    ds = AiToolkitDataset(config_for(chosen, rejected), batch_size=1, sd=None)
    (rejected / "a.png").write_bytes(b"corrupt after scan")

    def forbidden_replacement(index):
        raise AssertionError("DPO load must not substitute another item")

    monkeypatch.setattr(ds, "_get_replacement_index", forbidden_replacement)
    bad_index = next(i for i, item in enumerate(ds.file_list) if item.rejected_path == str(rejected / "a.png"))
    with pytest.raises(RuntimeError, match=r"DPO pair failed to load: chosen=.*a\.png, rejected=.*a\.png"):
        ds._get_single_item(bad_index)


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_native_latent_caching_caches_both_arms_independently(tmp_path, mode):
    chosen, rejected = make_tree(tmp_path / mode, ["a.png"])
    cfg = config_for(
        chosen,
        rejected,
        cache_latents=(mode == "memory"),
        cache_latents_to_disk=(mode == "disk"),
    )
    ds = AiToolkitDataset(cfg, batch_size=1, sd=FakeSd())
    item = ds._get_single_item(0)

    assert item.is_latent_cached
    assert item.rejected_item.is_latent_cached
    assert item.get_latent_path() != item.rejected_item.get_latent_path()
    assert "dpo_pair_signature" in item.get_latent_info_dict()
    assert item.get_latent_info_dict()["dpo_pair_signature"] == item.rejected_item.get_latent_info_dict()["dpo_pair_signature"]
    assert item.rejected_item.get_latent_info_dict()["dpo_arm"] == "rejected"
    if mode == "disk":
        assert os.path.exists(item.get_latent_path())
        assert os.path.exists(item.rejected_item.get_latent_path())

    batch = DataLoaderBatchDTO(file_items=[item])
    assert batch.tensor is None
    assert batch.rejected_tensor is None
    assert batch.latents.shape == batch.rejected_latents.shape
    assert not torch.equal(batch.latents, batch.rejected_latents)


def test_bucketed_dataloader_emits_separate_rejected_arm(tmp_path):
    chosen, rejected = make_tree(tmp_path, ["a.png", "b.png"])
    cfg = config_for(
        chosen,
        rejected,
        batch_size=2,
        num_workers=0,
        cache_latents_to_disk=True,
    )
    loader = get_dataloader_from_datasets([cfg], batch_size=2, sd=FakeSd())
    trigger_dataloader_setup_epoch(loader)
    batch = next(iter(loader))
    assert batch.tensor is None
    assert batch.rejected_tensor is None
    assert batch.latents.shape == batch.rejected_latents.shape
    assert batch.latents.shape[0] == 2
    assert len(batch.file_items) == 2


def test_replacing_rejected_bytes_invalidates_both_caches_even_with_identical_stat(tmp_path):
    chosen, rejected = make_tree(tmp_path, ["a.png"])
    mate = rejected / "a.png"
    Image.new("RGB", (96, 64), (0, 0, 1)).save(mate, compress_level=0)
    cfg = config_for(chosen, rejected, cache_latents_to_disk=True)
    original = AiToolkitDataset(cfg, batch_size=1, sd=FakeSd()).file_list[0]
    old_stat = mate.stat()
    Image.new("RGB", (96, 64), (230, 0, 1)).save(mate, compress_level=0)
    os.utime(mate, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert mate.stat().st_size == old_stat.st_size
    updated = AiToolkitDataset(cfg, batch_size=1, sd=FakeSd()).file_list[0]
    assert original.get_latent_path() != updated.get_latent_path()
    assert original.rejected_item.get_latent_path() != updated.rejected_item.get_latent_path()


def test_unsupported_single_arm_aux_options_are_rejected_explicitly(tmp_path):
    chosen, rejected = make_tree(tmp_path, ["a.png"])
    cfg = config_for(
        chosen,
        rejected,
        augmentations=[{"method": "HorizontalFlip", "params": {"p": 1.0}}],
    )
    with pytest.raises(ValueError, match=r"unsupported dataset option\(s\): augmentations"):
        AiToolkitDataset(cfg, batch_size=1, sd=None)


def test_cached_pixel_tensors_are_returned_for_both_arms(tmp_path):
    chosen, rejected = make_tree(tmp_path, ["a.png"])
    cfg = config_for(
        chosen,
        rejected,
        cache_latents_to_disk=True,
        cache_tensors_to_disk=True,
    )
    ds = AiToolkitDataset(cfg, batch_size=1, sd=FakeSd())
    batch = DataLoaderBatchDTO(file_items=[ds._get_single_item(0)])
    assert batch.latents is not None
    assert batch.rejected_latents is not None
    assert batch.tensor is not None
    assert batch.rejected_tensor is not None
    assert batch.tensor.shape == batch.rejected_tensor.shape


def test_ordinary_dataset_behavior_is_unchanged(tmp_path):
    folder = tmp_path / "ordinary"
    folder.mkdir()
    Image.new("RGB", (96, 64), (255, 0, 0)).save(folder / "a.png")
    (folder / "a.txt").write_text("ordinary caption")
    ds = AiToolkitDataset(DatasetConfig(folder_path=str(folder), resolution=64), batch_size=1, sd=None)
    assert len(ds.file_list) == 1
    item = ds._get_single_item(0)
    assert item.rejected_item is None
    assert item.caption == "ordinary caption"
    batch = DataLoaderBatchDTO(file_items=[item])
    assert batch.rejected_tensor is None
    assert batch.rejected_latents is None
