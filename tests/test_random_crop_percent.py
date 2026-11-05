"""random_crop_percent: config parsing, the crop window, and the per-access roll (works in loader workers, leaves
file_list and cached latents alone)."""
import random
from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader, Dataset

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset, check_random_crop_caching
from toolkit.dataloader_mixins import BucketsMixin


def test_config_percent_and_fraction():
    assert DatasetConfig(folder_path="x", random_crop_percent=6).random_crop_percent == pytest.approx(0.06)
    cfg = DatasetConfig(folder_path="x", random_crop_percent=0.06, cache_latents_to_disk=True)
    assert cfg.random_crop and cfg.random_crop_percent == pytest.approx(0.06)
    assert not cfg.cache_latents_to_disk and not cfg.cache_latents
    assert not DatasetConfig(folder_path="x").random_crop


class FakeItem(SimpleNamespace):
    def load_and_process_image(self, transform):
        pass

    def load_caption(self, captions):
        pass


def make_item(cfg, base=(1024, 768), crop=(1024, 768)):
    return FakeItem(dataset_config=cfg, scale_to_width=base[0], scale_to_height=base[1],
                    base_scale_to_width=base[0], base_scale_to_height=base[1],
                    crop_width=crop[0], crop_height=crop[1], crop_x=0, crop_y=0, path="fake.png",
                    dataloader_transforms=None)


class FakeDataset:
    roll_random_crop = BucketsMixin.roll_random_crop
    _roll_crop_on_access = AiToolkitDataset._roll_crop_on_access
    _get_single_item = AiToolkitDataset._get_single_item

    def __init__(self, cfg, caching=False, n=1):
        self.dataset_config = cfg
        self.is_caching_latents = caching
        self.file_list = [make_item(cfg) for _ in range(n)]
        self.caption_dict = None
        self.transform = None


def test_window_covers_94_to_100_percent():
    cfg = DatasetConfig(folder_path="x", random_crop_percent=0.06)
    ds = FakeDataset(cfg)
    random.seed(0)
    for _ in range(500):
        item = make_item(cfg)
        ds.roll_random_crop(item)
        coverage = item.crop_width / item.scale_to_width
        assert 0.94 - 1e-3 <= coverage <= 1.0
        assert item.crop_height / item.scale_to_height == pytest.approx(coverage, abs=2e-3)
        assert 0 <= item.crop_x <= item.scale_to_width - item.crop_width
        assert 0 <= item.crop_y <= item.scale_to_height - item.crop_height


def test_zero_percent_keeps_scale():
    cfg = DatasetConfig(folder_path="x", random_crop=True)
    ds = FakeDataset(cfg)
    item = make_item(cfg, base=(1100, 768))
    ds.roll_random_crop(item)
    assert (item.scale_to_width, item.scale_to_height) == (1100, 768)
    assert 0 <= item.crop_x <= 76 and item.crop_y == 0


def test_access_rolls_the_copy_only():
    cfg = DatasetConfig(folder_path="x", random_crop_percent=0.06)
    ds = FakeDataset(cfg)
    random.seed(1)
    items = [ds._get_single_item(0) for _ in range(20)]
    seen = {(i.scale_to_width, i.crop_x) for i in items}
    assert len(seen) > 5
    stored = ds.file_list[0]
    assert (stored.scale_to_width, stored.crop_x) == (1024, 0)


def test_no_roll_when_latents_are_cached_or_crop_off():
    cfg = DatasetConfig(folder_path="x", random_crop_percent=0.06)
    cached = FakeDataset(cfg, caching=True)
    assert all(cached._get_single_item(0).scale_to_width == 1024 for _ in range(20))
    off = FakeDataset(DatasetConfig(folder_path="x"))
    assert all(off._get_single_item(0).scale_to_width == 1024 for _ in range(20))
    # plain random_crop keeps its one setup crop (legacy behaviour)
    legacy = FakeDataset(DatasetConfig(folder_path="x", random_crop=True))
    assert all((i.scale_to_width, i.crop_x) == (1024, 0) for i in (legacy._get_single_item(0) for _ in range(20)))


def test_pool_that_caches_latents_rejects_percent_crop():
    cropped = DatasetConfig(folder_path="a", random_crop_percent=0.06)
    cached = DatasetConfig(folder_path="b", cache_latents_to_disk=True)
    with pytest.raises(ValueError, match="random_crop_percent"):
        check_random_crop_caching([cropped, cached], is_caching_latents=True)
    check_random_crop_caching([cropped], is_caching_latents=False)
    check_random_crop_caching([cached], is_caching_latents=True)


class WorkerDataset(Dataset):
    def __init__(self, fake):
        self.fake = fake

    def __len__(self):
        return 8

    def __getitem__(self, i):
        item = self.fake._get_single_item(0)
        return item.scale_to_width, item.crop_x, item.crop_y


def test_rolls_reach_persistent_workers():
    # the old per-epoch re-roll ran in the main process, so persistent workers kept their first crop
    fake = FakeDataset(DatasetConfig(folder_path="x", random_crop_percent=0.06))
    loader = DataLoader(WorkerDataset(fake), batch_size=None, num_workers=2, persistent_workers=True)
    epochs = [[tuple(int(v) for v in x) for x in loader] for _ in range(3)]
    assert epochs[0] != epochs[1] or epochs[1] != epochs[2]


class BucketFake(BucketsMixin):
    def __init__(self, cfg, sizes):
        cfg.bucket_tolerance = 16  # the trainer sets this from the model's divisibility
        self.dataset_config = cfg
        self.batch_size = 1
        self.file_list = [FakeItem(dataset_config=cfg, width=w, height=h, is_audio_model=False, num_frames=1,
                                   path=f"{i}.png") for i, (w, h) in enumerate(sizes)]
        self.epoch_num = 0
        self.dataset_path = "fake"
        self.is_video = False


SIZES = [(1024, 1024), (1365, 1024), (1024, 1365), (1200, 900), (768, 1024)]


@pytest.mark.parametrize("square", [False, True])
def test_setup_buckets_plain_random_crop_matches_main(monkeypatch, square):
    cfg = DatasetConfig(folder_path="x", random_crop=True, square_crop=square, resolution=1024)
    calls = []
    real = random.randint
    monkeypatch.setattr(random, "randint", lambda a, b: calls.append((a, b)) or real(a, b))
    ds = BucketFake(cfg, SIZES)
    ds.setup_buckets(quiet=True)
    if square:
        # main centres square crops even with random_crop on
        assert calls == []
        for item in ds.file_list:
            assert item.crop_x == max(0, int(item.scale_to_width / 2 - 1024 / 2))
    else:
        # main draws x and y for every item, zero slack included
        assert len(calls) == 2 * len(SIZES)
    assert all(not hasattr(i, "dataset_config") or i.dataset_config is cfg for i in ds.file_list)


def test_setup_buckets_percent_crop_window():
    cfg = DatasetConfig(folder_path="x", random_crop_percent=0.06, resolution=1024)
    random.seed(3)
    ds = BucketFake(cfg, SIZES)
    ds.setup_buckets(quiet=True)
    for item in ds.file_list:
        # the zoom over the bucket-fitted scale is at most 1 / 0.94 (bucket slack comes on top)
        assert 1.0 <= item.scale_to_width / item.base_scale_to_width <= 1 / 0.94 + 2e-3
        assert 1.0 <= item.scale_to_height / item.base_scale_to_height <= 1 / 0.94 + 2e-3
        assert 0 <= item.crop_x <= item.scale_to_width - item.crop_width
        assert 0 <= item.crop_y <= item.scale_to_height - item.crop_height
        assert (item.base_scale_to_width, item.base_scale_to_height) != (None, None)
