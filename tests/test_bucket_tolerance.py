import json
import os
from types import SimpleNamespace

import pytest
from PIL import Image

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.dataloader_mixins import BucketsMixin

# crop-variant style sizes: all below 768x768 pixels, so none are downscaled to a shared bucket
SIZES = [(512, 704), (528, 720), (544, 736), (560, 752), (576, 704), (592, 688),
         (600, 712), (616, 728), (520, 700), (540, 690), (700, 520), (710, 540)]
REAL_SIZES = r'E:/Modding/SlayTheSpire2-CG/datasets/pilot-v2-crop/train/.aitk_size.json'


class _Sd:
    def __init__(self, div):
        self.div = div

    def get_bucket_divisibility(self):
        return self.div


class _StubDataset(BucketsMixin):
    def __init__(self, sizes, config, batch_size=4):
        super().__init__()
        self.dataset_config = config
        self.dataset_path = 'stub'
        self.epoch_num = 0
        self.batch_size = batch_size
        self.is_video = False
        self.file_list = [SimpleNamespace(width=w, height=h, dataset_config=config, is_audio_model=False)
                          for w, h in sizes]


def _resolved_tolerance(div, **kwargs):
    # runs the real AiToolkitDataset.__init__ tolerance block without scanning a folder
    config = DatasetConfig(folder_path='unused', resolution=768, **kwargs)
    ds = AiToolkitDataset.__new__(AiToolkitDataset)
    try:
        AiToolkitDataset.__init__(ds, config, batch_size=4, sd=_Sd(div) if div else None)
    except Exception:
        pass  # folder scan fails after the tolerance block; only the tolerance matters here
    return config.bucket_tolerance


def _buckets(sizes, tolerance, resolution=768):
    config = DatasetConfig(folder_path='unused', resolution=resolution)
    config.bucket_tolerance = tolerance
    ds = _StubDataset(sizes, config)
    ds.setup_buckets(quiet=True)
    return ds


def test_unset_tolerance_keeps_model_divisibility():
    assert DatasetConfig(folder_path='unused').bucket_tolerance is None
    assert _resolved_tolerance(16) == 16
    assert _resolved_tolerance(32) == 32


def test_user_tolerance_kept_and_rounded_up_to_divisibility():
    assert _resolved_tolerance(16, bucket_tolerance=64) == 64
    assert _resolved_tolerance(16, bucket_tolerance=20) == 32
    assert _resolved_tolerance(32, bucket_tolerance=16) == 32


def test_no_model_does_not_crash(tmp_path):
    for i, (w, h) in enumerate(SIZES[:3]):
        Image.new('RGB', (w, h), (i * 40, 80, 120)).save(tmp_path / f'img{i}.png')
        (tmp_path / f'img{i}.txt').write_text('a cat')
    config = DatasetConfig(folder_path=str(tmp_path), resolution=512, bucket_tolerance=20)
    ds = AiToolkitDataset(config, batch_size=1, sd=None)
    assert config.bucket_tolerance == 24  # rounded up to the no-model divisibility of 8
    assert len(ds.file_list) == 3


def test_coarser_tolerance_merges_buckets_and_cuts_padding():
    fine = _buckets(SIZES, 16)
    coarse = _buckets(SIZES, 64)
    assert len(coarse.buckets) < len(fine.buckets)
    for ds, tol in ((fine, 16), (coarse, 64)):
        for b in ds.buckets.values():
            assert b.width % tol == 0 and b.height % tol == 0
    pad = lambda ds: sum(len(b) for b in ds.batch_indices) - len(SIZES)
    assert pad(coarse) < pad(fine)


@pytest.mark.skipif(not os.path.exists(REAL_SIZES), reason='real MOSS size database not present')
def test_real_moss_dataset_bucket_counts():
    folder = os.path.dirname(REAL_SIZES)
    files = {f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))}
    with open(REAL_SIZES) as f:
        db = json.load(f)
    sizes = [(v[0], v[1]) for k, v in db.items() if k != '__version__' and k.lstrip('\\') in files]
    counts = {tol: [len(_buckets(sizes, tol, res).buckets) for res in (768, 1024)] for tol in (16, 64)}
    assert counts[16] == [14, 110]
    assert counts[64] == [3, 19]
