"""A truncated latent cache must be rebuilt from its source, not trusted or dropped (docs/STATE_MACHINES.md M8, SM-07).

A process killed while saving a latent (Kill button, stop backstop, crash) left a truncated file at the final name.
In-memory caching then failed to load it and removed the item from the dataset on every run; disk-only caching
marked it cached and training crashed when the item was fetched. Real LatentCachingMixin.cache_latents_all_latents
and LatentCachingFileItemDTOMixin (path, info dict, get_latent); the encoder is a deterministic CPU fixture and the
source is a real PNG. CPU only.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open

from toolkit import dataloader_mixins as dm


class Item(dm.LatentCachingFileItemDTOMixin):
    def __init__(self, path):
        super().__init__()
        self.path = str(path)
        self.dataset_config = SimpleNamespace(auto_frame_count=False, num_frames=1, fps=24, do_i2v=False,
                                              do_audio=False, cache_tensors_to_disk=False, buckets=False)
        self.scale_to_width = self.scale_to_height = self.crop_width = self.crop_height = 16
        self.crop_x = self.crop_y = 0
        self.latent_space_version = 'fixture'
        self.flip_x = self.flip_y = self.is_video = self.is_audio_model = self.load_rgba = False
        self.audio_data = None
        self.tensor = None
        self.dataloader_transforms = None
        self.loads = 0

    def load_and_process_image(self, transform, only_load_latents=False):
        self.loads += 1
        pixels = np.asarray(Image.open(self.path).convert('RGB'), dtype=np.float32) / 127.5 - 1
        self.tensor = torch.from_numpy(pixels).permute(2, 0, 1)

    def cleanup(self):
        self.tensor = None


class Dataset(dm.LatentCachingMixin):
    def __init__(self, items, to_disk, to_memory):
        self.latent_cache = {}
        self.file_list = list(items)
        self.dataset_path = 'fixture'
        self.is_caching_latents_to_disk = to_disk
        self.is_caching_latents_to_memory = to_memory
        self.dataset_config = SimpleNamespace(cache_latents_num_workers=1, buckets=False)
        self.transform = None
        self.device_calls = []
        self.sd = SimpleNamespace(
            torch_dtype=torch.float32, device_torch=torch.device('cpu'), device='cpu',
            encode_images=lambda imgs: torch.nn.functional.avg_pool2d(imgs, 4),
            save_device_state=lambda: self.device_calls.append('save'),
            set_device_state_preset=lambda name: self.device_calls.append(name),
            restore_device_state=lambda: self.device_calls.append('restore'))

    def moves(self):
        # the explicit save_device_state call is the fixed code's (a real set_device_state_preset saves first too)
        return [c for c in self.device_calls if c != 'save']


def make_items(tmp_path, n=3):
    items = []
    for i in range(n):
        path = tmp_path / f'img{i}.png'
        Image.fromarray(np.full((16, 16, 3), 40 * i, dtype=np.uint8)).save(path)
        items.append(Item(path))
    return items


def truncate_cache_of(item, tmp_path):
    """Run one honest caching pass for this item, then cut its cache file as a kill mid-write would."""
    Dataset([item], to_disk=True, to_memory=False).cache_latents_all_latents()
    path = Path(item.get_latent_path(recalculate=True))
    good = path.read_bytes()
    (tmp_path / 'good.safetensors').write_bytes(good)
    path.write_bytes(good[:len(good) // 2])
    item.is_latent_cached = False
    item._encoded_latent = None
    return path, good


@pytest.mark.parametrize('to_disk,to_memory', [(True, False), (True, True), (False, True)],
                         ids=['disk-only', 'disk+memory', 'memory-only'])
def test_truncated_cache_is_rebuilt_from_the_source(tmp_path, to_disk, to_memory):
    items = make_items(tmp_path)
    victim = items[1]
    path, good = truncate_cache_of(victim, tmp_path)
    truncated = path.read_bytes()
    source = Path(victim.path).read_bytes()
    victim.loads = 0
    dataset = Dataset(items, to_disk=to_disk, to_memory=to_memory)
    dataset.cache_latents_all_latents()

    assert dataset.file_list == items  # the item keeps its place (and bucket)
    assert victim.is_latent_cached and victim.loads == 1  # re-encoded from its source image
    assert Path(victim.path).read_bytes() == source
    assert dataset.moves() == ['cache_latents', 'restore']
    expected = dataset.sd.encode_images(_pixels(victim.path)).squeeze(0)
    if to_disk:
        # the same file an uninterrupted pass wrote (the header's metadata map has no fixed byte order)
        with safe_open(str(path), framework='pt') as f, safe_open(str(tmp_path / 'good.safetensors'), framework='pt') as g:
            assert f.metadata() == g.metadata() and set(f.keys()) == set(g.keys()) == {'latent'}
            assert torch.equal(f.get_tensor('latent'), expected) and torch.equal(g.get_tensor('latent'), expected)
    else:
        assert path.read_bytes() == truncated  # not caching to disk: the file is not this run's to rewrite
    victim.cleanup_latent()  # what the dataloader does between epochs
    assert torch.equal(victim.get_latent(), expected)
    assert not list(path.parent.glob('.*.tmp'))


def _pixels(path):
    pixels = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def test_interrupted_cache_write_never_leaves_a_partial_file_at_the_cache_name(tmp_path, monkeypatch):
    items = make_items(tmp_path, 1)
    real_save = dm.save_file

    def killed_mid_write(state_dict, filename, metadata=None):
        real_save(state_dict, filename, metadata=metadata)
        data = Path(filename).read_bytes()
        Path(filename).write_bytes(data[:len(data) // 3])
        raise OSError('process killed while writing the cache')
    monkeypatch.setattr(dm, 'save_file', killed_mid_write)
    with pytest.raises(OSError, match='killed'):
        Dataset(items, to_disk=True, to_memory=False).cache_latents_all_latents()
    path = Path(items[0].get_latent_path(recalculate=True))
    assert not path.exists() and not list(path.parent.glob('*'))


# ---------------------------------------------------------------- R04: the borrowed device preset is given back
# Same real caching pass and PNG fixture; the holder's device-preset methods are recording stubs.

@pytest.mark.parametrize('failure', ['encode', 'publish'])
def test_device_preset_is_given_back_when_encoding_or_publishing_fails(tmp_path, monkeypatch, failure):
    items = make_items(tmp_path, 2)
    dataset = Dataset(items, to_disk=True, to_memory=False)
    if failure == 'encode':
        dataset.sd.encode_images = lambda imgs: (_ for _ in ()).throw(RuntimeError('encoder failed'))
        expected = RuntimeError
    else:
        monkeypatch.setattr(dm, 'save_file', lambda *a, **kw: (_ for _ in ()).throw(OSError('disk full')))
        expected = OSError
    with pytest.raises(expected):
        dataset.cache_latents_all_latents()
    assert dataset.moves() == ['cache_latents', 'restore']


def test_a_preset_that_fails_part_way_is_put_back_once(tmp_path):
    items = make_items(tmp_path, 1)
    dataset = Dataset(items, to_disk=True, to_memory=False)
    sd = dataset.sd
    sd.save_device_state = lambda: setattr(sd, 'device_state', 'entry snapshot')

    def preset(name):
        sd.device_state = 'half-written re-save'
        raise torch.cuda.OutOfMemoryError('CUDA out of memory (injected moving the VAE)')
    sd.set_device_state_preset = preset
    sd.restore_device_state = lambda: dataset.device_calls.append(('restore', sd.device_state))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        dataset.cache_latents_all_latents()
    assert dataset.device_calls == [('restore', 'entry snapshot')]


def test_a_failing_restore_does_not_hide_the_encoding_error(tmp_path):
    items = make_items(tmp_path, 1)
    dataset = Dataset(items, to_disk=True, to_memory=False)
    dataset.sd.encode_images = lambda imgs: (_ for _ in ()).throw(RuntimeError('encoder failed'))
    dataset.sd.restore_device_state = lambda: (_ for _ in ()).throw(RuntimeError('restore failed'))
    with pytest.raises(RuntimeError, match='encoder failed') as caught:
        dataset.cache_latents_all_latents()
    assert any('restore failed' in note for note in caught.value.__notes__)


def test_a_fully_cached_dataset_never_borrows_the_device(tmp_path):
    items = make_items(tmp_path, 2)
    Dataset(items, to_disk=True, to_memory=False).cache_latents_all_latents()
    for item in items:
        item.is_latent_cached = False
    again = Dataset(items, to_disk=True, to_memory=False)
    again.cache_latents_all_latents()
    assert again.device_calls == []


@pytest.mark.parametrize('failing', ['shutdown', 'close'])
@pytest.mark.parametrize('body_fails', [True, False])
def test_device_restore_survives_a_failing_executor_shutdown_or_progress_close(tmp_path, monkeypatch, failing,
                                                                              body_fails):
    items = make_items(tmp_path, 1)
    dataset = Dataset(items, to_disk=True, to_memory=False)
    if body_fails:
        monkeypatch.setattr(dm, 'save_file', lambda *a, **kw: (_ for _ in ()).throw(OSError('disk full')))
    if failing == 'shutdown':  # e.g. Ctrl+C while waiting for a prep thread
        real = dm.ThreadPoolExecutor.shutdown

        def shutdown(self, *a, **kw):
            real(self, *a, **kw)
            raise KeyboardInterrupt('interrupted while joining prep threads')
        monkeypatch.setattr(dm.ThreadPoolExecutor, 'shutdown', shutdown)
        cleanup_error = KeyboardInterrupt
    else:
        monkeypatch.setattr(dm.tqdm, 'close', lambda self: (_ for _ in ()).throw(OSError('console closed')))
        cleanup_error = OSError
    with pytest.raises(OSError if body_fails else cleanup_error) as caught:
        dataset.cache_latents_all_latents()
    if body_fails:
        assert 'disk full' in str(caught.value)  # the pass's own error stays the error
    assert dataset.moves() == ['cache_latents', 'restore']
