"""Text-embedding and CLIP-vision caches: a truncated file is rebuilt, a killed write never leaves a partial file at
the cache name, and the CLIP pass gives back its device preset (docs/STATE_MACHINES.md M16, P-01; R04 family).
Real TextEmbeddingCachingMixin.cache_text_embeddings / CLIPCachingMixin.cache_clip_vision_to_disk, real PromptEmbeds
save/load. Stubbed: the text encoder (a deterministic CPU function of the caption), the item's cache path (fixed per
item instead of the hashed path), the holder's device-preset methods (recording stubs). CPU only.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from toolkit import dataloader_mixins as dm
from toolkit import prompt_utils
from toolkit.prompt_utils import PromptEmbeds


class Item:
    def __init__(self, folder, i):
        self.path = str(folder / f'img{i}.png')
        self.caption = f'caption {i}'
        self.dataset_config = SimpleNamespace(diff_output_preservation=False, caption_dropout_rate=0,
                                              random_crop_percent=0, do_i2v=False)
        self.encode_control_in_text_embeddings = False
        self.control_path = None
        self.is_video = False
        self.is_text_embedding_cached = False
        self._te_path = str(folder / '_t_e_cache' / f'img{i}.safetensors')

    def get_text_embedding_path(self, recalculate=False):
        return self._te_path


def embed(caption):
    value = float(sum(map(ord, caption)))
    return PromptEmbeds(torch.full((1, 3, 4), value))


class Dataset(dm.TextEmbeddingCachingMixin):
    def __init__(self, items):
        self.file_list = items
        self.dataset_path = 'fixture'
        self.encoded = []

        def encode_prompt(captions, **kwargs):
            batch = captions if isinstance(captions, list) else [captions]
            if len(batch) == 1:  # plain PromptEmbeds cannot be split: the pass retries a batch one prompt at a time
                self.encoded.append(batch[0])
            return embed(batch[0])
        self.sd = SimpleNamespace(device='cpu', encode_prompt=encode_prompt,
                                  set_device_state_preset=lambda name: None)


def test_a_truncated_text_embedding_is_re_encoded(tmp_path):
    items = [Item(tmp_path, i) for i in range(2)]
    Dataset(items).cache_text_embeddings()
    victim = Path(items[1]._te_path)
    data = victim.read_bytes()
    victim.write_bytes(data[:len(data) // 2])  # a kill mid-write by a version without atomic publication
    again = Dataset(items)
    again.cache_text_embeddings()
    assert again.encoded == ['caption 1']  # only the unreadable one
    assert torch.equal(PromptEmbeds.load(str(victim)).text_embeds, embed('caption 1').text_embeds)


def test_a_killed_text_embedding_write_leaves_no_partial_file(tmp_path, monkeypatch):
    items = [Item(tmp_path, 0)]
    real_save = prompt_utils.save_file

    def killed(state_dict, filename, metadata=None):
        real_save(state_dict, filename, metadata=metadata)
        Path(filename).write_bytes(Path(filename).read_bytes()[:10])
        raise OSError('process killed while writing the cache')
    monkeypatch.setattr(prompt_utils, 'save_file', killed)
    with pytest.raises(OSError, match='killed'):
        Dataset(items).cache_text_embeddings()
    folder = Path(items[0]._te_path).parent
    assert not Path(items[0]._te_path).exists() and not list(folder.glob('*'))


def test_clip_vision_caching_gives_back_the_device_preset_when_it_fails():
    calls = []
    sd = SimpleNamespace(adapter=None,  # the pass refuses to run without an adapter, after taking the preset
                         save_device_state=lambda: calls.append('save'),
                         set_device_state_preset=lambda name: calls.append(name),
                         restore_device_state=lambda: calls.append('restore'))
    dataset = SimpleNamespace(is_caching_clip_vision_to_disk=True, dataset_path='fixture', sd=sd)
    dataset._cache_clip_vision_borrowed = lambda: dm.CLIPCachingMixin._cache_clip_vision_borrowed(dataset)
    with pytest.raises(Exception, match='must have an adapter'):
        dm.CLIPCachingMixin.cache_clip_vision_to_disk(dataset)
    assert [c for c in calls if c != 'save'] == ['cache_clip', 'restore']


# ---------------------------------------------------------------- CLIP vision: the real caching loop rebuilds both kinds

class ClipItem:
    def __init__(self, folder, i):
        self.i = i
        self.path = str(folder / f'img{i}.png')
        self.has_clip_augmentations = False
        self._embedding_path = str(folder / '_clip_vision_cache' / f'img{i}.safetensors')

    def get_clip_vision_embeddings_path(self, recalculate=False):
        return self._embedding_path

    def get_clip_vision_info_dict(self):
        return {'item': self.i}

    def load_clip_image(self):
        self.clip_image_tensor = torch.full((3, 4, 4), float(self.i))


def clip_dataset(folder):
    encoded = []

    def encoder(pixels, output_hidden_states=True):
        encoded.append(float(pixels.flatten()[0]))
        value = pixels.flatten()[0]
        return SimpleNamespace(image_embeds=value.repeat(1, 2), hidden_states=[value.repeat(1, 1, 2)] * 2)
    encoder.to = lambda device: encoder
    processor = lambda images, **kw: SimpleNamespace(pixel_values=images)  # noqa: E731
    adapter = SimpleNamespace(clip_image_processor=processor, image_encoder=encoder, input_size=4,
                              config=SimpleNamespace(quad_image=False, image_encoder_path='fixture'))
    sd = SimpleNamespace(adapter=adapter, device='cpu', device_torch=torch.device('cpu'), torch_dtype=torch.float32,
                         save_device_state=lambda: None, set_device_state_preset=lambda name: None,
                         restore_device_state=lambda: None)
    ds = dm.CLIPCachingMixin.__new__(dm.CLIPCachingMixin)
    ds.__dict__.update(is_caching_clip_vision_to_disk=True, dataset_path='fixture', sd=sd,
                       dataset_config=SimpleNamespace(clip_image_path=str(folder)), clip_vision_num_unconditional_cache=1,
                       file_list=[ClipItem(folder, i) for i in (1, 2)])
    return ds, encoded


def test_clip_caching_rebuilds_truncated_unconditional_and_per_image_files(tmp_path):
    from safetensors.torch import load_file
    ds, encoded = clip_dataset(tmp_path)
    ds.cache_clip_vision_to_disk()
    assert encoded == [0.0, 1.0, 2.0]  # the unconditional (zeros) image, then each item
    uncond = Path(ds.clip_vision_unconditional_cache[0])
    victim = Path(ds.file_list[1].get_clip_vision_embeddings_path())
    for path in (uncond, victim):
        data = path.read_bytes()
        path.write_bytes(data[:len(data) // 2])  # interrupted writes of an older version
    again, encoded_again = clip_dataset(tmp_path)
    again.cache_clip_vision_to_disk()
    assert encoded_again == [0.0, 2.0]  # only the two unreadable files are rebuilt
    assert torch.equal(load_file(str(victim))['image_embeds'], torch.full((1, 2), 2.))
    assert torch.equal(load_file(str(uncond))['image_embeds'], torch.zeros(1, 2))
    assert not list(victim.parent.glob('.*.tmp'))
