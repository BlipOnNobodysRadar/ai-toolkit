"""Replacing a caption must not lose the old one when the write fails (docs/STATE_MACHINES.md M14, SM-14).

BaseCaptioner.save_caption_for_file deleted the existing caption, then wrote the new one; the caption loop catches
the error and moves on, so a failed write (disk full, a kill, here an unencodable character) left an empty caption
in place of the old, possibly hand-written, one. Real BaseCaptioner method on real files; CPU only.
"""
from types import SimpleNamespace

import pytest

from extensions_built_in.captioner.BaseCaptioner import BaseCaptioner


def captioner():
    c = BaseCaptioner.__new__(BaseCaptioner)
    c.caption_config = SimpleNamespace(caption_extension='txt')
    return c


def test_failed_write_keeps_the_previous_caption(tmp_path):
    image, caption = tmp_path / 'a.png', tmp_path / 'a.txt'
    image.write_bytes(b'image')
    caption.write_text('hand-written caption', encoding='utf-8')
    with pytest.raises(UnicodeEncodeError):
        captioner().save_caption_for_file(str(image), 'model output with a lone surrogate \ud800')
    assert caption.read_text(encoding='utf-8') == 'hand-written caption'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.png', 'a.txt']


def test_caption_is_replaced(tmp_path):
    image, caption = tmp_path / 'a.png', tmp_path / 'a.txt'
    image.write_bytes(b'image')
    caption.write_text('old', encoding='utf-8')
    captioner().save_caption_for_file(str(image), 'new caption')
    assert caption.read_text(encoding='utf-8') == 'new caption'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.png', 'a.txt']
