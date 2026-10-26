"""RAWR pairs publish optimizer.pt as a hard link to the committed slot instead of a full copy (26 Oct 2026).

With 5-9.5 GB optimizers the copy doubled every save's bytes, slow on a USB 2 drive. A link is only safe while
nothing rewrites a linked file in place: atomic_save gives a slot a new file and swaps the name, and the plain
(non-RAWR) optimizer save now does the same. These pin that, plus the copy fallback where links aren't available.
"""
import os
from pathlib import Path

import pytest
import torch

from test_rawr_stochastic_pair import paired_trainer
from test_rawr_trainer_continuity import advance, cpu_only, trainer  # noqa: F401  (cpu_only is autouse)
from training_extras import rawr_pair as pair


def needs_hard_links(tmp_path):
    probe = tmp_path / 'link-probe'
    probe.write_bytes(b'x')
    try:
        os.link(probe, tmp_path / 'link-probe-2')
    except OSError:
        pytest.skip('this volume has no hard links')


def save_steps(f, steps):
    for i in steps:
        advance(f, i)
        f.save(i)
        f.grad_accumulation_step += 1


def failing_publish(*args):
    raise OSError('alias locked')


def test_optimizer_alias_is_the_committed_slot_file(tmp_path):
    needs_hard_links(tmp_path)
    f, _, _ = paired_trainer(tmp_path)
    alias = tmp_path / 'optimizer.pt'
    for step in range(3):  # slots a, b, a: the alias follows the committed one
        save_steps(f, [step])
        assert os.path.samefile(alias, pair.select(str(tmp_path), 'tiny')['optimizer'])
    assert not list(tmp_path.glob('.rawr-export.*'))


def test_slot_rewrites_never_reach_an_older_alias(tmp_path, monkeypatch):
    needs_hard_links(tmp_path)
    f, _, _ = paired_trainer(tmp_path)
    save_steps(f, [0])  # slot a committed, alias linked to it
    alias = tmp_path / 'optimizer.pt'
    before = alias.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(pair, '_publish', failing_publish)
        with pytest.warns(RuntimeWarning, match='pair committed'):
            save_steps(f, [1, 2])  # slots b then a are rewritten while the alias stays on the first a
    assert alias.read_bytes() == before
    assert Path(pair.select(str(tmp_path), 'tiny')['optimizer']).read_bytes() != before


def test_optimizer_alias_falls_back_to_a_copy(tmp_path, monkeypatch):
    def no_links(*args):
        raise OSError('no hard links on this volume')

    monkeypatch.setattr(pair.os, 'link', no_links)
    f, _, _ = paired_trainer(tmp_path)
    save_steps(f, [0])
    alias, slot = tmp_path / 'optimizer.pt', Path(pair.select(str(tmp_path), 'tiny')['optimizer'])
    assert not os.path.samefile(alias, slot) and alias.read_bytes() == slot.read_bytes()
    assert not list(tmp_path.glob('.rawr-export.*'))


def test_plain_optimizer_save_never_writes_through_a_link(tmp_path):
    needs_hard_links(tmp_path)
    f, _, _ = trainer(tmp_path)
    f._rawr_checkpoint_config = None  # a non-RAWR run in a folder that once held RAWR pairs
    committed = tmp_path / 'committed-slot.pt'
    committed.write_bytes(b'committed optimizer bytes')
    os.link(committed, tmp_path / 'optimizer.pt')
    advance(f, 0)
    f.save(0)
    assert committed.read_bytes() == b'committed optimizer bytes'
    assert torch.load(tmp_path / 'optimizer.pt', map_location='cpu', weights_only=True)['param_groups']
