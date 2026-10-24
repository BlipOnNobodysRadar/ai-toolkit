import copy
import importlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

from training_extras import rawr_checkpoint as rc, rawr_pair as pair, rawr_stochastic as rs
from test_rawr_trainer_continuity import cpu_only, equal, trainer, advance


@pytest.mark.parametrize('stop', [1, 2, 4, 5])
def test_sampler_and_all_cpu_rng_streams_continue_at_epoch_boundaries(stop):
    dataset = TensorDataset(torch.arange(4))
    sampler = rs.RecordedSampler(BatchSampler(RandomSampler(dataset), 1, False))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    iterator = iter(loader)
    for _ in range(stop):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)
        torch.rand(3)
        random.random()
        np.random.random()
    saved_sampler, saved_rng = sampler.state_dict(), rs.rng_state()
    def future(loader, iterator):
        result = []
        for _ in range(7):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            result.append((int(batch[0]), torch.rand(3), random.random(), float(np.random.random())))
        return result
    expected = future(loader, iterator)
    restored_sampler = rs.RecordedSampler(BatchSampler(RandomSampler(dataset), 1, False))
    restored_sampler.load_state_dict(saved_sampler)
    restored_loader = DataLoader(dataset, batch_sampler=restored_sampler, num_workers=0)
    restored_iterator = iter(restored_loader)
    rs.restore_rng(saved_rng)
    assert equal(expected, future(restored_loader, restored_iterator))


def paired_trainer(tmp_path):
    f, p, scheduler_params = trainer(tmp_path)
    f.train_config.lr_scheduler = 'easy:RAWR'
    f.train_config.start_step = None
    f.network_config = SimpleNamespace(pretrained_lora_path=None)
    def save_weights(path, dtype, metadata, extra_state_dict):
        save_file({'weight': p.detach().clone().to(dtype)}, path, metadata=metadata)
    f.network = SimpleNamespace(multiplier=1., save_weights=save_weights)
    return f, p, scheduler_params


@pytest.mark.parametrize('failure', ['model', 'optimizer', 'manifest'])
def test_actual_save_selects_previous_committed_pair_after_interruption(tmp_path, monkeypatch, failure):
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    f, p, scheduler_params = paired_trainer(tmp_path)
    advance(f, 0)
    f.save(0)
    first = pair.select(str(tmp_path), 'tiny')
    saved_p = load_file(first['model'])['weight'].to(p.dtype)
    saved_optimizer = torch.load(first['optimizer'], weights_only=True)
    f.grad_accumulation_step += 1
    advance(f, 1)
    with monkeypatch.context() as patch:
        def fail_model(path, **kwargs):
            from pathlib import Path
            Path(path).write_bytes(b'partial model')
            raise OSError('model interrupted')
        def fail_optimizer(state, path):
            from pathlib import Path
            Path(path).write_bytes(b'partial optimizer')
            raise OSError('optimizer interrupted')
        def fail_manifest(*args):
            raise OSError('manifest interrupted')
        if failure == 'model':
            patch.setattr(f.network, 'save_weights', fail_model)
        elif failure == 'optimizer':
            patch.setattr(rc, 'atomic_save', fail_optimizer)
        else:
            patch.setattr(pair, '_write_manifest', fail_manifest)
        with pytest.raises((OSError, RuntimeError)):
            f.save(1)
    selected_path = base.get_latest_save_path(f)
    assert selected_path == first['model']
    assert f._rawr_paired_optimizer_path == first['optimizer']
    assert pair.select(str(tmp_path), 'tiny') == first
    base.load_training_state_from_metadata(f, selected_path)
    assert f.step_num == 1
    assert torch.equal(load_file(selected_path)['weight'].to(p.dtype), saved_p)
    assert equal(torch.load(f._rawr_paired_optimizer_path, weights_only=True), saved_optimizer)
    # Recover pending gradients/scheduler and reproduce the update from the
    # selected committed pair, not from the interrupted inactive slot.
    b, pb, _ = trainer(tmp_path / 'recover')
    pb.data.copy_(saved_p)
    b.optimizer.load_state_dict(saved_optimizer)
    captured = rc.capture(saved_optimizer)
    from training_extras.optim import create_scheduler
    for g in b.optimizer.param_groups:
        g['lr'] = g['initial_lr'] = .01
    b.lr_scheduler = create_scheduler(b.optimizer, 'easy:RAWR', 12, scheduler_params)
    rc.restore(b.lr_scheduler, captured, b._rawr_checkpoint_config, 1)
    b.grad_accumulation_step = rc.restore_training_state(b.optimizer, captured, b.train_config, 1)
    advance(b, 1)
    assert torch.equal(p, pb)
    assert equal(f.optimizer.state_dict(), b.optimizer.state_dict())


def test_manifest_failure_preserves_bytes_and_two_slots_are_reused(tmp_path, monkeypatch):
    f, _, _ = paired_trainer(tmp_path)
    for i in range(4):
        advance(f, i)
        f.save(i)
        f.grad_accumulation_step += 1
    assert sorted(x.name for x in (tmp_path / pair.DIRECTORY).iterdir()) == ['a', 'b']
    manifest_path = tmp_path / pair.MANIFEST
    before = manifest_path.read_bytes()
    original_replace = pair.os.replace
    def fail_manifest_replace(source, target):
        if str(target) == str(manifest_path):
            raise OSError('manifest rename interrupted')
        return original_replace(source, target)
    monkeypatch.setattr(pair.os, 'replace', fail_manifest_replace)
    with pytest.raises(RuntimeError, match='Could not save RAWR'):
        f.save(4)
    assert manifest_path.read_bytes() == before
    assert pair.select(str(tmp_path), 'tiny')['step'] == 4
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('file', ['model', 'optimizer'])
def test_corrupt_or_swapped_committed_file_is_rejected(tmp_path, file):
    f, _, _ = paired_trainer(tmp_path)
    advance(f, 0)
    f.save(0)
    selected = pair.select(str(tmp_path), 'tiny')
    from pathlib import Path
    Path(selected[file]).write_bytes(b'same label, different content')
    with pytest.raises(ValueError, match='missing or corrupt'):
        pair.select(str(tmp_path), 'tiny')


def test_alias_publish_failure_is_committed_and_unpaired_exports_are_rejected(tmp_path, monkeypatch):
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    f, _, _ = paired_trainer(tmp_path)
    advance(f, 0)
    with monkeypatch.context() as patch:
        patch.setattr(pair, '_publish', lambda *a: (_ for _ in ()).throw(OSError('alias locked')))
        with pytest.warns(RuntimeWarning, match='pair committed'):
            f.save(0)
    selected = base.get_latest_save_path(f)
    assert selected.endswith('model.safetensors')
    assert pair.select(str(tmp_path), 'tiny')['step'] == 1
    del f._rawr_pair_generation
    with pytest.raises(ValueError, match='committed-pair manifest'):
        base.load_training_state_from_metadata(f, selected)


def test_legacy_selection_without_manifest_is_unchanged(tmp_path):
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    f, p, _ = paired_trainer(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / 'tiny_000000001.safetensors'
    save_file({'weight': p.detach().clone()}, str(path), metadata={'training_info': json.dumps({'step': 2})})
    assert base.get_latest_save_path(f) == str(path)
    base.load_training_state_from_metadata(f, str(path))
    assert f.step_num == 2
    assert not hasattr(f, '_rawr_paired_optimizer_path')


def test_unsupported_save_mode_cannot_leave_a_stale_commit_pointer(tmp_path):
    f, _, _ = paired_trainer(tmp_path)
    advance(f, 0)
    f.save(0)
    original = pair.select(str(tmp_path), 'tiny')
    f.ema = object()
    with pytest.raises(ValueError, match='unsupported save mode'):
        f.save(1)
    assert pair.select(str(tmp_path), 'tiny') == original


@pytest.mark.parametrize('mode', ['workers', 'generator'])
def test_unavailable_loader_state_is_explicit_and_cannot_downgrade_supported_resume(mode):
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.stable_diffusion_model import StableDiffusion
    f = SDTrainer.__new__(SDTrainer)
    f.sd = StableDiffusion.__new__(StableDiffusion)
    f.sd.is_multistage = False
    f.accelerator = SimpleNamespace(device=torch.device('cpu'), num_processes=1)
    f.train_config = SimpleNamespace(disable_sampling=True, validation_config=None, do_paramiter_swapping=False)
    f.ema = None
    f.data_loader_reg = None
    # Constructing a DataLoader does not start workers; do not iterate it.
    f.data_loader = DataLoader(TensorDataset(torch.arange(4)), shuffle=True,
                              num_workers=1 if mode == 'workers' else 0,
                              generator=torch.Generator() if mode == 'generator' else None)
    with pytest.warns(RuntimeWarning, match='stochastic continuation unavailable'):
        state = rs.StochasticResume(f).state_dict()
    assert state['supported'] is False and state['reason']
    with pytest.raises(ValueError, match='Cannot restore RAWR stochastic state'):
        rs.StochasticResume(f, {'supported': True, 'version': 1})


@pytest.mark.parametrize('changed_caption', [False, True])
def test_dataset_layout_restore_checks_input_identity(tmp_path, changed_caption):
    image = tmp_path / 'image.png'
    caption = tmp_path / 'image.txt'
    image.write_bytes(b'fixture image identity')
    caption.write_text('original caption')
    config = SimpleNamespace(caption_ext='txt')
    item = SimpleNamespace(path=str(image), dataset_config=config, flip_x=False, flip_y=False, crop_x=3)
    dataset = SimpleNamespace(file_list=[item], source_configs=[config], caption_dict=None,
                              epoch_num=4, batch_indices=[[0]])
    saved = rs._dataset_state(dataset)
    item.crop_x = 99
    dataset.epoch_num = 1
    if changed_caption:
        caption.write_text('changed caption content')
        with pytest.raises(ValueError, match='dataset files, captions, order or configuration changed'):
            rs._restore_dataset(dataset, saved)
        assert item.crop_x == 99
    else:
        rs._restore_dataset(dataset, saved)
        assert item.crop_x == 3 and dataset.epoch_num == 4


def test_version_three_cannot_silently_omit_stochastic_state(tmp_path):
    f, _, _ = trainer(tmp_path)
    advance(f, 0)
    state = rc.add_state(f.optimizer.state_dict(), f.lr_scheduler, f._rawr_checkpoint_config,
                         1, training_state=rc.training_state(f, 1))
    state[rc.KEY]['version'] = 3
    with pytest.raises(ValueError, match='Missing RAWR stochastic checkpoint state'):
        rc.restore(f.lr_scheduler, rc.capture(state), f._rawr_checkpoint_config, 1)
