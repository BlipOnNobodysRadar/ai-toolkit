"""training_extras state (EDM2 etc.) travels in the committed RAWR pair, step-stamped (deep dive, 1 Nov 2026).

Before: rawr_pair.begin() switched the pair off whenever training_extras was set (the H3 config), so H3 resumed from
the float16 export, root training_extras.pt was restored with no step check, and the resume glob could take a
derived {name}_000000300-comfy export. CPU only, tiny tensors.
"""
import importlib
import json
import os
import time
from types import SimpleNamespace

import pytest
import torch

from test_rawr_stochastic_pair import paired_trainer
from test_rawr_trainer_continuity import advance, cpu_only  # noqa: F401  (cpu_only is autouse)
from test_training_extras import _sched
from training_extras import rawr_pair as pair
from training_extras import runtime as rt

EDM = {'edm2': {'enabled': True, 'channels': 16, 'optimizer': 'adamw'}}


def extras():
    return rt.TrainingExtras(EDM, _sched(with_alphas=False), True, 'epsilon', torch.device('cpu'), 10)


def train_extras(ex):
    loss = torch.tensor([1., 2.], requires_grad=True)
    value, _ = ex.augment(loss, torch.empty(0), torch.empty(0), torch.empty(0), torch.tensor([1., 500.]))
    value.mean().backward()
    ex.step()


def save_steps(f, steps):
    for i in steps:
        advance(f, i)
        if getattr(f, 'training_extras', None) is not None:
            train_extras(f.training_extras)
        f.save(i)
        f.grad_accumulation_step += 1


def test_pair_carries_hashed_step_stamped_extras(tmp_path):
    f, _, _ = paired_trainer(tmp_path)
    f.training_extras = extras()
    save_steps(f, [0, 1])
    manifest = json.loads((tmp_path / pair.MANIFEST).read_text())
    assert 'training_extras_sha256' in manifest
    selected = pair.select(str(tmp_path), 'tiny')
    slot_extras = selected['training_extras']
    assert slot_extras is not None and os.path.dirname(slot_extras) == os.path.dirname(selected['model'])
    # the curve stays a save-folder report; the legacy alias matches the committed slot
    assert (tmp_path / 'edm2-weighting.csv').exists()
    assert not os.path.exists(os.path.join(os.path.dirname(slot_extras), 'edm2-weighting.csv'))
    assert (tmp_path / 'training_extras.pt').read_bytes() == open(slot_extras, 'rb').read()
    restored = extras()
    restored.restore(slot_extras, expected_step=f.meta['training_info']['completed_steps'])
    for a, b in zip(f.training_extras.parameters(), restored.parameters()):
        assert torch.equal(a, b)


def test_tampered_pair_extras_refused(tmp_path):
    f, _, _ = paired_trainer(tmp_path)
    f.training_extras = extras()
    save_steps(f, [0])
    slot_extras = pair.select(str(tmp_path), 'tiny')['training_extras']
    with open(slot_extras, 'ab') as fh:
        fh.write(b'x')
    with pytest.raises(ValueError, match='training_extras state is missing or corrupt'):
        pair.select(str(tmp_path), 'tiny')
    os.unlink(slot_extras)
    with pytest.raises(ValueError, match='training_extras state is missing or corrupt'):
        pair.select(str(tmp_path), 'tiny')


def test_manifest_without_extras_still_selects(tmp_path):
    f, _, _ = paired_trainer(tmp_path)
    save_steps(f, [0])
    assert 'training_extras_sha256' not in json.loads((tmp_path / pair.MANIFEST).read_text())
    assert pair.select(str(tmp_path), 'tiny')['training_extras'] is None


def test_extras_step_stamp_checked(tmp_path):
    ex = extras()
    train_extras(ex)
    stamped = tmp_path / 'stamped.pt'
    ex.checkpoint(stamped, curve_dir=tmp_path, step=3)
    extras().restore(stamped, expected_step=3)
    extras().restore(stamped)  # no expectation (explicit start_step) is accepted
    with pytest.raises(ValueError, match='step 3 does not match resume step 4'):
        extras().restore(stamped, expected_step=4)
    legacy = tmp_path / 'legacy.pt'
    ex.checkpoint(legacy)
    with pytest.warns(RuntimeWarning, match='no step stamp'):
        extras().restore(legacy, expected_step=4)


def latest(tmp_path, name):
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    f = SimpleNamespace(job=SimpleNamespace(name='h3'), save_root=str(tmp_path), network_config=None,
                        train_config=SimpleNamespace(lr_scheduler='cosine', merge_network_on_save=False))
    return base.get_latest_save_path(f, name)


def touch(path, directory=False):
    time.sleep(0.02)  # distinct creation times
    if directory:
        os.mkdir(path)
    else:
        path.write_bytes(b'x')


def test_resume_glob_ignores_derived_exports(tmp_path):
    touch(tmp_path / 'h3_000000300.safetensors')
    touch(tmp_path / 'h3_000000300-comfy.safetensors')
    touch(tmp_path / 'h3-comfy.safetensors')
    touch(tmp_path / 'h3_LoRA_000000400.safetensors')
    touch(tmp_path / 'h3_refiner_000000500.safetensors')
    assert latest(tmp_path, 'h3') == str(tmp_path / 'h3_000000300.safetensors')
    assert latest(tmp_path, 'h3_LoRA') == str(tmp_path / 'h3_LoRA_000000400.safetensors')
    assert latest(tmp_path, 'h3_refiner') == str(tmp_path / 'h3_refiner_000000500.safetensors')
    touch(tmp_path / 'h3.safetensors')  # the final, step-less save
    assert latest(tmp_path, 'h3') == str(tmp_path / 'h3.safetensors')
    touch(tmp_path / 'h3_000000600', directory=True)  # diffusers folder
    assert latest(tmp_path, 'h3') == str(tmp_path / 'h3_000000600')
    touch(tmp_path / 'trig_000000100.pt')  # embedding saved as .pt
    assert latest(tmp_path, 'trig') == str(tmp_path / 'trig_000000100.pt')


def test_resume_glob_is_not_fooled_by_folder_names(tmp_path):
    root = tmp_path / 'runs_LoRA_cn'
    root.mkdir()
    touch(root / 'h3_000000100.safetensors')
    assert latest(root, 'h3') == str(root / 'h3_000000100.safetensors')


class _Stop(Exception):
    pass


def hook_restore(tmp_path, monkeypatch, step, generation=None, paired=None, start_step=None):
    """Runs SDTrainer.hook_before_train_loop up to the extras restore, then stops."""
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    class Probe(SDTrainer):
        @property
        def is_caching_text_embeddings(self):
            raise _Stop

    monkeypatch.setattr(base, 'hook_before_train_loop', lambda self: None)
    t = Probe.__new__(Probe)
    t.train_config = SimpleNamespace(training_extras=EDM, loss_type='mse', t0_loss_target=False, train_turbo=False,
                                     dtype='bf16', steps=10, start_step=start_step)
    t.do_guided_loss = False
    t.sd = SimpleNamespace(noise_scheduler=_sched(with_alphas=False), is_flow_matching=True, prediction_type='epsilon')
    t.device_torch, t.save_root, t.step_num = torch.device('cpu'), str(tmp_path), step
    if generation is not None:
        t._rawr_pair_generation, t._rawr_paired_extras_path = generation, paired
    with pytest.raises(_Stop):
        t.hook_before_train_loop()
    return t.training_extras


def test_trainer_restores_the_paired_extras_with_step_check(tmp_path, monkeypatch):
    f, _, _ = paired_trainer(tmp_path)
    f.training_extras = extras()
    save_steps(f, [0, 1])
    selected = pair.select(str(tmp_path), 'tiny')
    step = f.meta['training_info']['completed_steps']
    # a newer, unrelated root file must not be read when a pair is selected
    (tmp_path / 'training_extras.pt').unlink()
    (tmp_path / 'training_extras.pt').write_bytes(b'not state')
    restored = hook_restore(tmp_path, monkeypatch, step, selected['generation'], selected['training_extras'])
    for a, b in zip(f.training_extras.parameters(), restored.parameters()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match='does not match resume step'):
        hook_restore(tmp_path, monkeypatch, step + 1, selected['generation'], selected['training_extras'])
    hook_restore(tmp_path, monkeypatch, step + 1, selected['generation'], selected['training_extras'], start_step=step + 1)
    with pytest.raises(ValueError, match='has no training_extras state'):
        hook_restore(tmp_path, monkeypatch, step, selected['generation'], None)


def test_trainer_legacy_root_extras_step_checked(tmp_path, monkeypatch):
    ex = extras()
    train_extras(ex)
    ex.checkpoint(tmp_path / 'training_extras.pt', step=5)
    hook_restore(tmp_path, monkeypatch, 5)
    with pytest.raises(ValueError, match='step 5 does not match resume step 7'):
        hook_restore(tmp_path, monkeypatch, 7)
