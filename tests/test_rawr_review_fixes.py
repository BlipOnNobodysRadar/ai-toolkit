"""Regression coverage for the independent resume-correctness review."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from training_extras import rawr_checkpoint as rc, rawr_pair as pair, rawr_stochastic as rs
from test_rawr_trainer_continuity import cpu_only, equal, advance
from test_rawr_stochastic_pair import paired_trainer


@pytest.mark.parametrize('variant', ['fp16-export', 'learnable-snr'])
def test_actual_review_variant_restores_complete_observed_state(variant):
    path = Path(__file__).resolve().parents[1] / 'testing/rawr_review/evidence/variants' / f'{variant}-comparison.json'
    if not path.exists():
        pytest.skip('run the local actual-trainer review variants first')
    result = json.loads(path.read_text())
    assert result['checkpoint_stochastic_supported'] and result['committed_pair']
    assert result['cuda_initialized'] is False
    assert [row['step'] for row in result['rows']] == list(range(3, 8))
    for row in result['rows']:
        assert row['parameter_max_abs_error_before'] == row['parameter_max_abs_error_after'] == 0.0
        assert all(row[key] for key in ('optimizer_exact', 'scheduler_exact', 'torch_rng_exact',
                   'pending_gradient_norm_exact', 'snr_state_exact_before_and_after',
                   'all_cpu_rng_exact_before_and_after', 'gradient_tensors_exact_before_and_after'))


@pytest.mark.parametrize('extension', ['txt', '.caption'])
def test_caption_identity_watches_the_file_read_by_actual_loader(tmp_path, extension):
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    config = DatasetConfig(folder_path=str(tmp_path), caption_ext=extension)
    image = tmp_path / 'image.png'
    image.write_bytes(b'not decoded by caption loading')
    caption = tmp_path / ('image' + config.caption_ext)
    decoy = tmp_path / ('image.' + config.caption_ext)
    caption.write_text('the actual caption')
    decoy.write_text('wrong caption path')
    item = FileItemDTO.__new__(FileItemDTO)
    item.path, item.dataset_config = str(image), config
    item.raw_caption = item.raw_caption_short = item.trigger_word = None
    item.flip_x = item.flip_y = item.is_reg = False
    item.load_caption(None)
    assert item.raw_caption == 'the actual caption'
    dataset = SimpleNamespace(file_list=[item], source_configs=[config], caption_dict=None,
                              epoch_num=1, batch_indices=[[0]])
    state = rs._dataset_state(dataset)
    decoy.write_text('irrelevant changed decoy')
    assert rs.dataset_identity(dataset) == state['identity']
    caption.write_text('changed real training caption')
    item.raw_caption = None
    item.load_caption(None)
    assert item.raw_caption == 'changed real training caption'
    with pytest.raises(ValueError, match='dataset files, captions, order or configuration changed'):
        rs._restore_dataset(dataset, state)


def snr_trainer(enabled=True):
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.train_tools import LearnableSNRGamma
    f = SDTrainer.__new__(SDTrainer)
    f.sd = StableDiffusion.__new__(StableDiffusion)
    f.sd.is_multistage = False
    f.accelerator = SimpleNamespace(device=torch.device('cpu'), num_processes=1)
    f.train_config = SimpleNamespace(disable_sampling=True, validation_config=None,
                                     do_paramiter_swapping=False)
    f.ema = f.training_extras = None
    f.data_loader = f.data_loader_reg = None
    f.step_num = 0
    scheduler = SimpleNamespace(alphas_cumprod=torch.linspace(.99, .01, 10))
    f.snr_gos = LearnableSNRGamma(scheduler, device='cpu') if enabled else None
    return f


def snr_update(f, step):
    return f.snr_gos.forward(torch.tensor([.11 + step * .01, .38]), torch.tensor([2, 5]))


def test_snr_optimizer_and_full_rolling_history_continue_after_restore(tmp_path):
    f = snr_trainer()
    for step in range(14):
        snr_update(f, step)
    state = rs.StochasticResume(f).state_dict()
    rc.atomic_save(state, str(tmp_path / 'state.pt'))
    restored = snr_trainer()
    rs.StochasticResume(restored, torch.load(tmp_path / 'state.pt', weights_only=True))
    assert equal(rs.snr_state(f), rs.snr_state(restored))
    assert len(restored.snr_gos.buffer) == restored.snr_gos.max_buffer_size
    for step in range(14, 18):
        assert equal(snr_update(f, step), snr_update(restored, step))
        assert equal(rs.snr_state(f), rs.snr_state(restored))


@pytest.mark.parametrize('before,after', [(False, True), (True, False)])
def test_actual_learnable_snr_cannot_be_added_or_removed_on_resume(before, after):
    state = rs.StochasticResume(snr_trainer(before)).state_dict()
    with pytest.raises(ValueError, match='learnable SNR configuration changed'):
        rs.StochasticResume(snr_trainer(after), state)


@pytest.mark.parametrize('legacy', [None, 'unsupported', 'v1', 'missing'])
def test_incomplete_snr_resume_is_rejected(legacy):
    f = snr_trainer()
    state = rs.StochasticResume(f).state_dict()
    if legacy is None:
        state = None
    elif legacy == 'unsupported':
        state = {'supported': False, 'reason': 'legacy'}
    elif legacy == 'v1':
        state['version'] = 1
        del state['snr']
    else:
        del state['snr']
    f.step_num = 3
    with pytest.raises(ValueError, match='(?i)SNR'):
        rs.StochasticResume(f, state)


def test_version_one_without_snr_still_restores():
    f = snr_trainer(False)
    state = rs.StochasticResume(f).state_dict()
    state['version'] = 1
    del state['snr']
    rs.StochasticResume(f, state)


def test_native_resume_precision_is_independent_of_public_export(tmp_path):
    f, p, _ = paired_trainer(tmp_path)
    f.save_config.dtype = 'float16'
    p.data.copy_(torch.tensor([.123456789123, -.987654321987], dtype=p.dtype))
    advance(f, 0)
    f.save(0)
    selected = pair.select(str(tmp_path), 'tiny')
    private = load_file(selected['model'])['weight']
    public = load_file(str(tmp_path / 'tiny_000000000.safetensors'))['weight']
    assert private.dtype == p.dtype and torch.equal(private, p)
    assert public.dtype == torch.float16 and torch.equal(public, p.to(torch.float16))
    assert not torch.equal(public.to(p.dtype), private)


def test_failed_public_conversion_preserves_alias_and_committed_native_pair(tmp_path, monkeypatch):
    f, p, _ = paired_trainer(tmp_path)
    f.save_config.dtype = 'float16'
    advance(f, 0)
    f.save(0)
    alias = tmp_path / 'tiny_000000000.safetensors'
    previous = alias.read_bytes()
    save = f.network.save_weights
    def partial_export(path, dtype, **kwargs):
        if dtype is not None:
            from pathlib import Path
            Path(path).write_bytes(b'interrupted export')
            raise RuntimeError('export conversion interrupted')
        save(path, dtype=dtype, **kwargs)
    monkeypatch.setattr(f.network, 'save_weights', partial_export)
    p.data.add_(.0123456789)
    with pytest.warns(RuntimeWarning, match='pair committed'):
        f.save(0)
    assert alias.read_bytes() == previous
    assert torch.equal(load_file(pair.select(str(tmp_path), 'tiny')['model'])['weight'], p)
    assert not list(tmp_path.glob('.rawr-export.*'))


def test_snr_is_committed_with_optimizer_and_restored_after_failed_manifest(tmp_path, monkeypatch):
    f, _, _ = paired_trainer(tmp_path)
    aux = snr_trainer()
    f.snr_gos = aux.snr_gos
    f._rawr_stochastic = rs.StochasticResume(aux)
    snr_update(aux, 0)
    advance(f, 0)
    f.save(0)
    selected = pair.select(str(tmp_path), 'tiny')
    saved = copy.deepcopy(rs.snr_state(aux))
    expected = snr_update(aux, 1)
    monkeypatch.setattr(pair, '_write_manifest', lambda *a: (_ for _ in ()).throw(OSError('interrupted')))
    with pytest.raises(RuntimeError):
        f.save(1)
    assert pair.select(str(tmp_path), 'tiny') == selected
    # The legacy scalar JSON has newer values after the interrupted save.
    legacy = json.loads((tmp_path / 'learnable_snr.json').read_text())
    assert legacy['gamma'] != saved['parameters']['gamma'].item()
    checkpoint = torch.load(selected['optimizer'], weights_only=True)
    restored = snr_trainer()
    rs.StochasticResume(restored, checkpoint[rc.KEY]['training_state']['stochastic'])
    assert equal(rs.snr_state(restored), saved)
    assert equal(snr_update(restored, 1), expected)
    assert equal(rs.snr_state(restored), rs.snr_state(aux))
