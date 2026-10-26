"""Where per-step reseeding (train.reseed_per_step) meets the RAWR recorded-state path (26 Oct 2026 reconcile).

Exactly one mechanism owns the RNG and data position. These pin the seams the reconcile review found: switching
modes between save and resume is refused in both directions, an exact reseeded resume prints no "approximate"
warning, a custom SNR class keeps its old fallback while the standard one is captured outside the recorded path,
and the dataset identity follows the caption source the loader actually reads. In-memory; no fixtures, no GPU.
"""
import warnings
from types import SimpleNamespace

import pytest
import torch

from toolkit.config_modules import DatasetConfig
from toolkit.train_tools import LearnableSNRGamma
from training_extras import rawr_checkpoint as rc
from training_extras import rawr_stochastic as rs


def reseeded(**extra):
    return SimpleNamespace(training_seed=1234, _rng_seed=lambda *parts: 7, step_num=3, **extra)


def not_reseeded(**extra):
    # reseeding off and outside the recorded path's support (GPU), as in an ordinary run with no training_seed
    return SimpleNamespace(training_seed=None, accelerator=SimpleNamespace(device='cuda:0', num_processes=1),
                           step_num=3, **extra)


@pytest.mark.parametrize('with_snr', [False, True])
def test_reseeded_checkpoint_refuses_resume_with_reseeding_off(with_snr):
    snr = LearnableSNRGamma(None, device='cpu') if with_snr else None
    with pytest.raises(ValueError, match='saved with per-step reseeding'):
        rs.StochasticResume(not_reseeded(snr_gos=snr), {'supported': False, 'reason': rs.RESEEDED})


def test_recorded_checkpoint_refuses_resume_with_reseeding_on():
    with pytest.raises(ValueError, match='reseed_per_step: false'):
        rs.StochasticResume(reseeded(), {'supported': True, 'version': 2})


@pytest.mark.parametrize('saved', [{'supported': False, 'reason': rs.RESEEDED}, None], ids=['reseeded', 'pre-v3'])
def test_reseeded_resume_is_not_reported_approximate(saved):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        resume = rs.StochasticResume(reseeded(), saved)
        resume.restore_rng(saved)
    assert resume.state_dict() == {'supported': False, 'reason': rs.RESEEDED}


def test_unsupported_mode_still_warns():
    trainer = not_reseeded()
    with pytest.warns(RuntimeWarning, match='continuation unavailable'):
        resume = rs.StochasticResume(trainer, None)
    with pytest.warns(RuntimeWarning, match='approximate'):
        resume.restore_rng(None)


def trainer_for_save(snr):
    p = torch.nn.Parameter(torch.zeros(2))
    return SimpleNamespace(train_config=SimpleNamespace(gradient_accumulation_steps=1, gradient_accumulation=1),
                           grad_accumulation_step=1, step_num=4, optimizer=torch.optim.SGD([p], lr=0.1),
                           snr_gos=snr, _rawr_stochastic=None)


class ScalarOnlySNR:
    """A custom SNR class with only the four scalars the legacy JSON export reads."""
    offset_1 = offset_2 = gamma = scale = torch.tensor(1.0)


def test_custom_snr_keeps_its_fallback():
    trainer = trainer_for_save(ScalarOnlySNR())
    state = rc.training_state(trainer, 5)
    assert 'snr' not in state
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rc.restore_auxiliary_state(trainer, {'payload': {'training_state': state}})


def test_standard_snr_travels_outside_the_recorded_path():
    snr = LearnableSNRGamma(None, device='cpu')
    snr.buffer = [torch.tensor(0.5), torch.tensor(0.25)]
    state = rc.training_state(trainer_for_save(snr), 5)
    assert state['snr']['buffer'] and state['dataset_identity'] == [None, None]
    fresh = trainer_for_save(LearnableSNRGamma(None, device='cpu'))
    rc.restore_auxiliary_state(fresh, {'payload': {'training_state': state}})
    assert [v.item() for v in fresh.snr_gos.buffer] == [0.5, 0.25]


def test_identity_follows_the_caption_source_actually_read(tmp_path):
    image, sidecar = tmp_path / 'image.png', tmp_path / 'image.txt'
    image.write_bytes(b'fixture image')
    sidecar.write_text('unused sidecar')
    config = DatasetConfig(folder_path=str(tmp_path), caption_ext='txt')
    item = SimpleNamespace(path=str(image), dataset_config=config, flip_x=False, flip_y=False)
    dataset = SimpleNamespace(file_list=[item], source_configs=[config],
                              caption_dict={str(image): {'caption': 'from the json'}})
    before = rs.dataset_identity(dataset)
    sidecar.write_text('edited, but the json caption is what the loader reads')
    assert rs.dataset_identity(dataset) == before
    dataset.caption_dict = {str(image): {'caption': 'edited json caption'}}
    assert rs.dataset_identity(dataset) != before
    dataset.caption_dict = None  # now the sidecar is the source, so its edit counts
    without = rs.dataset_identity(dataset)
    sidecar.write_text('edited again')
    assert rs.dataset_identity(dataset) != without


@pytest.mark.parametrize('rawr', [True, False])
def test_ui_save_keeps_unfinished_microbatches_for_rawr(rawr):
    # A UI "save now" mid-window (legacy gradient_accumulation_steps) must checkpoint the pending gradients, not
    # zero them first; other schedulers keep upstream's clearing.
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.ones(2)
    seen = []
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer, t.progress_bar, t.step_num = True, None, 5
    t.optimizer = torch.optim.SGD([p], lr=0.1)
    t._rawr_checkpoint_config = {} if rawr else None
    t.should_save = lambda: True
    t.update_db_key = lambda *args: None
    t.ensure_params_requires_grad = lambda *args, **kwargs: None
    t.save = lambda step: seen.append(None if p.grad is None else p.grad.clone())
    t.maybe_save()
    if rawr:
        assert seen[0] is not None and torch.equal(seen[0], torch.ones(2)) and torch.equal(p.grad, torch.ones(2))
    else:
        assert seen[0] is None or not seen[0].any()
