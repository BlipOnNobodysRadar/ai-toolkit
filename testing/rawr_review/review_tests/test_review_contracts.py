"""Independent contracts. Failing assertions are review findings, not fixes."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from training_extras import rawr_stochastic as rs

ROOT = Path(__file__).resolve().parents[1]

def test_normalized_caption_extension_is_checked(tmp_path):
    from toolkit.config_modules import DatasetConfig
    config = DatasetConfig(folder_path=str(tmp_path), caption_ext='txt')
    assert config.caption_ext == '.txt'
    image, caption = tmp_path / 'image.png', tmp_path / 'image.txt'
    image.write_bytes(b'identity fixture; never decoded')
    caption.write_text('original caption')
    item = SimpleNamespace(path=str(image), dataset_config=config, flip_x=False, flip_y=False)
    dataset = SimpleNamespace(file_list=[item], source_configs=[config], caption_dict=None,
                              epoch_num=1, batch_indices=[[0]])
    saved = rs._dataset_state(dataset)
    caption.write_text('substantially changed caption with different size')
    observed = {
        'normalized_extension': config.caption_ext,
        'actual_caption_path': str(caption),
        'watched_caption_path': str(image.with_suffix('')) + '.' + config.caption_ext,
        'identity_before': saved['identity'],
        'identity_after': rs.dataset_identity(dataset),
    }
    (ROOT / 'evidence/caption-identity.json').write_text(json.dumps(observed, indent=2))
    with pytest.raises(ValueError, match='dataset files, captions, order or configuration changed'):
        rs._restore_dataset(dataset, saved)

def test_supported_resume_cannot_silently_enable_learnable_snr():
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.stable_diffusion_model import StableDiffusion
    f = SDTrainer.__new__(SDTrainer)
    f.sd = StableDiffusion.__new__(StableDiffusion)
    f.sd.is_multistage = False
    f.accelerator = SimpleNamespace(device=torch.device('cpu'), num_processes=1)
    f.train_config = SimpleNamespace(disable_sampling=True, validation_config=None,
                                     do_paramiter_swapping=False)
    f.ema = f.training_extras = f.snr_gos = None
    f.data_loader = f.data_loader_reg = None
    saved = rs.StochasticResume(f).state_dict()
    assert saved['supported'] is True
    f.snr_gos = object()  # production LearnableSNRGamma owns unsaved optimizer+buffer
    with pytest.raises(ValueError, match='Cannot restore RAWR stochastic state'):
        rs.StochasticResume(f, saved)

@pytest.mark.parametrize('variant', ['fp16-export', 'learnable-snr'])
def test_actual_supported_resume_is_parameter_exact(variant):
    path = ROOT / f'evidence/variants/{variant}-comparison.json'
    if not path.exists():
        pytest.skip('run the serial independent actual-trainer variant first')
    result = json.loads(path.read_text())
    assert result['cuda_initialized'] is False
    assert result['checkpoint_stochastic_supported'] is True
    assert all(row['parameter_max_abs_error_after'] == 0.0 for row in result['rows']), result
