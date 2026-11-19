"""RAWR resume guards (docs/STATE_MACHINES.md, SM-04).

OPUS-02: a changed LoRA shape (network.did_change_weights) skipped loading the paired optimizer, and RAWR then
restarted its cycle and moments at step > 0 without a word. OPUS-12: a non-RAWR run in a folder holding a committed
pair left the manifest as the resume point, so switching back to RAWR rewound past it. Real BaseSDTrainProcess
methods and real rawr_pair; CPU only.
"""
import json
from types import SimpleNamespace

import pytest
import torch

from test_rawr_trainer_continuity import advance, cpu_only, trainer  # noqa: F401  (cpu_only is autouse)
from training_extras import rawr_checkpoint as rc
from training_extras import rawr_pair


def loader(tmp_path, did_change_weights, step_num=4):
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    p = torch.nn.Parameter(torch.ones(2))
    saved_opt = torch.optim.AdamW([p], lr=.01)
    p.grad = torch.ones(2)
    saved_opt.step()
    path = tmp_path / 'slot-optimizer.pt'
    torch.save(saved_opt.state_dict(), path)
    q = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([q], lr=.01)
    t = SDTrainer.__new__(SDTrainer)
    t.save_root, t.step_num = str(tmp_path), step_num
    t.network = SimpleNamespace(did_change_weights=did_change_weights)
    t._rawr_paired_optimizer_path = str(path)
    return t, optimizer


def test_rawr_refuses_a_resume_whose_paired_optimizer_was_not_loaded(tmp_path):
    t, optimizer = loader(tmp_path, did_change_weights=True)
    with pytest.raises(RuntimeError, match='paired with the selected model, but it was not loaded'):
        t._load_optimizer_state(optimizer, True)
    assert optimizer.state_dict()['state'] == {}


def test_rawr_resume_still_loads_the_pair(tmp_path):
    t, optimizer = loader(tmp_path, did_change_weights=False)
    captured = t._load_optimizer_state(optimizer, True)
    assert captured is not None and optimizer.state_dict()['state']


def test_other_schedulers_keep_the_fresh_optimizer_after_a_shape_change(tmp_path):
    t, optimizer = loader(tmp_path, did_change_weights=True)
    (tmp_path / 'optimizer.pt').write_bytes((tmp_path / 'slot-optimizer.pt').read_bytes())
    assert t._load_optimizer_state(optimizer, False) is None
    assert optimizer.state_dict()['state'] == {}


def committed_manifest(root):
    manifest = {'version': 1, 'slot': 'a', 'generation': 'f' * 32, 'name': 'tiny', 'step': 3,
                'model_sha256': '0' * 64, 'optimizer_sha256': '0' * 64}
    (root / rawr_pair.MANIFEST).write_text(json.dumps(manifest), encoding='utf-8')
    return (root / rawr_pair.MANIFEST).read_bytes()


def test_non_rawr_save_refuses_a_folder_with_a_committed_pair(tmp_path):
    f, _, _ = trainer(tmp_path)
    f._rawr_checkpoint_config = None  # the scheduler was switched away from RAWR in the same output folder
    before = committed_manifest(tmp_path)
    advance(f, 0)
    with pytest.raises(ValueError, match='committed RAWR resume pair'):
        f.save(0)
    assert (tmp_path / rawr_pair.MANIFEST).read_bytes() == before  # the user's pair is left alone
    assert not (tmp_path / 'optimizer.pt').exists() and not list(tmp_path.glob('tiny*.safetensors'))


def test_startup_check_refuses_before_anything_is_written(tmp_path):
    before = committed_manifest(tmp_path)
    with pytest.raises(ValueError, match='Use a new output folder, or keep the RAWR scheduler'):
        rawr_pair.refuse_stale_manifest(str(tmp_path))
    assert (tmp_path / rawr_pair.MANIFEST).read_bytes() == before
    rawr_pair.refuse_stale_manifest(str(tmp_path / 'fresh'))  # no manifest: nothing to refuse


def test_rawr_saves_in_a_committed_folder_still_work(tmp_path):
    f, _, _ = trainer(tmp_path)
    committed_manifest(tmp_path)
    f.network = object()  # a pair-capable (LoRA) RAWR run continues the committed folder
    pair = rawr_pair.begin(f)
    assert pair['slot'] == 'b'
