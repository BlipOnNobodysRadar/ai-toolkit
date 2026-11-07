"""Which checkpoint is "latest" for resume and pruning (docs/STATE_MACHINES.md M17, OPUS-11/P-07; Codex
windows-resume-probe). The real BaseSDTrainProcess.get_latest_save_path and clean_up_saves, run unbound on a minimal
trainer namespace, over files written by the real writers (torch.save, safetensors.save_file, a Diffusers-style
folder whose aitk_meta.yaml is written last). Write times are set with os.utime instead of sleeping. On NTFS a file
rewritten in place keeps its creation time, which is what the original getctime ranking used.
"""
import os
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import save_file

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def trainer(root, keep=2):
    return SimpleNamespace(job=SimpleNamespace(name='run'), save_root=str(root), network_config=None,
                           train_config=SimpleNamespace(lr_scheduler='constant'), embed_config=None,
                           accelerator=SimpleNamespace(is_main_process=True),
                           save_config=SimpleNamespace(max_step_saves_to_keep=keep), sd=SimpleNamespace())


def write(path, kind, value):
    if kind == 'pt':
        torch.save({'weight': torch.tensor([value])}, path)
    elif kind == 'safetensors':
        save_file({'weight': torch.tensor([float(value)])}, str(path))
    else:  # Diffusers folder: weights first, aitk_meta.yaml last (BaseModel.save_model order)
        os.makedirs(path, exist_ok=True)
        save_file({'weight': torch.tensor([float(value)])}, os.path.join(path, 'model.safetensors'))
        with open(os.path.join(path, 'aitk_meta.yaml'), 'w') as f:
            yaml.dump({'training_info': {'step': value}}, f)


def stamp(path, when):
    target = os.path.join(path, 'aitk_meta.yaml') if os.path.isdir(path) else path
    os.utime(target, (when, when))


SUFFIX = {'pt': '.pt', 'safetensors': '.safetensors', 'diffusers': ''}


@pytest.mark.parametrize('kind', ['pt', 'safetensors', 'diffusers'])
def test_a_final_save_rewritten_after_a_periodic_one_is_the_resume_point(tmp_path, kind):
    final, periodic = tmp_path / f'run{SUFFIX[kind]}', tmp_path / f'run_000000100{SUFFIX[kind]}'
    write(final, kind, 0)        # created first (an earlier final save)
    write(periodic, kind, 100)
    write(final, kind, 200)      # rewritten in place: NTFS keeps its creation time
    stamp(final, 1_000_300)
    stamp(periodic, 1_000_200)
    if kind == 'diffusers':
        os.utime(final, (1_000_000, 1_000_000))  # the folder's own time is not what a rewrite updates
    assert BaseSDTrainProcess.get_latest_save_path(trainer(tmp_path)) == str(final)


@pytest.mark.parametrize('kind', ['pt', 'safetensors', 'diffusers'])
def test_equal_write_times_rank_by_step_then_the_final_save(tmp_path, kind):
    names = [f'run_000000100{SUFFIX[kind]}', f'run_000000200{SUFFIX[kind]}']
    for i, name in enumerate(names):
        write(tmp_path / name, kind, i)
        stamp(tmp_path / name, 1_000_000)
    assert BaseSDTrainProcess.get_latest_save_path(trainer(tmp_path)) == str(tmp_path / names[1])
    final = tmp_path / f'run{SUFFIX[kind]}'
    write(final, kind, 9)
    stamp(final, 1_000_000)
    assert BaseSDTrainProcess.get_latest_save_path(trainer(tmp_path)) == str(final)


def test_a_newer_lower_step_save_after_a_step_reset_is_still_the_latest(tmp_path):
    # a deliberate restart from step 0 in the same folder: write time, not step, decides
    write(tmp_path / 'run_000000300.pt', 'pt', 300)
    stamp(tmp_path / 'run_000000300.pt', 1_000_000)
    write(tmp_path / 'run_000000050.pt', 'pt', 50)
    stamp(tmp_path / 'run_000000050.pt', 1_000_100)
    assert BaseSDTrainProcess.get_latest_save_path(trainer(tmp_path)).endswith('run_000000050.pt')


@pytest.mark.parametrize('kind', ['pt', 'safetensors', 'diffusers'])
def test_pruning_keeps_the_most_recently_written_saves(tmp_path, kind):
    # an old run left 200 and 300; a restarted run rewrote 100 in place (oldest creation time, newest write)
    paths = {step: tmp_path / f'run_{step:09d}{SUFFIX[kind]}' for step in (100, 200, 300)}
    write(paths[100], kind, 100)
    write(paths[200], kind, 200)
    write(paths[300], kind, 300)
    write(paths[100], kind, 101)
    for step, when in ((200, 1_000_200), (300, 1_000_300), (100, 1_000_400)):
        stamp(paths[step], when)
    latest = BaseSDTrainProcess.clean_up_saves(trainer(tmp_path, keep=2))
    assert not paths[200].exists() and paths[100].exists() and paths[300].exists()
    assert latest == str(paths[100])
