"""Resume discovery and pruning own exactly this run's saves (docs/STATE_MACHINES.md, OPUS-RES-01, OPUS-PRUNE-01,
OPUS-SAVE-01, OPUS-RAWR-01). The real BaseSDTrainProcess methods run unbound on a minimal trainer namespace over real
temporary files; the Wan 2.2 14B files come from its real save_lora and are read back by its real load_lora. The
bracket and alias layouts are the audit's G and H cases (opus-probes/probe_resume_discovery.py). CPU only."""
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess, find_save_sets
from training_extras import rawr_pair

base = BaseSDTrainProcess


def trainer(root, name='run', keep=2, network=False, **extra):
    t = SimpleNamespace(job=SimpleNamespace(name=name), save_root=str(root), network_config=None, network=None,
                        train_config=SimpleNamespace(lr_scheduler='constant', merge_network_on_save=False),
                        embed_config=None, accelerator=SimpleNamespace(is_main_process=True),
                        save_config=SimpleNamespace(max_step_saves_to_keep=keep), sd=SimpleNamespace())
    if network:
        t.network_config = SimpleNamespace(pretrained_lora_path=None)
        t.network = object()
        t.is_fine_tuning = False
    for key, value in extra.items():
        setattr(t, key, value)
    return t


def write(root, names, start=1_000_000):
    """Real safetensors files, written oldest first (one second apart)."""
    os.makedirs(root, exist_ok=True)
    for i, name in enumerate(names):
        path = os.path.join(root, name)
        save_file({'w': torch.tensor([float(i)])}, path)
        os.utime(path, (start + i, start + i))


def listing(root):
    return sorted(os.listdir(root))


@pytest.mark.parametrize('folder,name', [('out [v2]', 'run'), ('out', 'run[v2]'),       # balanced: G1, G4
                                         ('out [v2', 'run'), ('out v2]', 'run'),         # controls: G2, G3
                                         ('out', 'run[v2'), ('out', 'runv2]')])           # controls: G5, G6
def test_brackets_in_the_folder_or_name_are_literal(tmp_path, folder, name):
    root = tmp_path / folder / name
    write(root, [f'{name}_000000100.safetensors', f'{name}_000000200.safetensors', f'{name}_000000300.safetensors'])
    assert base.get_latest_save_path(trainer(root, name)) == str(root / f'{name}_000000300.safetensors')
    base.clean_up_saves(trainer(root, name, keep=1))
    assert listing(root) == [f'{name}_000000300.safetensors']


def test_a_bracketed_folder_neither_resumes_nor_prunes_a_sibling_folder(tmp_path):
    # H1: training folder 'out[v2]' (holding run/) next to 'outv' (holding another run/)
    own, other = tmp_path / 'out[v2]' / 'run', tmp_path / 'outv' / 'run'
    write(own, ['run_000000100.safetensors'])
    write(other, ['run_000000200.safetensors', 'run_000000300.safetensors'], start=2_000_000)
    assert base.get_latest_save_path(trainer(own)) == str(own / 'run_000000100.safetensors')
    base.clean_up_saves(trainer(own, keep=1))
    assert listing(other) == ['run_000000200.safetensors', 'run_000000300.safetensors']
    # and with nothing of its own, nothing foreign is selected
    os.remove(own / 'run_000000100.safetensors')
    assert base.get_latest_save_path(trainer(own)) is None


def test_a_bracketed_job_never_prunes_another_job(tmp_path):
    # H2: jobs 'style[2]' and 'style2' in one training root
    own, other = tmp_path / 'style[2]', tmp_path / 'style2'
    write(own, ['style[2]_000000100.safetensors', 'style[2]_000000200.safetensors'])
    write(other, ['style2_000000100.safetensors', 'style2_000000200.safetensors', 'style2_000000300.safetensors'])
    base.clean_up_saves(trainer(own, 'style[2]', keep=1))
    assert listing(other) == ['style2_000000100.safetensors', 'style2_000000200.safetensors',
                              'style2_000000300.safetensors']
    assert listing(own) == ['style[2]_000000200.safetensors']


def test_prefix_sharing_jobs_and_derived_exports_are_not_this_runs_saves(tmp_path):
    # F: a later derived export is kept and never chosen; a prefix-sharing job in the same folder is left alone
    write(tmp_path, ['run_000000100.safetensors', 'run_000000200.safetensors', 'run_000000200-comfy.safetensors',
                     'run_extra_000000300.safetensors', 'running_000000300.safetensors'])
    assert base.get_latest_save_path(trainer(tmp_path)) == str(tmp_path / 'run_000000200.safetensors')
    base.clean_up_saves(trainer(tmp_path, keep=1))
    assert listing(tmp_path) == ['run_000000200-comfy.safetensors', 'run_000000200.safetensors',
                                 'run_extra_000000300.safetensors', 'running_000000300.safetensors']


def test_state_files_are_never_checkpoints(tmp_path):
    write(tmp_path, ['optimizer.pt', 'training_extras.pt'])
    assert base.get_latest_save_path(trainer(tmp_path, 'optimizer')) is None
    assert base.get_latest_save_path(trainer(tmp_path, 'training_extras')) is None


def test_keep_one_keeps_the_newest_lora_and_the_newest_adapter(tmp_path):
    # E2: with an adapter, the LoRA is {job}_LoRA and the IP adapter {job}_ip; one budget used to delete the newest LoRA
    names = []
    for step in (100, 200, 300):
        names += [f'run_LoRA_{step:09d}.safetensors', f'run_ip_{step:09d}.safetensors']
    write(tmp_path, names)
    t = trainer(tmp_path, keep=1, network=True, named_lora=True,
                adapter_config=SimpleNamespace(type='ip', train=True))
    base.clean_up_saves(t)
    assert listing(tmp_path) == ['run_LoRA_000000300.safetensors', 'run_ip_000000300.safetensors']
    assert base.get_latest_save_path(t, base._lora_save_name(t)) == str(tmp_path / 'run_LoRA_000000300.safetensors')
    assert base.get_latest_save_path(t, base._adapter_save_name(t)) == str(tmp_path / 'run_ip_000000300.safetensors')


def test_kinds_this_run_does_not_write_are_left_alone(tmp_path):
    # an earlier configuration of the same job (a LoRA named {job} before an adapter made it {job}_LoRA)
    write(tmp_path, ['run_000000100.safetensors', 'run_000000200.safetensors',
                     'run_LoRA_000000300.safetensors', 'run_LoRA_000000400.safetensors'])
    t = trainer(tmp_path, keep=1, network=True, named_lora=True)
    base.clean_up_saves(t)
    assert listing(tmp_path) == ['run_000000100.safetensors', 'run_000000200.safetensors',
                                 'run_LoRA_000000400.safetensors']


def test_the_final_save_is_never_pruned_and_ranks_by_write_time(tmp_path):
    write(tmp_path, ['run.safetensors', 'run_000000100.safetensors', 'run_000000200.safetensors'])
    assert base.get_latest_save_path(trainer(tmp_path)) == str(tmp_path / 'run_000000200.safetensors')
    os.utime(tmp_path / 'run.safetensors', (3_000_000, 3_000_000))  # a final save rewritten after the periodic ones
    assert base.get_latest_save_path(trainer(tmp_path)) == str(tmp_path / 'run.safetensors')
    base.clean_up_saves(trainer(tmp_path, keep=1))
    assert listing(tmp_path) == ['run.safetensors', 'run_000000200.safetensors']


def test_decorator_beside_a_lora_has_its_own_name(tmp_path):
    t = trainer(tmp_path, network=True, decorator_config=SimpleNamespace())
    assert base._lora_save_name(t) == 'run' and base._decorator_save_name(t) == 'run_decorator'
    kinds = {kind: name for kind, name, _ in base._save_artifact_kinds(t)}
    assert kinds['LoRA'] == 'run' and kinds['decorator'] == 'run_decorator'
    base._refuse_colliding_save_names(t)
    # decorator-only training keeps its old name, so its existing saves still resume
    alone = trainer(tmp_path, decorator_config=SimpleNamespace(), is_fine_tuning=False)
    assert base._decorator_save_name(alone) == 'run'


def test_two_artifacts_sharing_a_name_are_refused(tmp_path):
    t = trainer(tmp_path, network=True, named_lora=True, embed_config=SimpleNamespace(trigger='run_LoRA'))
    with pytest.raises(ValueError, match='would both be saved as run_LoRA'):
        base._refuse_colliding_save_names(t)


def test_adapter_discovery_uses_the_writers_name(tmp_path):
    # a reference adapter is written as {job}_adapter (discovery used to look for {job}_ref); with only an embedding
    # beside it the writer suffixes it too
    assert base._adapter_save_name(trainer(tmp_path, network=True,
                                           adapter_config=SimpleNamespace(type='reference', train=True))) == 'run_adapter'
    assert base._adapter_save_name(trainer(tmp_path, embed_config=SimpleNamespace(trigger='tok'),
                                           adapter_config=SimpleNamespace(type='t2i', train=True))) == 'run_t2i'
    assert base._adapter_save_name(trainer(tmp_path, adapter_config=SimpleNamespace(type='ip', train=True))) == 'run'


# ---- Wan 2.2 14B split LoRAs, through the model's real save_lora/load_lora ----

def wan(split=True, high=True, low=True):
    from extensions_built_in.diffusion_models.wan22.wan22_14b_model import Wan2214bModel
    model = SimpleNamespace(network=SimpleNamespace(network_config=SimpleNamespace(split_multistage_loras=split)),
                            train_high_noise=high, train_low_noise=low)
    model.lora_save_suffixes = lambda: Wan2214bModel.lora_save_suffixes(model)
    model.save_lora = lambda sd, path, meta=None: Wan2214bModel.save_lora(model, sd, path, meta)
    model.load_lora = lambda path: Wan2214bModel.load_lora(model, path)
    return model


def wan_state(step):
    return {f'diffusion_model.transformer_1.blocks.0.lora_A.weight': torch.full((2, 2), float(step)),
            f'diffusion_model.transformer_2.blocks.0.lora_A.weight': torch.full((2, 2), -float(step))}


def wan_trainer(root, model, keep=2):
    return trainer(root, keep=keep, network=True, sd=model)


def save_wan(root, model, steps, start=1_000_000):
    os.makedirs(root, exist_ok=True)
    for i, step in enumerate(steps):
        path = os.path.join(root, f'run_{step:09d}.safetensors')
        model.save_lora(wan_state(step), path)
        for written in os.listdir(root):
            if written.startswith(f'run_{step:09d}'):
                os.utime(os.path.join(root, written), (start + i, start + i))


def wan_latest(t):
    return base.get_latest_save_path(t, base._lora_save_name(t), suffixes=base._lora_save_suffixes(t))


def test_wan_split_pair_is_found_and_loads_both_halves(tmp_path):
    model = wan()
    save_wan(tmp_path, model, (100, 200))
    assert listing(tmp_path) == ['run_000000100_high_noise.safetensors', 'run_000000100_low_noise.safetensors',
                                 'run_000000200_high_noise.safetensors', 'run_000000200_low_noise.safetensors']
    t = wan_trainer(tmp_path, model)
    latest = wan_latest(t)
    assert latest == str(tmp_path / 'run_000000200_high_noise.safetensors')
    loaded = model.load_lora(latest)
    assert {k: v[0, 0].item() for k, v in loaded.items()} == {k: v[0, 0].item() for k, v in wan_state(200).items()}


def test_wan_newest_save_missing_a_half_falls_back_to_the_last_complete_one(tmp_path, capsys):
    model = wan()
    save_wan(tmp_path, model, (100, 200))
    os.remove(tmp_path / 'run_000000200_low_noise.safetensors')  # interrupted between the two halves
    t = wan_trainer(tmp_path, model)
    assert wan_latest(t) == str(tmp_path / 'run_000000100_high_noise.safetensors')
    assert 'Skipping the incomplete save' in capsys.readouterr().out
    os.remove(tmp_path / 'run_000000100_low_noise.safetensors')
    with pytest.raises(ValueError, match='none is complete'):
        wan_latest(t)


def test_wan_single_stage_writes_and_finds_one_file(tmp_path):
    model = wan(low=False)
    os.makedirs(tmp_path, exist_ok=True)
    model.save_lora({'diffusion_model.blocks.0.lora_A.weight': torch.ones(2, 2)},
                    str(tmp_path / 'run_000000100.safetensors'))
    assert listing(tmp_path) == ['run_000000100_high_noise.safetensors']
    assert wan_latest(wan_trainer(tmp_path, model)) == str(tmp_path / 'run_000000100_high_noise.safetensors')


def test_wan_unsplit_layout_is_one_file(tmp_path):
    model = wan(split=False)
    save_wan(tmp_path, model, (100,))
    assert listing(tmp_path) == ['run_000000100.safetensors']
    assert wan_latest(wan_trainer(tmp_path, model)) == str(tmp_path / 'run_000000100.safetensors')


def test_wan_pruning_counts_a_split_save_once(tmp_path):
    model = wan()
    save_wan(tmp_path, model, (100, 200, 300))
    base.clean_up_saves(wan_trainer(tmp_path, model, keep=1))
    assert listing(tmp_path) == ['run_000000300_high_noise.safetensors', 'run_000000300_low_noise.safetensors']


def test_wan_pruning_keeps_the_last_complete_save_behind_an_interrupted_one(tmp_path):
    model = wan()
    save_wan(tmp_path, model, (100, 200, 300))
    os.remove(tmp_path / 'run_000000300_low_noise.safetensors')
    base.clean_up_saves(wan_trainer(tmp_path, model, keep=1))
    assert listing(tmp_path) == ['run_000000200_high_noise.safetensors', 'run_000000200_low_noise.safetensors',
                                 'run_000000300_high_noise.safetensors']


def rawr_trainer(root, model):
    return SimpleNamespace(save_root=str(root), _rawr_checkpoint_config={}, network=object(), is_fine_tuning=False,
                           train_config=SimpleNamespace(merge_network_on_save=False), sd=model,
                           embedding=None, adapter=None, decorator=None, ema=None, snr_gos=None)


def test_rawr_refuses_a_split_lora_layout_before_writing(tmp_path):
    t = rawr_trainer(tmp_path, wan())
    with pytest.raises(ValueError, match='split_multistage_loras: false'):
        rawr_pair.refuse_unsupported_writer(t)
    with pytest.raises(ValueError, match='split_multistage_loras: false'):
        rawr_pair.begin(t)
    assert not os.path.exists(tmp_path / rawr_pair.DIRECTORY)
    # one-file layouts keep their pairs
    rawr_pair.refuse_unsupported_writer(rawr_trainer(tmp_path, wan(split=False)))
    assert rawr_pair.begin(rawr_trainer(tmp_path, SimpleNamespace()))['slot'] == 'a'


def test_a_split_layout_without_a_pair_mode_is_not_refused(tmp_path):
    # with EMA (or an embedding/adapter) RAWR takes the plain save path, which writes split files fine
    t = rawr_trainer(tmp_path, wan())
    t.ema = object()
    rawr_pair.refuse_unsupported_writer(t)
    assert rawr_pair.begin(t) is None


def test_find_save_sets_groups_by_step_and_extension(tmp_path):
    write(tmp_path, ['run_000000100.safetensors', 'run_000000100.pt', 'run.safetensors'])
    os.makedirs(tmp_path / 'run_000000200')  # a Diffusers folder
    sets = find_save_sets(str(tmp_path), 'run')
    assert {(s.step, s.extension) for s in sets} == {(100, '.pt'), (100, '.safetensors'), (200, ''),
                                                     (None, '.safetensors')}
