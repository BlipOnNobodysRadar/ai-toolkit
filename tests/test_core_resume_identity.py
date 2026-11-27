"""Optimizer state, EMA state and network weights resume only into what they were saved for (docs/STATE_MACHINES.md,
OPUS-RES-02, OPUS-PARAM-ID-01, OPUS-EMA-01). A real tiny LoRASpecialNetwork, real AdamW and the real EMA class; the
save is the real BaseSDTrainProcess.save, and the resume runs the real trainer methods in _run_training's order
(discovery, load_weights, the key checks, the optimizer load, setup_ema and the EMA restore) on a trainer namespace.
The diffusion model and data are replaced by a deterministic gradient. CPU only."""
import os
from types import SimpleNamespace

import pytest
import torch

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit import checkpoint_identity
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork

base = BaseSDTrainProcess


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block_a = torch.nn.Linear(4, 4, bias=False)
        self.block_b = torch.nn.Linear(4, 4, bias=False)  # same shape: a selection swap keeps every group size


def network(only='block_a'):
    torch.manual_seed(0)
    config = NetworkConfig(type='lora', linear=2, linear_alpha=2)
    net = LoRASpecialNetwork(None, Toy(), lora_dim=2, alpha=2, train_text_encoder=False, target_lin_modules=['Toy'],
                             is_transformer=True, network_config=config,
                             only_if_contains=None if only is None else [only])
    net.apply_to(None, None, False, True)
    net.to('cpu', dtype=torch.float32)
    return net


def trainer(root, only='block_a', optimizer='adamw', ema=False, pretrained=None, model='base-v1'):
    net = network(only)
    params = list(net.parameters())
    opt = (torch.optim.AdamW if optimizer == 'adamw' else torch.optim.Adam)(params, lr=0.01)
    t = SimpleNamespace(
        job=SimpleNamespace(name='run'), save_root=str(root), network=net, sd=SimpleNamespace(),
        network_config=SimpleNamespace(pretrained_lora_path=pretrained), embed_config=None, embedding=None,
        decorator=None, adapter=None, snr_gos=None, named_lora=False, is_fine_tuning=False,
        accelerator=SimpleNamespace(is_main_process=True), optimizer=opt, lr_scheduler=SimpleNamespace(),
        params=[{'params': params}], model_config=SimpleNamespace(name_or_path=model, arch='toy'),
        train_config=SimpleNamespace(lr_scheduler='constant', merge_network_on_save=False, start_step=None,
                                     ema_config=SimpleNamespace(use_ema=ema, ema_decay=0.9, use_feedback=False,
                                                                feedback_rate=0.0, param_multiplier=1.0)),
        save_config=SimpleNamespace(dtype='float16', max_step_saves_to_keep=4),  # exports are lower precision
        meta={'training_info': {}}, step_num=0, epoch_num=0, start_step=0, _checkpoint_completed_step=0, ema=None,
        post_save_hook=lambda path: None)
    t.update_training_metadata = lambda: t.meta['training_info'].update(step=t.step_num, epoch=0)
    t.clean_up_saves = lambda: base.clean_up_saves(t)
    t.load_training_state_from_metadata = lambda path: base.load_training_state_from_metadata(t, path)
    t.setup_ema = lambda: base.setup_ema(t)
    base._parameter_manifest(t, opt)
    if ema:
        t.setup_ema()
    return t


def train(t, steps):
    for _ in range(steps):
        for p in t.network.parameters():
            p.grad = p.detach() * 0.3 + 0.05
        t.optimizer.step()
        t.optimizer.zero_grad(set_to_none=True)
        if t.ema is not None:
            t.ema.update()
        t.step_num += 1
        t._checkpoint_completed_step = t.step_num


def save(t):
    base.save(t, t.step_num)


def resume(t):
    """_run_training's resume order, with the real methods."""
    path = base.get_latest_save_path(t, base._lora_save_name(t), suffixes=base._lora_save_suffixes(t))
    if path is not None:
        extra = base.load_weights(t, path)
        base._check_model_lora_ignores_on_resume(t, path, extra)
        base._check_resumed_network_keys(t, path)
    base._load_optimizer_state(t, t.optimizer, False)
    t.setup_ema()
    base._restore_ema_resume_state(t)
    return path


def weights(t):
    return [p.detach().clone() for p in t.network.parameters()]


def moments(t):
    return [s['exp_avg'].clone() for s in t.optimizer.state.values()]


def same(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def test_a_save_resumes_its_own_optimizer_state(tmp_path, capsys):
    a = trainer(tmp_path)
    train(a, 3)
    save(a)
    b = trainer(tmp_path)
    resume(b)
    assert b.step_num == 3
    assert same(moments(b), moments(a))
    # the network checkpoint is a float16 export: compare at that precision
    assert all(torch.equal(x.half(), y.half()) for x, y in zip(weights(b), weights(a)))
    assert 'Optimizer state is from the same save as' in capsys.readouterr().out


def test_a_same_shaped_module_swap_is_refused(tmp_path):
    a = trainer(tmp_path, only='block_a')
    train(a, 2)
    save(a)
    b = trainer(tmp_path, only='block_b')
    with pytest.raises(ValueError, match='saved for a different network'):
        resume(b)
    assert b.optimizer.state_dict()['state'] == {}


def test_resume_loads_moments_when_the_original_gpu_is_unavailable(tmp_path, monkeypatch):
    a = trainer(tmp_path)
    train(a, 3)
    with monkeypatch.context() as patch:
        # Serialize the device label of a GPU which is absent on this CPU runner.
        registry = [(0, lambda storage: 'cuda:19', lambda storage, location: None)]
        patch.setattr(torch.serialization, '_package_registry', registry + torch.serialization._package_registry)
        save(a)
    b = trainer(tmp_path)
    resume(b)
    assert b.step_num == 3
    assert same(moments(b), moments(a))


def test_the_swap_is_refused_by_the_optimizer_check_too(tmp_path):
    # the network check comes first in a resume; the optimizer check holds on its own (also the RAWR path)
    a = trainer(tmp_path, only='block_a')
    train(a, 2)
    save(a)
    b = trainer(tmp_path, only='block_b')
    checkpoint = str(tmp_path / 'run_000000002.safetensors')
    b._resume_record = {'path': checkpoint,
                        'artifacts': [(checkpoint, checkpoint_identity.read_checkpoint_generation(checkpoint))]}
    b.step_num = 2
    for rawr in (False, True):
        b._rawr_paired_optimizer_path = str(tmp_path / 'optimizer.pt')
        with pytest.raises(ValueError, match='different trainable parameters'):
            base._load_optimizer_state(b, b.optimizer, rawr)
        assert b.optimizer.state_dict()['state'] == {}


def test_a_reordered_parameter_list_is_refused(tmp_path):
    a = trainer(tmp_path)
    train(a, 2)
    save(a)
    b = trainer(tmp_path)
    params = list(b.network.parameters())[::-1]  # same names and shapes, other positions
    b.optimizer = torch.optim.AdamW(params, lr=0.01)
    b._resume_manifest = None
    base._parameter_manifest(b, b.optimizer)
    with pytest.raises(ValueError, match='different trainable parameters'):
        resume(b)


def test_another_optimizer_class_is_refused(tmp_path):
    a = trainer(tmp_path)
    train(a, 2)
    save(a)
    with pytest.raises(ValueError, match='holds torch.optim.adamw.AdamW state'):
        resume(trainer(tmp_path, optimizer='adam'))


def stop_during_optimizer_save(monkeypatch):
    from training_extras import rawr_checkpoint

    def stopped(*args, **kwargs):
        raise KeyboardInterrupt  # a UI Stop landing after the checkpoint, before optimizer.pt
    monkeypatch.setattr(rawr_checkpoint, 'atomic_save', stopped)


def test_a_save_stopped_before_its_optimizer_state_resumes_from_the_last_complete_save(tmp_path, monkeypatch, capsys):
    a = trainer(tmp_path)
    train(a, 2)
    save(a)
    at_two, moments_two = weights(a), moments(a)
    train(a, 2)
    with monkeypatch.context() as m:
        stop_during_optimizer_save(m)
        with pytest.raises(KeyboardInterrupt):
            save(a)
    assert (tmp_path / 'run_000000004.safetensors').exists()
    os.utime(tmp_path / 'run_000000004.safetensors')  # newest on any clock
    b = trainer(tmp_path)
    assert resume(b) == str(tmp_path / 'run_000000002.safetensors')
    assert b.step_num == 2 and same(moments(b), moments_two)
    assert all(torch.equal(x.half(), y.half()) for x, y in zip(weights(b), at_two))
    assert 'has no optimizer state' in capsys.readouterr().out


def test_reading_the_saved_generation_does_not_pin_optimizer_pt(tmp_path):
    # discovery memory-maps optimizer.pt to read its record; on Windows a live mapping would make the next save's
    # replace of optimizer.pt fail
    from training_extras import rawr_checkpoint
    a = trainer(tmp_path)
    train(a, 2)
    save(a)
    path = str(tmp_path / 'optimizer.pt')
    generation = checkpoint_identity.read_optimizer_generation(path)
    assert generation == checkpoint_identity.read_checkpoint_generation(str(tmp_path / 'run_000000002.safetensors'))
    rawr_checkpoint.atomic_save({'state': {}, 'param_groups': []}, path)
    assert checkpoint_identity.read_optimizer_generation(path) is None


def test_an_optimizer_state_of_no_remaining_checkpoint_is_refused(tmp_path, monkeypatch):
    a = trainer(tmp_path)
    train(a, 2)
    save(a)
    train(a, 2)
    with monkeypatch.context() as m:
        stop_during_optimizer_save(m)
        with pytest.raises(KeyboardInterrupt):
            save(a)
    os.remove(tmp_path / 'run_000000002.safetensors')  # pruned or deleted
    b = trainer(tmp_path)
    with pytest.raises(ValueError, match='does not belong to'):
        resume(b)
    assert b.optimizer.state_dict()['state'] == {}


def test_optimizer_state_with_nothing_resumed_is_refused(tmp_path):
    a = trainer(tmp_path)
    train(a, 2)
    save(a)
    os.remove(tmp_path / 'run_000000002.safetensors')
    with pytest.raises(ValueError, match='no checkpoint of run was found'):
        resume(trainer(tmp_path))


def test_a_pretrained_lora_never_takes_this_folders_optimizer_state(tmp_path):
    a = trainer(tmp_path / 'old')
    train(a, 2)
    save(a)
    os.makedirs(tmp_path / 'new')
    os.replace(tmp_path / 'old' / 'optimizer.pt', tmp_path / 'new' / 'optimizer.pt')
    pretrained = str(tmp_path / 'old' / 'run_000000002.safetensors')
    b = trainer(tmp_path / 'new', pretrained=pretrained)
    base.load_weights(b, pretrained)
    with pytest.raises(ValueError, match='optimizer state from an earlier run'):
        base._load_optimizer_state(b, b.optimizer, False)
    # without the stray file the pretrained weights load as a fresh start
    os.remove(tmp_path / 'new' / 'optimizer.pt')
    c = trainer(tmp_path / 'new', pretrained=pretrained)
    base.load_weights(c, pretrained)
    base._load_optimizer_state(c, c.optimizer, False)
    assert c.step_num == 0


def test_a_pretrained_partial_load_is_not_strict(tmp_path):
    a = trainer(tmp_path / 'old', only=None)  # both blocks
    save(a)
    pretrained = str(tmp_path / 'old' / 'run_000000000.safetensors')
    b = trainer(tmp_path / 'new', only='block_b', pretrained=pretrained)
    base.load_weights(b, pretrained)
    base._check_resumed_network_keys(b, pretrained)  # no refusal: a pretrained LoRA loads partially by design
    assert b.network.last_load_report['unexpected']  # block_a's weights, dropped


def test_extra_saved_modules_are_refused_on_resume(tmp_path):
    a = trainer(tmp_path, only=None)
    train(a, 1)
    save(a)
    with pytest.raises(ValueError, match='have no module in this network'):
        resume(trainer(tmp_path, only='block_a'))


def write_legacy(t):
    """What the previous version saved: no generation in the checkpoint, no record in optimizer.pt."""
    save(t)
    from safetensors import safe_open
    from safetensors.torch import save_file
    path = os.path.join(t.save_root, f'run_{t.step_num:09d}.safetensors')
    with safe_open(path, framework='pt') as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    meta.pop(checkpoint_identity.META_GENERATION)
    save_file(tensors, path, meta)
    state = torch.load(os.path.join(t.save_root, 'optimizer.pt'), weights_only=True)
    state.pop(checkpoint_identity.KEY)
    state.pop(checkpoint_identity.EMA_KEY, None)
    torch.save(state, os.path.join(t.save_root, 'optimizer.pt'))


def test_legacy_optimizer_state_loads_without_an_exact_claim(tmp_path, capsys):
    a = trainer(tmp_path)
    train(a, 2)
    write_legacy(a)
    b = trainer(tmp_path)
    resume(b)
    assert same(moments(b), moments(a))
    out = capsys.readouterr().out
    assert 'Legacy optimizer state' in out and 'not certified as an exact continuation' in out


def test_legacy_optimizer_state_cannot_pair_with_a_new_checkpoint(tmp_path):
    a = trainer(tmp_path)
    train(a, 2)
    write_legacy(a)
    train(a, 1)
    a.network.save_weights(str(tmp_path / 'run_000000003.safetensors'), dtype=torch.float16,
                           metadata={checkpoint_identity.META_GENERATION: '"' + 'a' * 32 + '"',
                                     'training_info': '{"step": 3, "completed_steps": 3}'})
    with pytest.raises(ValueError, match='has no save record'):
        resume(trainer(tmp_path))


def test_a_changed_base_model_warns(tmp_path):
    a = trainer(tmp_path)
    train(a, 2)
    save(a)
    b = trainer(tmp_path, model='base-v2')
    with pytest.warns(RuntimeWarning, match='base model settings changed .*name_or_path'):
        resume(b)
    assert same(moments(b), moments(a))


def test_a_moss_lora_round_trip_resumes_strictly(tmp_path):
    # the strict check must not refuse a model whose key conversion splits and fuses tensors (MOSS q/k/v)
    from test_moss_image import _network
    net = _network()
    path = str(tmp_path / 'run_000000001.safetensors')
    net.save_weights(path, dtype=torch.float32)
    fresh = _network()
    fresh.load_weights(path)
    assert fresh.last_load_report == {'not_loaded': [], 'unexpected': []}


# ---- EMA: the export is the EMA weights; resume continues the raw trajectory ----

def ema_state(t):
    return [s.clone() for s in t.ema.shadow_params]


def test_ema_resume_continues_the_raw_weights_and_the_shadow_exactly(tmp_path):
    a = trainer(tmp_path, ema=True)
    train(a, 3)
    save(a)
    assert not t_is_eval(a)
    b = trainer(tmp_path, ema=True)
    resume(b)
    assert same(weights(b), weights(a)) and same(ema_state(b), ema_state(a)) and same(moments(b), moments(a))
    train(a, 2)
    train(b, 2)
    assert same(weights(b), weights(a)) and same(ema_state(b), ema_state(a))


def t_is_eval(t):
    return not t.ema._is_train_mode


def test_ema_export_differs_from_the_raw_weights(tmp_path):
    # what made the old resume wrong: the checkpoint holds the EMA weights, not the training weights
    a = trainer(tmp_path, ema=True)
    train(a, 3)
    save(a)
    b = trainer(tmp_path)
    base.load_weights(b, str(tmp_path / 'run_000000003.safetensors'))
    halves = lambda values: [v.half() for v in values]
    assert same(halves(weights(b)), halves(ema_state(a)))
    assert not same(halves(weights(b)), halves(weights(a)))


def test_an_ema_checkpoint_resumed_without_ema_restores_the_raw_weights(tmp_path, capsys):
    a = trainer(tmp_path, ema=True)
    train(a, 3)
    save(a)
    b = trainer(tmp_path, ema=False)
    resume(b)
    assert b.ema is None and same(weights(b), weights(a))
    assert 'EMA is now off' in capsys.readouterr().out


def test_ema_enabled_on_a_checkpoint_saved_without_it_starts_from_the_resumed_weights(tmp_path, capsys):
    a = trainer(tmp_path)
    train(a, 3)
    save(a)
    b = trainer(tmp_path, ema=True)
    resume(b)
    assert same(ema_state(b), weights(b))
    assert 'EMA starts from the resumed weights' in capsys.readouterr().out


def test_a_legacy_ema_checkpoint_says_it_is_not_exact(tmp_path, capsys):
    a = trainer(tmp_path, ema=True)
    train(a, 3)
    write_legacy(a)
    b = trainer(tmp_path, ema=True)
    resume(b)
    assert 'continues from them with a fresh EMA (not an exact continuation)' in capsys.readouterr().out


# ---- review follow-ups (Codex 13:30): whole-record validation, generation-or-content binding, honest naming ----

def rewrite_optimizer_pt(root, edit):
    from training_extras import rawr_checkpoint
    path = os.path.join(str(root), 'optimizer.pt')
    state = torch.load(path, weights_only=True)
    edit(state)
    rawr_checkpoint.atomic_save(state, path)


def resume_up_to_ema_restore(t):
    path = base.get_latest_save_path(t, base._lora_save_name(t))
    base.load_weights(t, path)
    base._check_resumed_network_keys(t, path)
    base._load_optimizer_state(t, t.optimizer, False)
    t.setup_ema()


@pytest.mark.parametrize('entry', ['raw', 'shadow'])
def test_a_bad_late_ema_entry_restores_nothing(tmp_path, entry):
    a = trainer(tmp_path, ema=True)
    train(a, 3)
    save(a)
    rewrite_optimizer_pt(tmp_path, lambda s: s[checkpoint_identity.EMA_KEY][entry].__setitem__(-1, torch.zeros(3)))
    b = trainer(tmp_path, ema=True)
    resume_up_to_ema_restore(b)
    before, shadow_before = weights(b), ema_state(b)
    with pytest.raises(ValueError, match=f'saved EMA {entry} weight'):
        base._restore_ema_resume_state(b)
    assert same(weights(b), before) and same(ema_state(b), shadow_before)


def test_changed_ema_settings_are_reported_and_used(tmp_path, capsys):
    a = trainer(tmp_path, ema=True)
    train(a, 3)
    save(a)
    b = trainer(tmp_path, ema=True)
    b.train_config.ema_config.ema_decay = 0.8
    resume(b)
    assert b.ema.decay == 0.8 and same(ema_state(b), ema_state(a))
    assert 'EMA settings changed since this checkpoint (decay: 0.9 -> 0.8)' in capsys.readouterr().out


def strip_generation(path):
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(path, framework='pt') as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    meta.pop(checkpoint_identity.META_GENERATION)
    save_file(tensors, path, meta)


@pytest.mark.parametrize('start_step', [None, 2])
def test_new_optimizer_state_beside_an_unidentified_checkpoint_is_refused(tmp_path, start_step):
    # equal steps are no proof of pairing, and neither is an explicit start_step
    a = trainer(tmp_path)
    train(a, 2)
    save(a)
    strip_generation(str(tmp_path / 'run_000000002.safetensors'))
    b = trainer(tmp_path)
    b.train_config.start_step = start_step
    with pytest.raises(ValueError, match='its format carries no save record'):
        resume(b)
    assert b.optimizer.state_dict()['state'] == {}


def test_formats_without_metadata_are_bound_by_content(tmp_path):
    # the embedding writer has no room for the generation: its file hash is recorded instead
    embedding = tmp_path / 'tok_000000002.safetensors'
    embedding.write_bytes(b'vector bytes')
    identity = {'generation': 'a' * 32, 'content': {embedding.name: checkpoint_identity.sha256(str(embedding))}}
    assert checkpoint_identity.belongs(identity, str(embedding), None)
    assert checkpoint_identity.belongs(identity, str(tmp_path / 'lora.safetensors'), 'a' * 32)
    embedding.write_bytes(b'other bytes')
    assert not checkpoint_identity.belongs(identity, str(embedding), None)
    assert not checkpoint_identity.belongs(identity, str(tmp_path / 'unknown.safetensors'), None)


def test_an_unreadable_save_record_is_refused(tmp_path):
    a = trainer(tmp_path)
    train(a, 2)
    save(a)
    rewrite_optimizer_pt(tmp_path, lambda s: s[checkpoint_identity.KEY].__setitem__('version', 99))
    b = trainer(tmp_path)
    with pytest.raises(ValueError, match='save record this version cannot read'):
        resume(b)


def with_free_parameters(t, first, second):
    """Two same-shaped trainable parameters that belong to no module, in the given order."""
    t.optimizer.add_param_group({'params': [first, second]})
    t._resume_manifest = None
    return base._parameter_manifest(t, t.optimizer)


def test_unnamed_parameters_are_refused_before_training(tmp_path):
    t = trainer(tmp_path)
    manifest = with_free_parameters(t, torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.ones(3)))
    with pytest.raises(ValueError, match='2 trainable parameters .* belong to no module'):
        checkpoint_identity.require_named(manifest)  # what _run_training calls right after building the optimizer


def test_reordered_unnamed_parameters_are_refused_without_loading_moments(tmp_path):
    # a positional record (a fixture namespace can still write one) passes every shape check after a reorder;
    # certified restore must refuse it before load_state_dict
    x, y = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.ones(3))
    a = trainer(tmp_path)
    with_free_parameters(a, x, y)
    x.grad, y.grad = torch.full((3,), .1), torch.full((3,), -.7)
    train(a, 2)
    save(a)
    record = torch.load(tmp_path / 'optimizer.pt', weights_only=True)[checkpoint_identity.KEY]
    assert [e[0] for e in record['parameters'][1]] == ['?1.0', '?1.1']
    b = trainer(tmp_path)
    with_free_parameters(b, torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.zeros(3)))  # swapped
    with pytest.raises(ValueError, match='records 2 parameters only by position'):
        resume(b)
    assert b.optimizer.state_dict()['state'] == {}


def test_named_producers_are_certified(tmp_path):
    # negative controls: the embedding's token vectors are text-encoder parameters, and any module the model holds
    # names its parameters; neither is positional
    t = trainer(tmp_path)
    t.sd.text_encoder = torch.nn.Embedding(4, 3)
    t.sd.extra_head = torch.nn.Linear(3, 3)
    t.optimizer.add_param_group({'params': [t.sd.text_encoder.weight, t.sd.extra_head.weight]})
    t._resume_manifest = None
    manifest = base._parameter_manifest(t, t.optimizer)
    checkpoint_identity.require_named(manifest)
    assert [e[0] for e in manifest[1]] == ['text_encoder.weight', 'sd.extra_head.weight']
    train(t, 1)
    save(t)
    b = trainer(tmp_path)
    b.sd.text_encoder = torch.nn.Embedding(4, 3)
    b.sd.extra_head = torch.nn.Linear(3, 3)
    b.optimizer.add_param_group({'params': [b.sd.text_encoder.weight, b.sd.extra_head.weight]})
    b._resume_manifest = None
    resume(b)
    assert same(moments(b), moments(t))


def test_legacy_factored_optimizer_state_still_loads(tmp_path):
    # negative control: no generic "every state tensor has the parameter's shape" rule (Adafactor keeps row and
    # column statistics)
    a = trainer(tmp_path)
    a.optimizer = torch.optim.Adafactor(list(a.network.parameters()), lr=0.01)
    a._resume_manifest = None
    train(a, 2)
    write_legacy(a)
    shapes = {tuple(v.shape) for st in a.optimizer.state.values() for v in st.values() if torch.is_tensor(v)}
    b = trainer(tmp_path)
    b.optimizer = torch.optim.Adafactor(list(b.network.parameters()), lr=0.01)
    b._resume_manifest = None
    resume(b)
    assert b.optimizer.state_dict()['state'] and len(shapes) > 1


def test_both_halves_of_a_split_save_are_bound(tmp_path):
    from safetensors.torch import save_file
    high, low = str(tmp_path / 'run_000000002_high_noise.safetensors'), str(tmp_path / 'run_000000002_low_noise.safetensors')
    meta = lambda g: {checkpoint_identity.META_GENERATION: f'"{g}"', 'training_info': '{"step": 2, "completed_steps": 2}'}
    save_file({'w': torch.ones(1)}, high, meta('a' * 32))
    save_file({'w': torch.ones(1)}, low, meta('b' * 32))  # left over from another save
    t = trainer(tmp_path)
    t._latest_save_files = {high: [high, low]}
    base.load_training_state_from_metadata(t, high)
    assert [g for _, g in t._resume_record['artifacts']] == ['a' * 32, 'b' * 32]
    identity = {'version': 1, 'generation': 'a' * 32, 'step': 2, 'optimizer': 'x', 'parameters': [], 'content': {}}
    with pytest.raises(ValueError, match=r'does not belong to .*low_noise.*a different save'):
        checkpoint_identity.check(identity, {'optimizer': 'x', 'parameters': []}, t._resume_record['artifacts'], 2,
                                  True, high, 'optimizer.pt')


def test_a_failed_optimizer_save_prunes_nothing_and_the_resume_falls_back(tmp_path, monkeypatch, capsys):
    # non-RAWR saves print and continue when optimizer.pt cannot be written; pruning then (keep=1) deleted the only
    # checkpoint the old optimizer.pt belongs to
    from training_extras import rawr_checkpoint
    a = trainer(tmp_path)
    a.save_config.max_step_saves_to_keep = 1
    train(a, 2)
    save(a)
    moments_two = moments(a)
    train(a, 2)

    def disk_full(*args, **kwargs):
        raise OSError('No space left on device')
    with monkeypatch.context() as m:
        m.setattr(rawr_checkpoint, 'atomic_save', disk_full)
        save(a)
    assert 'Keeping older saves' in capsys.readouterr().out
    assert sorted(os.listdir(tmp_path)) == ['optimizer.pt', 'run_000000002.safetensors', 'run_000000004.safetensors']
    os.utime(tmp_path / 'run_000000004.safetensors')
    b = trainer(tmp_path)
    assert resume(b) == str(tmp_path / 'run_000000002.safetensors')
    assert b.step_num == 2 and same(moments(b), moments_two)
    # the next successful save prunes as usual
    train(a, 1)
    save(a)
    assert sorted(os.listdir(tmp_path)) == ['optimizer.pt', 'run_000000005.safetensors']
