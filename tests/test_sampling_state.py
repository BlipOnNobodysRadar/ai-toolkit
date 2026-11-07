"""State that sampling and the prior prediction borrow from training (docs/STATE_MACHINES.md M6/M7, SM-05/SM-06).

Real LoRASpecialNetwork on a tiny bf16 module, the real BaseModel/StableDiffusion.generate_images entry points,
real ExponentialMovingAverage, the real BaseSDTrainProcess.sample and SDTrainer.get_prior_prediction, and the real
sample-step hook installer. Failures are injected where generation or the prior forward would raise. CPU only.
"""
import importlib
from types import SimpleNamespace

import pytest
import torch

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)


def lora(seed=0):
    toy = Toy()
    with torch.no_grad():
        toy.proj.weight.copy_(torch.randn(64, 64, generator=torch.Generator().manual_seed(seed)))
    network = LoRASpecialNetwork(None, toy, lora_dim=4, alpha=4, train_text_encoder=False,
                                 target_lin_modules=['Toy'], is_transformer=True,
                                 network_config=NetworkConfig(type='lora', linear=4, linear_alpha=4))
    network.apply_to(None, toy, False, True)
    with torch.no_grad():
        for key, p in network.named_parameters():
            if key.endswith('lora_up.weight'):
                p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(seed + 1)) * .05)
    network.train()
    return toy, network


def test_merge_round_trip_moves_bf16_base_weights():
    # The mechanism, on the real merge code: why training must not merge for sampling.
    toy, network = lora()
    before = toy.proj.weight.detach().clone()
    for _ in range(20):
        network.merge_in(1.0)
        network.merge_out(1.0)
    assert not torch.equal(toy.proj.weight, before)


def holder(cls_path, network, unet, **extra):
    module, name = cls_path.rsplit('.', 1)
    cls = getattr(importlib.import_module(module), name)
    h = cls.__new__(cls)
    calls = []
    h.__dict__.update(model_config=SimpleNamespace(assistant_lora_path=None, inference_lora_path=None),
                      network=network, device_torch=torch.device('cpu'), torch_dtype=torch.bfloat16,
                      invert_assistant_lora=False, adapter=None, refiner_unet=None, calls=calls, **extra)
    h.unet = unet
    h.save_device_state = lambda: calls.append('save_device_state')
    h.set_device_state_preset = lambda name: calls.append(f'preset:{name}')
    h.restore_device_state = lambda: calls.append('restore_device_state')
    return h, calls


MODELS = ['toolkit.models.base_model.BaseModel', 'toolkit.stable_diffusion_model.StableDiffusion']


def image_configs(multiplier=1.0):
    return [SimpleNamespace(network_multiplier=multiplier)]


@pytest.mark.parametrize('cls_path', MODELS)
def test_failed_generation_gives_back_the_borrowed_training_state(cls_path):
    toy, network = lora()
    network.multiplier = .7
    weights = toy.proj.weight.detach().clone()
    h, calls = holder(cls_path, network, toy, merge_network_for_sampling=False)

    def failing_pipeline(*args, **kwargs):
        torch.rand(5)  # sampling consumes the training RNG
        raise RuntimeError('generation failure')
    h.get_generation_pipeline = failing_pipeline  # BaseModel
    h.__dict__.pop('noise_scheduler', None)  # StableDiffusion fails building its pipeline (no scheduler)
    rng = torch.get_rng_state()
    with pytest.raises((RuntimeError, AttributeError)):
        h.generate_images(image_configs())
    assert calls == ['save_device_state', 'preset:generate', 'restore_device_state']
    assert torch.equal(torch.get_rng_state(), rng)
    assert network.training and network.multiplier == .7 and not network.is_merged_in
    assert torch.equal(toy.proj.weight, weights)


@pytest.mark.parametrize('cls_path', MODELS)
def test_training_samples_leave_frozen_base_weights_bit_identical(cls_path):
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    toy, network = lora()
    weights = toy.proj.weight.detach().clone()
    h, _ = holder(cls_path, network, toy, merge_network_for_sampling=False)
    for _ in range(20):
        state = sampling_state.borrow(h, image_configs(), BlankNetwork())
        assert not network.is_merged_in  # sampling runs through the LoRA hooks
        sampling_state.return_(h, state)
    assert torch.equal(toy.proj.weight, weights)


def test_inference_holders_keep_the_merge():
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    toy, network = lora()
    h, _ = holder(MODELS[0], network, toy)  # no merge_network_for_sampling: an inference-only holder
    state = sampling_state.borrow(h, image_configs(), BlankNetwork())
    assert network.is_merged_in
    sampling_state.return_(h, state)
    assert not network.is_merged_in


def test_entry_eval_mode_is_preserved():
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    toy, network = lora()
    network.eval()
    h, _ = holder(MODELS[0], network, toy, merge_network_for_sampling=False)
    sampling_state.return_(h, sampling_state.borrow(h, image_configs(), BlankNetwork()))
    assert not network.training


# ---------------------------------------------------------------- R02: entry state and owned give-back
# Real LoRASpecialNetwork modules (trained network and auxiliary LoRA) and the real helper; the holder's device-preset
# methods are stubs that record calls (there is one CPU device, so module moves are not exercised here).

def aux_holder(cls_path=MODELS[0], assistant=None, inference=None, invert=False, aux_active=False,
               aux_dtype=torch.bfloat16, network=None):
    _, aux = lora(seed=3)
    aux.force_to('cpu', aux_dtype)
    aux.is_active = aux_active
    toy, _ = lora()
    h, calls = holder(cls_path, network, toy, merge_network_for_sampling=False, assistant_lora=aux)
    h.invert_assistant_lora = invert
    h.model_config = SimpleNamespace(assistant_lora_path=assistant, inference_lora_path=inference)
    return h, calls, aux


def dtypes(module):
    return {t.dtype for m in [module] + list(module.unet_loras) for t in m.parameters()}


def snapshot(module):
    """Every parameter and buffer of the LoRA and its modules: values, dtype and device (lora_up is random, so the
    values are nontrivial)."""
    out = {}
    for i, m in enumerate([module] + list(module.unet_loras)):
        for name, t in list(m.named_parameters()) + list(m.named_buffers()):
            out[(i, name)] = t.detach().clone()
    return out


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype
                                        and a[k].device == b[k].device for k in a)


def round_trip(h):
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    sampling_state.return_(h, sampling_state.borrow(h, image_configs(), BlankNetwork()))


@pytest.mark.parametrize('entry_active', [True, False])
def test_plain_assistant_activation_is_given_back_as_it_was(entry_active):
    h, _, aux = aux_holder(assistant='fixture', aux_active=entry_active, aux_dtype=torch.float32)
    before = snapshot(aux)
    round_trip(h)
    assert aux.is_active is entry_active and same(snapshot(aux), before)  # never moved or cast


@pytest.mark.parametrize('role', ['inference', 'inverted assistant'])
def test_an_auxiliary_lora_switched_on_for_sampling_goes_back_off_unchanged(role):
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    kw = dict(inference='fixture') if role == 'inference' else dict(assistant='fixture', invert=True)
    h, _, aux = aux_holder(**kw, aux_active=False)  # parked as the loaders leave it: CPU, holder dtype
    before = snapshot(aux)
    state = sampling_state.borrow(h, image_configs(), BlankNetwork())
    assert aux.is_active
    sampling_state.return_(h, state)
    assert not aux.is_active and same(snapshot(aux), before)


def test_an_active_inference_lora_stays_active_and_unchanged():
    # Codex's sampling_entry_probe: the phase02 helper returned it inactive and in the holder dtype
    h, _, aux = aux_holder(inference='fixture', aux_active=True)
    before = snapshot(aux)
    round_trip(h)
    assert aux.is_active and same(snapshot(aux), before)


@pytest.mark.parametrize('role', ['inference', 'inverted assistant'])
def test_an_auxiliary_lora_in_another_dtype_is_refused_before_anything_changes(role):
    # Codex auxiliary_lora_roundtrip_probe: fp32 auxiliary, bf16 holder -> the round trip changed 512/512 values
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    toy, network = lora()
    network.multiplier = .7
    kw = dict(inference='fixture') if role == 'inference' else dict(assistant='fixture', invert=True)
    h, calls, aux = aux_holder(**kw, aux_dtype=torch.float32, network=network)
    before = snapshot(aux)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match='cannot borrow it without changing its weights'):
        sampling_state.borrow(h, image_configs(), BlankNetwork())
    assert same(snapshot(aux), before) and not aux.is_active and calls == []
    assert network.training and network.multiplier == .7 and torch.equal(torch.get_rng_state(), rng)


def test_a_floating_buffer_in_another_dtype_counts_as_mixed_placement():
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    h, calls, aux = aux_holder(inference='fixture')
    aux.unet_loras[0].register_buffer('fixture_buffer', torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match='spread over'):
        sampling_state.borrow(h, image_configs(), BlankNetwork())
    assert calls == [] and not aux.is_active


def test_a_premerged_network_is_neither_merged_again_nor_merged_out():
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    toy, network = lora()
    network.merge_in(1.0)
    merged = toy.proj.weight.detach().clone()
    h, _ = holder(MODELS[0], network, toy)  # inference-only holder: merging is allowed
    state = sampling_state.borrow(h, image_configs(), BlankNetwork())
    assert torch.equal(toy.proj.weight, merged)  # not merged a second time
    sampling_state.return_(h, state)
    assert network.is_merged_in and torch.equal(toy.proj.weight, merged)


def test_a_mixed_placement_auxiliary_lora_is_refused_before_anything_changes():
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    toy, network = lora()
    network.multiplier = .7
    h, calls, aux = aux_holder(inference='fixture', network=network)
    aux.unet_loras[0].lora_down.to(torch.float32)
    with pytest.raises(ValueError, match='spread over'):
        sampling_state.borrow(h, image_configs(), BlankNetwork())
    assert calls == [] and not aux.is_active and network.training and network.multiplier == .7


@pytest.mark.parametrize('cls_path', MODELS)
def test_a_preset_that_fails_part_way_is_put_back_and_the_rest_given_back(cls_path):
    toy, network = lora()
    network.multiplier = .7
    h, calls, aux = aux_holder(cls_path, inference='fixture', network=network)
    seen = []
    h.save_device_state = lambda: setattr(h, 'device_state', 'entry snapshot')

    def preset(name):
        h.device_state = 'half-written re-save'  # set_device_state_preset re-saves first, then moves modules
        raise torch.cuda.OutOfMemoryError('CUDA out of memory (injected while moving to the generate preset)')
    h.set_device_state_preset = preset
    h.restore_device_state = lambda: seen.append(h.device_state)
    rng = torch.get_rng_state()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        h.generate_images(image_configs())
    assert seen == ['entry snapshot']  # restored once, from the snapshot taken before any move
    assert network.training and network.multiplier == .7 and not aux.is_active and dtypes(aux) == {torch.bfloat16}
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('cls_path', MODELS)
def test_a_failing_device_restore_neither_hides_the_generation_error_nor_skips_the_rest(cls_path):
    toy, network = lora()
    network.multiplier = .7
    h, calls, aux = aux_holder(cls_path, inference='fixture', network=network)

    def failing_pipeline(*args, **kwargs):
        torch.rand(5)
        raise RuntimeError('generation failure')
    h.get_generation_pipeline = failing_pipeline
    h.__dict__.pop('noise_scheduler', None)

    def restore():
        raise torch.cuda.OutOfMemoryError('CUDA out of memory (injected moving modules back)')
    h.restore_device_state = restore
    rng = torch.get_rng_state()
    with pytest.raises((RuntimeError, AttributeError)) as caught:
        h.generate_images(image_configs())
    assert not isinstance(caught.value, torch.cuda.OutOfMemoryError)  # the generation error is the error
    assert any('moving modules back' in note for note in caught.value.__notes__)
    assert torch.equal(torch.get_rng_state(), rng)
    assert network.training and network.multiplier == .7 and not aux.is_active and dtypes(aux) == {torch.bfloat16}


def test_a_failing_give_back_after_successful_generation_is_raised_after_every_step():
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    toy, network = lora()
    network.multiplier = .7
    h, calls, aux = aux_holder(inference='fixture', network=network)
    state = sampling_state.borrow(h, image_configs(.3), BlankNetwork())
    h.restore_device_state = lambda: (_ for _ in ()).throw(RuntimeError('restore failed'))
    torch.rand(3)
    with pytest.raises(RuntimeError, match='restore failed'):
        sampling_state.return_(h, state)
    assert network.training and network.multiplier == .7 and not aux.is_active
    assert torch.equal(torch.get_rng_state(), state.rng)


def test_a_merge_that_fails_part_way_is_not_merged_out():
    from toolkit import sampling_state
    from toolkit.stable_diffusion_model import BlankNetwork
    toy, network = lora()
    entry = toy.proj.weight.detach().clone()
    module = network.unet_loras[0]
    real_merge = type(module).merge_in

    def merge_then_fail(merge_weight=1.0):
        real_merge(module, merge_weight=merge_weight)  # this module merged, the next one would fail
        raise torch.cuda.OutOfMemoryError('CUDA out of memory (injected during merge_in)')
    module.merge_in = merge_then_fail
    h, calls = holder(MODELS[0], network, toy)
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        sampling_state.borrow(h, image_configs(), BlankNetwork())
    assert any('must be reloaded' in note for note in caught.value.__notes__)
    assert not torch.equal(toy.proj.weight, entry)  # left merged (and reported), not subtracted a second way
    assert calls == [] and network.training


def trainer_for_sample(tmp_path, adapter, ema):
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    t = SDTrainer.__new__(SDTrainer)

    def fail(*args, **kwargs):
        raise RuntimeError('generation failure')
    t.__dict__.update(accelerator=SimpleNamespace(is_main_process=True), save_root=str(tmp_path),
                      sample_config=SimpleNamespace(seed=0, prompts=[], samples=[], walk_seed=False, sampler=None,
                                                    ext='jpg'),
                      adapter_config=None, embedding=None, trigger_word=None, adapter=adapter, ema=ema,
                      sd=SimpleNamespace(generate_images=fail))
    return t


def test_failed_sample_restores_raw_weights_and_adapter_flag(tmp_path):
    from toolkit.custom_adapter import CustomAdapter
    from toolkit.ema import ExponentialMovingAverage
    p = torch.nn.Parameter(torch.ones(3))
    ema = ExponentialMovingAverage([p], decay=.5)
    with torch.no_grad():
        p.fill_(9.)  # raw weights moved on; the EMA shadow is still 1
    adapter = CustomAdapter.__new__(CustomAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.is_sampling = False
    t = trainer_for_sample(tmp_path, adapter, ema)
    with pytest.raises(RuntimeError, match='generation failure'):
        t.sample(5)
    assert torch.equal(p, torch.full((3,), 9.)) and ema._is_train_mode
    assert adapter.is_sampling is False


def test_prior_prediction_oom_restores_adapter_network_and_unet_mode():
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.custom_adapter import CustomAdapter
    adapter = CustomAdapter.__new__(CustomAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.is_active = True
    unet = torch.nn.Linear(1, 1)
    unet.train()

    def oom(**kwargs):
        raise torch.cuda.OutOfMemoryError('CUDA out of memory (injected in the prior forward)')
    t = SDTrainer.__new__(SDTrainer)
    t.__dict__.update(adapter=adapter, network=SimpleNamespace(is_active=True), embedding=None,
                      device_torch=torch.device('cpu'), sd=SimpleNamespace(unet=unet, predict_noise=oom),
                      train_config=SimpleNamespace(unload_text_encoder=False, dtype='float32', cfg_scale=1.,
                                                   do_guidance_loss=False, cfg_rescale=0.))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        t.get_prior_prediction(torch.zeros(1, 1), torch.zeros(1, 1), False, [1.], torch.zeros(1), {}, None,
                               torch.zeros(1, 1))
    assert adapter.is_active and t.network.is_active and unet.training


# ---------------------------------------------------------------- R03/C14: EMA mode transitions, P-04: save()
# Real ExponentialMovingAverage on CPU parameters; copy failures are injected into Tensor.copy_ (deterministic fault
# injection, not a real device-copy failure).

def ema_fixture():
    from toolkit.ema import ExponentialMovingAverage
    params = [torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1) * 2)]
    ema = ExponentialMovingAverage(params)
    ema.shadow_params = [torch.ones(1) * 9, torch.ones(1) * 8]
    return ema, params


def failing_copies(monkeypatch, *failing_calls):
    real = torch.Tensor.copy_
    calls = [0]

    def copy_(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] in failing_calls:
            raise RuntimeError(f'injected copy_ failure #{calls[0]}')
        return real(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'copy_', copy_)
    return lambda: monkeypatch.setattr(torch.Tensor, 'copy_', real)


def values(params):
    return [p.item() for p in params]


def test_ema_eval_that_fails_part_way_gives_the_raw_weights_back_and_stays_training(monkeypatch):
    # Codex ema_entry_probe: the second copy_ failed, leaving [9, 2] with _is_train_mode True
    ema, params = ema_fixture()
    undo = failing_copies(monkeypatch, 2)
    with pytest.raises(RuntimeError, match='#2'):
        ema.eval()
    undo()
    assert values(params) == [1., 2.] and ema._is_train_mode and ema.collected_params is None
    ema.eval()  # a later attempt works normally
    assert values(params) == [9., 8.] and not ema._is_train_mode
    ema.train()
    assert values(params) == [1., 2.] and ema._is_train_mode


def test_ema_eval_whose_rollback_also_fails_claims_evaluation_so_train_retries(monkeypatch):
    ema, params = ema_fixture()
    undo = failing_copies(monkeypatch, 2, 3)  # copy_to's second copy, then the rollback's first copy
    with pytest.raises(RuntimeError, match='#2') as caught:
        ema.eval()
    undo()
    assert any('#3' in note for note in caught.value.__notes__)
    assert not ema._is_train_mode and ema.collected_params is not None  # mixed weights are not called raw
    ema.train()
    assert values(params) == [1., 2.] and ema._is_train_mode


def test_ema_train_that_fails_part_way_can_be_retried(monkeypatch):
    ema, params = ema_fixture()
    ema.eval()
    undo = failing_copies(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        ema.train()
    undo()
    assert not ema._is_train_mode and ema.collected_params is not None
    ema.train()
    assert values(params) == [1., 2.] and ema._is_train_mode


def test_average_parameters_restores_a_partial_copy(monkeypatch):
    ema, params = ema_fixture()
    undo = failing_copies(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        with ema.average_parameters():
            pass
    undo()
    assert values(params) == [1., 2.]


class FailingNetwork:
    """Stub network: records the weights and multiplier save_weights sees, then fails like a full disk."""
    def __init__(self, params):
        self.multiplier, self.params, self.seen = .7, params, None

    def save_weights(self, *args, **kwargs):
        self.seen = (values(self.params), self.multiplier)
        raise OSError('disk full (injected)')


def trainer_for_save(tmp_path, ema, network):
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    t = SDTrainer.__new__(SDTrainer)
    t.__dict__.update(accelerator=SimpleNamespace(is_main_process=True), save_root=str(tmp_path), ema=ema,
                      network=network, is_fine_tuning=False, named_lora=False, embedding=None, adapter=None,
                      decorator=None, snr_gos=None, step_num=3, meta={'training_info': {'step': 3}},
                      job=SimpleNamespace(name='tiny'), optimizer=None, lr_scheduler=None,
                      train_config=SimpleNamespace(merge_network_on_save=False, lr_scheduler='constant'),
                      save_config=SimpleNamespace(dtype='float32'))
    t.update_training_metadata = lambda: None
    return t


@pytest.mark.parametrize('ema_training', [True, False])
def test_failed_save_gives_back_raw_weights_ema_mode_and_multiplier(tmp_path, ema_training):
    ema, params = ema_fixture()
    if not ema_training:
        ema.eval()
    network = FailingNetwork(params)
    t = trainer_for_save(tmp_path, ema, network)
    with pytest.raises(OSError, match='disk full'):
        t.save(3)
    assert network.seen == ([9., 8.], 1.0)  # the save itself saw EMA weights at multiplier 1
    assert network.multiplier == .7
    assert ema._is_train_mode is ema_training
    assert values(params) == ([1., 2.] if ema_training else [9., 8.])


# ---------------------------------------------------------------- sample-step hook ownership

class Scheduler:
    timesteps = [0, 1]

    def step(self, *args, **kwargs):
        return ('class step',)


def hook_holder():
    seen = []
    return SimpleNamespace(sample_step_hook=lambda *a: seen.append(a)), seen


def test_unwrap_restores_a_callers_own_step():
    from toolkit.sample_step_hook import install_sample_step_hooks
    scheduler = Scheduler()
    def callers_step(*args, **kwargs):
        return ('caller step',)
    scheduler.step = callers_step
    h, seen = hook_holder()
    unwrap = install_sample_step_hooks(h, SimpleNamespace(scheduler=scheduler))
    assert scheduler.step(None, 0, None) == ('caller step',) and seen  # wrapped, and the caller's step still runs
    unwrap()
    assert scheduler.step is callers_step


def test_unwrap_of_an_inherited_step_and_repeated_installs():
    from toolkit.sample_step_hook import install_sample_step_hooks
    scheduler = Scheduler()
    h, _ = hook_holder()
    first = install_sample_step_hooks(h, SimpleNamespace(scheduler=scheduler))
    second = install_sample_step_hooks(h, SimpleNamespace(scheduler=scheduler))  # a stale wrapper is replaced
    assert scheduler.step._aitk_orig_step.__func__ is Scheduler.step  # not stacked
    second()
    first()
    assert 'step' not in vars(scheduler) and scheduler.step() == ('class step',)


def test_unwrap_leaves_a_wrapper_installed_over_ours():
    from toolkit.sample_step_hook import install_sample_step_hooks
    scheduler = Scheduler()
    h, _ = hook_holder()
    unwrap = install_sample_step_hooks(h, SimpleNamespace(scheduler=scheduler))
    def outer(*args, **kwargs):
        return ('outer',)
    scheduler.step = outer
    unwrap()
    assert scheduler.step is outer


def test_settle_keeps_the_primary_error_when_logging_fails(monkeypatch):
    from toolkit import sampling_state
    monkeypatch.setattr(sampling_state, 'print_acc', lambda *a: (_ for _ in ()).throw(OSError('console gone')))
    primary = RuntimeError('generation failure')
    sampling_state.settle([lambda: (_ for _ in ()).throw(RuntimeError('restore failed'))], primary)
    assert any('restore failed' in note for note in primary.__notes__)
