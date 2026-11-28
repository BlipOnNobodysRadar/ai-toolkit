"""Accumulation window, OOM and commit-stage protocol (docs/STATE_MACHINES.md M1/M2, issue groups SM-01..SM-03).

Real SDTrainer instance (constructed without __init__), real BaseSDTrainProcess loop methods, real Accelerate,
real optimizers, the real RAWR serializers and BaseSDTrainProcess.save. Only the diffusion forward is a scalar
fixture, and CUDA OOM is injected where a real one would surface. `iterate()` repeats run()'s call order for one
loop iteration (setup, hook under the OOM protocol, periodic save, end of iteration); every body is production code.
CPU only: run with CUDA_VISIBLE_DEVICES=-1.
"""
import copy
import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from toolkit.optimizer import get_optimizer
from toolkit.optimizers import pending_grads
from training_extras import rawr_checkpoint as rc
from training_extras.optim import create_optimizer, create_scheduler

OOM = torch.cuda.OutOfMemoryError
SCHEDULER = dict(first_cycle_max_steps=5, warmup_steps=1, min_lr=1e-5, gamma=.9, cycle_multiplier=1.5)
LOW_PRECISION = [('adafactor', {}), ('automagic', {}), ('adam8', {}), ('prodigy8bit', {}),
                 ('automagic3', {'fused': False}), ('automagic4', {'fused': False}),
                 ('adamconvrot', {'fused': False}), ('automagicexperiment', {'fused': False})]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    already = torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError('CUDA operation in CPU regression')
    for name in ('_lazy_init', 'init', 'current_device', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda *a, **kw: None)
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    assert already or not torch.cuda.is_initialized()
    AcceleratorState._reset_state(reset_partial_state=True)


def trainer(tmp_path, gas=1, rawr=True, optimizer='adamw', optimizer_params=None, dtype=torch.float64,
            max_grad_norm=0.):
    """A real SDTrainer whose model is one parameter vector; train_single_accumulation is the scalar fixture."""
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    p = torch.nn.Parameter(torch.tensor([.25, -.5], dtype=dtype))
    if optimizer == 'sgd':
        raw = torch.optim.SGD([p], lr=1.)
    else:
        raw = get_optimizer([p], optimizer, .01, dict(optimizer_params or {}))
    config = rc.configuration(raw, 40, SCHEDULER) if rawr else None
    scheduler = (create_scheduler(raw, 'easy:RAWR', 40, SCHEDULER) if rawr
                 else torch.optim.lr_scheduler.LambdaLR(raw, lambda _: 1.))
    if rawr:
        scheduler.step(0)
    accelerator = Accelerator(cpu=True)
    wrapped, scheduler = accelerator.prepare(raw, scheduler)
    f = SDTrainer.__new__(SDTrainer)
    f.__dict__.update(
        accelerator=accelerator, optimizer=wrapped, lr_scheduler=scheduler, params=[{'params': [p]}],
        sd=SimpleNamespace(is_multistage=False),
        train_config=SimpleNamespace(gradient_accumulation_steps=gas, gradient_accumulation=1, optimizer=optimizer,
                                     max_grad_norm=max_grad_norm, merge_network_on_save=False,
                                     do_paramiter_swapping=False, steps=1000),
        model_config=SimpleNamespace(low_vram=False), steps_this_boundary=0, timer=lambda _: nullcontext(),
        adapter=None, ema=None, embedding=None, decorator=None, snr_gos=None, named_lora=False,
        is_fine_tuning=False, network=None, job=SimpleNamespace(name='tiny'), save_root=str(tmp_path),
        save_config=SimpleNamespace(dtype='float32'), meta={'training_info': {'step': 0}}, progress_bar=None,
        modules_being_trained=[], training_extras=None, step_num=0, grad_accumulation_step=1,
        is_grad_accumulation_step=False, _checkpoint_completed_step=0, num_consecutive_oom=0,
        _rawr_checkpoint_config=config, oom_plan={}, seen_updates=[])
    f.update_training_metadata = lambda: f.meta['training_info'].update(step=f.step_num)
    f.clean_up_saves = lambda: None
    f.post_save_hook = lambda _: None

    def train_one(target, accum_scale):
        where = f.oom_plan.get(f.step_num)
        if where == 'forward':
            raise OOM('CUDA out of memory. Tried to allocate 2.00 GiB (injected before backward)')
        torch.manual_seed(1000 + f.step_num)  # stochastic-rounding accumulation draws from the CPU RNG
        x = p if p.dtype == torch.float64 else p.float()
        loss = (x - target).square().mean()
        accelerator.backward(loss * accum_scale)
        if where == 'after_backward':
            raise OOM('CUDA out of memory. Tried to allocate 2.00 GiB (injected after backward)')
        return loss.detach()
    f.train_single_accumulation = train_one
    raw.register_step_pre_hook(lambda opt, *a: f.seen_updates.append(
        (f.step_num, None if p.grad is None else p.grad.detach().clone())))
    f._install_update_commit_observers(raw)
    return f, p, raw


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def target(step):
    return torch.tensor([step * .1 + .2, -.3 - step * .05], dtype=torch.float64)


def iterate(f, step, raw, save_every=None, start=0):
    """One loop iteration in run()'s order (run() inlines only the batch fetch and logging around these calls)."""
    f.step_num = step
    f.is_grad_accumulation_step = True
    f._setup_accumulation_iteration()
    torch.manual_seed(5000 + step)
    _, did_oom = f._run_train_step(step, [target(step)], raw)
    if save_every and step != start and step % save_every == 0:
        f._save_at_loop_step()
    f._end_loop_iteration(step)
    return did_oom


# ---------------------------------------------------------------- SM-01: window reset after a skipped OOM

@pytest.mark.parametrize('where', ['forward', 'after_backward'])
def test_oom_skips_the_whole_window_and_the_next_update_is_complete(tmp_path, where):
    f, p, raw = trainer(tmp_path, gas=4, rawr=False, optimizer='sgd')
    f.oom_plan = {1: where}
    oom = [iterate(f, step, raw) for step in range(10)]
    assert oom == [False, True] + [False] * 8
    # the window holding steps 0-1 is dropped; the next two windows are steps 2-5 and 6-9
    assert [s for s, _ in f.seen_updates] == [5, 9]
    start = torch.tensor([.25, -.5], dtype=torch.float64)
    expected = sum((start - target(s)) / 4 for s in range(2, 6))
    assert torch.allclose(f.seen_updates[0][1], expected, rtol=0, atol=1e-12)
    assert f._checkpoint_completed_step == 10


def test_whole_epoch_window_restarts_after_oom(tmp_path):
    f, p, raw = trainer(tmp_path, gas=-1, rawr=False, optimizer='sgd')
    f.oom_plan = {2: 'after_backward'}
    for step in range(4):
        iterate(f, step, raw)
    # epoch rollover marks the next iteration as the update (the batch fetch does this in run())
    f.step_num = 4
    f.is_grad_accumulation_step = False
    f.grad_accumulation_step = 0
    f._run_train_step(4, [target(4)], raw)
    start = torch.tensor([.25, -.5], dtype=torch.float64)
    assert [s for s, _ in f.seen_updates] == [4]
    # only the microbatches after the OOM (3 and 4), summed: the documented whole-epoch semantics
    assert torch.allclose(f.seen_updates[0][1], (start - target(3)) + (start - target(4)), rtol=0, atol=1e-12)


@pytest.mark.parametrize('gas,oom_step', [(1, 4), (2, 4), (2, 5), (3, 4)])
def test_oom_on_a_save_step_publishes_a_pair_that_resumes_like_the_uninterrupted_run(tmp_path, gas, oom_step):
    a, pa, raw_a = trainer(tmp_path / 'a', gas=gas)
    a.oom_plan = {oom_step: 'after_backward'}
    for step in range(oom_step + 1):
        iterate(a, step, raw_a, save_every=oom_step)
    saved = torch.load(tmp_path / 'a/optimizer.pt', map_location='cpu', weights_only=True)
    resume_at = rc.completed_step(oom_step, a._checkpoint_completed_step, a._rawr_checkpoint_config)
    assert resume_at == oom_step + 1  # the skipped step counts as done, as in the uninterrupted run
    b, pb, raw_b = trainer(tmp_path / 'b', gas=gas)
    pb.data.copy_(pa.data)  # the pair's model file; this fixture's model is the one parameter
    b.optimizer.load_state_dict(saved)
    captured = rc.capture(saved)
    for g in b.optimizer.param_groups:  # base startup resets declared LR before RAWR construction
        g['lr'] = g['initial_lr'] = .01
    b.lr_scheduler = create_scheduler(b.optimizer, 'easy:RAWR', 40, SCHEDULER)
    assert rc.restore(b.lr_scheduler, captured, b._rawr_checkpoint_config, resume_at)
    b.optimizer.zero_grad()
    b.grad_accumulation_step = rc.restore_training_state(b.optimizer, captured, b.train_config, resume_at)
    assert b.grad_accumulation_step == a.grad_accumulation_step == 1
    for step in range(resume_at, resume_at + 4):
        iterate(a, step, raw_a)
        iterate(b, step, raw_b)
        assert torch.equal(pa, pb)
        assert equal(a.optimizer.state_dict(), b.optimizer.state_dict())
        assert equal(a.lr_scheduler.state_dict(), b.lr_scheduler.state_dict())
        assert (pa.grad is None and pb.grad is None) or torch.equal(pa.grad, pb.grad)


def test_training_state_refuses_a_position_restore_would_reject(tmp_path):
    # the pre-fix loop state after an OOM on an update iteration: counter reset to 0 and the step not completed
    f, p, _ = trainer(tmp_path, gas=1)
    f.save(None)  # an earlier committed save (step 0)
    previous = (tmp_path / 'optimizer.pt').read_bytes()
    f.step_num, f.grad_accumulation_step, f._checkpoint_completed_step = 4, 0, 4
    with pytest.raises(RuntimeError, match='Could not save RAWR') as caught:
        f.save(4)
    assert 'accumulation position' in str(caught.value.__cause__)
    assert (tmp_path / 'optimizer.pt').read_bytes() == previous


# ---------------------------------------------------------------- SM-01: commit stage (W5)

def test_oom_inside_optimizer_step_aborts_without_completing_the_step(tmp_path):
    f, p, raw = trainer(tmp_path, gas=1, rawr=False)
    real_step = raw.step

    def step_then_oom(*args, **kwargs):
        real_step(*args, **kwargs)  # parameters and moments already changed
        raise OOM('CUDA out of memory (injected inside optimizer.step)')
    raw.step = step_then_oom
    before = p.detach().clone()
    f.step_num = 3
    f._setup_accumulation_iteration()
    with pytest.raises(RuntimeError, match='cannot be skipped or retried'):
        f._run_train_step(3, [target(3)], raw)
    assert not torch.equal(p, before)
    assert f._checkpoint_completed_step == 0 and f.num_consecutive_oom == 0


def test_generic_trainer_update_is_observed_through_the_optimizer(tmp_path):
    # a trainer whose hook_train_loop knows nothing about commit stages (slider trainers etc.)
    f, p, raw = trainer(tmp_path, gas=1, rawr=False)

    def hook(batch):
        (p - batch[0]).square().mean().backward()
        raw.step()
        raise OOM('CUDA out of memory (injected after a generic optimizer step)')
    f.hook_train_loop = hook
    with pytest.raises(RuntimeError, match=r'after optimizer.step\(\) began'):
        f._run_train_step(0, [target(0)], raw)


def test_unobservable_optimizer_aborts_on_update_iterations_only(tmp_path):
    f, p, raw = trainer(tmp_path, gas=2, rawr=False)
    f._remove_update_commit_observers()
    f._update_commit_observable = False
    f.hook_train_loop = lambda batch: (_ for _ in ()).throw(OOM('CUDA out of memory (injected)'))
    f.is_grad_accumulation_step = True
    assert f._run_train_step(0, [target(0)], raw) == (None, True)
    f.is_grad_accumulation_step = False
    with pytest.raises(RuntimeError, match='cannot observe'):
        f._run_train_step(1, [target(1)], raw)


def test_fused_backward_oom_aborts_but_a_forward_oom_is_skipped(tmp_path):
    f, p, raw = trainer(tmp_path, gas=1, rawr=False, optimizer='automagic3', dtype=torch.bfloat16)
    assert pending_grads.steps_in_backward(f.optimizer)
    before = p.detach().clone()
    f.oom_plan = {0: 'forward'}
    assert iterate(f, 0, raw) is True
    assert torch.equal(p, before)
    f.oom_plan = {1: 'after_backward'}
    with pytest.raises(RuntimeError, match='fused backward update began'):
        iterate(f, 1, raw)
    assert not torch.equal(p, before)  # the fused hooks already applied this step's update


@pytest.mark.parametrize('name,params,steps_in_backward', [
    ('automagic2', {}, True), ('automagic3', {}, True), ('automagic3', {'fused': False}, False),
    ('automagic4', {}, True), ('adamconvrot', {'fused': True}, True), ('adamconvrot', {}, False),
    ('automagicexperiment', {'fused': True}, True), ('adamw', {'fused': True}, False), ('adamw', {}, False),
])
@pytest.mark.parametrize('gas,batches', [(1, 1), (2, 1), (-1, 1), (1, 2)])
def test_fused_backward_optimizers_refuse_multi_backward_accumulation(tmp_path, name, params, steps_in_backward,
                                                                       gas, batches):
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    p = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = get_optimizer([p], name, 1e-4, dict(params))
    assert pending_grads.steps_in_backward(optimizer) is steps_in_backward  # torch AdamW fused=True steps in step()
    f = SimpleNamespace(train_config=SimpleNamespace(gradient_accumulation_steps=gas, gradient_accumulation=batches,
                                                     optimizer=name))
    if steps_in_backward and (gas, batches) != (1, 1):
        with pytest.raises(ValueError, match='inside backward'):
            base._check_accumulation_support(f, optimizer)
    else:
        base._check_accumulation_support(f, optimizer)


# ---------------------------------------------------------------- SM-02: hidden low-precision buffers

def one_backward(p, seed=7, scale=.25):
    # Mixed-sign elements; the first microbatch (seed 100) dominates with the opposite sign, so a window that lost
    # it flips every element's update, which even a sign-like first step (Automagic4) shows.
    weights = torch.randn(p.shape, generator=torch.Generator().manual_seed(0)) * (10. if seed == 100 else -1.)
    torch.manual_seed(seed)
    ((p.float() * weights).sum() * scale).backward()


@pytest.mark.parametrize('name,params', LOW_PRECISION)
def test_zero_grad_drops_the_hidden_buffer(name, params):
    p = torch.nn.Parameter(torch.full((4, 4), .5, dtype=torch.bfloat16))
    optimizer = get_optimizer([p], name, 1e-3, dict(params))
    one_backward(p)
    assert p.grad is None and hasattr(p, pending_grads.ACCUM_GRAD_ATTR)
    optimizer.zero_grad(set_to_none=True)
    assert p.grad is None and not hasattr(p, pending_grads.ACCUM_GRAD_ATTR)


@pytest.mark.parametrize('name,params', LOW_PRECISION)
def test_rawr_capture_restores_the_hidden_buffer_where_it_was(name, params):
    p = torch.nn.Parameter(torch.full((4, 4), .5, dtype=torch.bfloat16))
    optimizer = get_optimizer([p], name, 1e-3, dict(params))
    one_backward(p)
    pending = getattr(p, pending_grads.ACCUM_GRAD_ATTR).clone()
    host = SimpleNamespace(train_config=SimpleNamespace(gradient_accumulation_steps=4, gradient_accumulation=1),
                           grad_accumulation_step=1, step_num=0, optimizer=optimizer, _rawr_stochastic=None)
    state = rc.training_state(host, 1)
    assert torch.equal(state['gradients'][0][0], pending) and state['gradient_locations'] == [[pending_grads.HIDDEN]]
    q = torch.nn.Parameter(torch.full((4, 4), .5, dtype=torch.bfloat16))
    resumed = get_optimizer([q], name, 1e-3, dict(params))
    assert rc.restore_training_state(resumed, {'payload': {'training_state': state}}, host.train_config, 1) == 2
    assert q.grad is None and torch.equal(getattr(q, pending_grads.ACCUM_GRAD_ATTR), pending)


@pytest.mark.parametrize('name', ['automagic3'])
@pytest.mark.parametrize('stop', [1, 2])
def test_interrupted_low_precision_window_matches_the_uninterrupted_update(name, stop):
    """bf16 Automagic3 unfused: 3 microbatches with stochastic-rounding accumulation, then the update.
    (Automagic4 shares this accumulation path but its controller makes no first-step update, so the control below
    cannot tell windows apart for it; its buffer capture/restore is covered above.)"""
    def fresh():
        p = torch.nn.Parameter(torch.full((4, 4), .5, dtype=torch.bfloat16))
        return p, get_optimizer([p], name, .05, {'fused': False})  # updates span many bf16 ULPs at 0.5

    def finish(p, optimizer, microbatches):
        for k in microbatches:
            one_backward(p, seed=100 + k, scale=.1 * (k + 1))
        torch.manual_seed(999)
        optimizer.step()
    a, opt_a = fresh()
    finish(a, opt_a, range(3))
    lost, opt_lost = fresh()  # what a resume that dropped the pending window would compute
    finish(lost, opt_lost, range(stop, 3))
    assert not torch.equal(a, lost), 'fixture cannot tell a lost window from a kept one'

    b, opt_b = fresh()
    for k in range(stop):
        one_backward(b, seed=100 + k, scale=.1 * (k + 1))
    host = SimpleNamespace(train_config=SimpleNamespace(gradient_accumulation_steps=3, gradient_accumulation=1),
                           grad_accumulation_step=stop, step_num=stop - 1, optimizer=opt_b, _rawr_stochastic=None)
    state = copy.deepcopy(rc.training_state(host, stop))  # periodic save, before the end-of-iteration increment
    c, opt_c = fresh()
    opt_c.load_state_dict(copy.deepcopy(opt_b.state_dict()))
    opt_c.zero_grad()
    assert rc.restore_training_state(opt_c, {'payload': {'training_state': state}}, host.train_config, stop) == stop + 1
    finish(c, opt_c, range(stop, 3))
    assert torch.equal(a, c)


def test_adamconvrot_interrupted_window_matches_the_uninterrupted_update():
    def fresh():
        p = torch.nn.Parameter(torch.full((4, 4), .5, dtype=torch.bfloat16))
        return p, get_optimizer([p], 'adamconvrot', 1e-3, {})
    a, opt_a = fresh()
    one_backward(a)
    torch.manual_seed(999)
    opt_a.step()
    b, opt_b = fresh()
    one_backward(b)
    host = SimpleNamespace(train_config=SimpleNamespace(gradient_accumulation_steps=2, gradient_accumulation=1),
                           grad_accumulation_step=1, step_num=0, optimizer=opt_b, _rawr_stochastic=None)
    state = copy.deepcopy(rc.training_state(host, 1))
    c, opt_c = fresh()
    rc.restore_training_state(opt_c, {'payload': {'training_state': state}}, host.train_config, 1)
    torch.manual_seed(999)
    opt_c.step()
    assert torch.equal(a, c) and not torch.equal(a, torch.full_like(a, .5))


def test_pairs_without_locations_still_restore_into_grad():
    p = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([p], lr=.1)
    state = {'gradient_accumulation_steps': 2, 'gradient_accumulation': 1, 'next_accumulation_step': 2,
             'gradients': [[torch.ones(2)]]}
    config = SimpleNamespace(gradient_accumulation_steps=2, gradient_accumulation=1)
    assert rc.restore_training_state(optimizer, {'payload': {'training_state': state}}, config, 1) == 2
    assert torch.equal(p.grad, torch.ones(2))


def test_hidden_location_falls_back_to_grad_for_an_optimizer_without_the_buffer():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    optimizer = torch.optim.SGD([p], lr=.1)
    state = {'gradient_accumulation_steps': 2, 'gradient_accumulation': 1, 'next_accumulation_step': 2,
             'gradients': [[torch.ones(2, dtype=torch.bfloat16)]], 'gradient_locations': [[pending_grads.HIDDEN]]}
    config = SimpleNamespace(gradient_accumulation_steps=2, gradient_accumulation=1)
    rc.restore_training_state(optimizer, {'payload': {'training_state': state}}, config, 1)
    assert torch.equal(p.grad, torch.ones(2, dtype=torch.bfloat16)) and not hasattr(p, pending_grads.ACCUM_GRAD_ATTR)


def test_training_state_joins_offloaded_grad_copies_before_reading(monkeypatch):
    from toolkit.memory_management import manager_modules
    order = []
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.ones(2)

    class Grads(list):
        def __iter__(self):
            order.append('read')
            return super().__iter__()
    monkeypatch.setattr(manager_modules, 'sync_grad_transfers', lambda: order.append('sync'))
    host = SimpleNamespace(train_config=SimpleNamespace(gradient_accumulation_steps=2, gradient_accumulation=1),
                           grad_accumulation_step=1, step_num=0, _rawr_stochastic=None,
                           optimizer=SimpleNamespace(param_groups=[{'params': Grads([p])}]))
    rc.training_state(host, 1)
    assert order[:2] == ['sync', 'read']


def test_grad_clipping_sees_the_hidden_buffer(tmp_path):
    # Adam-like updates are scale-invariant on a first step, so look at the gradient the update consumed instead
    f, p, raw = trainer(tmp_path, gas=1, rawr=False, optimizer='automagic3', optimizer_params={'fused': False},
                        dtype=torch.bfloat16, max_grad_norm=1e-3)
    iterate(f, 0, raw)
    (step, grad), = f.seen_updates
    assert grad is not None, 'clipping and the step saw no gradient: it was still in the hidden buffer'
    assert grad.float().norm().item() == pytest.approx(1e-3, rel=2e-2)


def test_skipped_oom_discards_the_hidden_buffer(tmp_path):
    f, p, raw = trainer(tmp_path, gas=2, rawr=False, optimizer='automagic3', optimizer_params={'fused': False},
                        dtype=torch.bfloat16)
    f.oom_plan = {1: 'after_backward'}
    iterate(f, 0, raw)
    assert hasattr(p, pending_grads.ACCUM_GRAD_ATTR)
    assert iterate(f, 1, raw) is True
    assert p.grad is None and not hasattr(p, pending_grads.ACCUM_GRAD_ATTR)
    assert f.grad_accumulation_step == 1


# ---------------------------------------------------------------- SM-03: saves and samples keep the window

@pytest.mark.parametrize('rawr', [True, False])
def test_periodic_save_keeps_the_unfinished_window(tmp_path, rawr):
    f, p, raw = trainer(tmp_path, gas=2, rawr=rawr)
    iterate(f, 0, raw)
    pending = p.grad.clone()
    f._save_at_loop_step()
    assert torch.equal(p.grad, pending)


def ui_trainer(p):
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer, t.progress_bar, t.step_num = True, None, 5
    t.optimizer = torch.optim.SGD([p], lr=0.1)
    t.train_config = SimpleNamespace(free_u=False, unload_text_encoder=False, do_paramiter_swapping=False)
    t.update_db_key = lambda *args: None
    t.ensure_params_requires_grad = lambda *args, **kwargs: None
    return t


def test_ui_sample_keeps_the_unfinished_window():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.full((2,), .25)
    seen = []
    t = ui_trainer(p)
    t.should_sample = lambda: True
    t.sample = lambda step: seen.append(None if p.grad is None else p.grad.clone())
    t.maybe_sample()
    assert seen and seen[0] is not None and torch.equal(p.grad, torch.full((2,), .25))


# ---------------------------------------------------------------- R01: commit observers live only as long as run()

def observers(raw, p):
    """(step pre-hooks on the optimizer, gradient hooks on the parameter)."""
    return len(raw._optimizer_step_pre_hooks), len(p._backward_hooks or {})


def run_failing(f, raw, failure, monkeypatch):
    """BaseSDTrainProcess.run() (the real wrapper) around a stand-in for the training body: the real observer install,
    one real loop iteration, then the failure at the point run() would raise it. Model setup and the rest of the
    loop are not reproduced on CPU."""
    def body():
        f._install_update_commit_observers(raw)
        iterate(f, 0, raw)
        if failure == 'save':  # real periodic save; the optimizer state's disk write fails
            disk_full = lambda *a, **kw: (_ for _ in ()).throw(OSError('disk full'))
            monkeypatch.setattr(rc, 'atomic_save', disk_full)
            monkeypatch.setattr('training_extras.rawr_pair.commit', disk_full)
            f.step_num = 1
            f._save_at_loop_step()
        elif failure == 'sample':  # sample() failure propagates out of the loop
            f.sample(1)
        elif failure == 'interrupt':  # Ctrl+C during end_step_hook
            f.end_step_hook()
        raise AssertionError('the stand-in body must fail')
    f._run_training = body
    f.sample = lambda step: (_ for _ in ()).throw(RuntimeError('sampling failed'))
    f.end_step_hook = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
    expected = {'save': RuntimeError, 'sample': RuntimeError, 'interrupt': KeyboardInterrupt}[failure]
    with pytest.raises(expected):
        f.run()


@pytest.mark.parametrize('failure', ['save', 'sample', 'interrupt'])
@pytest.mark.parametrize('optimizer,params,dtype', [('adamw', {}, torch.float64),
                                                     ('automagic3', {}, torch.bfloat16)])  # automagic3: fused
def test_failed_run_removes_its_observers_from_a_reused_optimizer(tmp_path, monkeypatch, failure, optimizer, params,
                                                                  dtype):
    a, p, raw = trainer(tmp_path / 'a', gas=1, optimizer=optimizer, optimizer_params=params, dtype=dtype)
    a._remove_update_commit_observers()
    baseline = observers(raw, p)  # the fixture's own seen_updates step hook
    run_failing(a, raw, failure, monkeypatch)
    assert a._update_commit_handles == [] and observers(raw, p) == baseline
    # a later job in the same process reuses the optimizer: only its own trainer is marked
    b, _, _ = trainer(tmp_path / 'b', gas=1)
    b._install_update_commit_observers(raw)
    a._update_commit_stage = b._update_commit_stage = None
    if pending_grads.steps_in_backward(raw):
        (p.float() * 2).sum().backward()
        expected = 'a fused backward update'
    else:
        raw.step()
        expected = 'optimizer.step()'
    assert a._update_commit_stage is None and b._update_commit_stage == expected
    b._remove_update_commit_observers()
    assert observers(raw, p) == baseline


def test_a_partial_observer_install_is_removed_by_run(tmp_path, monkeypatch):
    f, p, raw = trainer(tmp_path, gas=1, optimizer='automagic3', dtype=torch.bfloat16)
    f._remove_update_commit_observers()
    baseline = observers(raw, p)
    # the parameter hook is registered, then the optimizer pre-hook registration fails
    monkeypatch.setattr(raw, 'register_step_pre_hook', lambda hook: (_ for _ in ()).throw(RuntimeError('no hooks')))
    f._run_training = lambda: f._install_update_commit_observers(raw)
    with pytest.raises(RuntimeError, match='no hooks'):
        f.run()
    assert observers(raw, p) == baseline and f._update_commit_handles == []


def test_reinstalling_observers_does_not_stack_them(tmp_path):
    f, p, raw = trainer(tmp_path, gas=1)
    once = observers(raw, p)
    f._install_update_commit_observers(raw)
    assert observers(raw, p) == once


# ---------------------------------------------------------------- SM-08: graceful stop is marked before the write

@pytest.mark.parametrize('trainer_module,cls,flag', [
    ('extensions_built_in.sd_trainer.DiffusionTrainer', 'DiffusionTrainer', 'should_stop'),
    ('extensions_built_in.sd_trainer.DiffusionTrainer', 'DiffusionTrainer', 'should_return_to_queue'),
    ('extensions_built_in.sd_trainer.UITrainer', 'UITrainer', 'should_stop'),
    ('extensions_built_in.captioner.BaseCaptioner', 'BaseCaptioner', 'should_stop'),
])
def test_stop_is_marked_before_the_blocking_status_write(trainer_module, cls, flag):
    klass = getattr(importlib.import_module(trainer_module), cls)
    t = klass.__new__(klass)
    t.is_ui_trainer, t.is_ui_captioner, t.is_stopping = True, True, False
    t.should_stop = lambda: flag == 'should_stop'
    t.should_return_to_queue = lambda: flag == 'should_return_to_queue'
    seen = []

    async def status(*args):
        return None
    t._update_status = status
    t._run_async_operation = lambda coro: (seen.append(t.is_stopping), coro.close())
    with pytest.raises(Exception, match='Job'):
        t.maybe_stop()
    assert seen == [True]


def test_ui_sample_does_not_change_the_cpu_training_rng():
    import random
    import numpy as np
    from training_extras.rawr_stochastic import rng_state, restore_rng
    t = ui_trainer(torch.nn.Parameter(torch.zeros(2)))
    t.should_sample = lambda: True
    draws = lambda: (torch.rand(()).item(), random.random(), np.random.random())
    t.sample = lambda step: draws()
    saved = rng_state()
    expected = draws()
    restore_rng(saved)
    t.maybe_sample()
    assert draws() == expected


@pytest.mark.parametrize('max_norm,scaling', [(0., False), (1., False), (0., True)])
def test_dpo_finite_loss_overflow_preserves_adam_state(tmp_path, max_norm, scaling):
    f, p, raw = trainer(tmp_path, rawr=False, max_grad_norm=max_norm)
    iterate(f, 0, raw)  # initialize Adam moments
    old_state = copy.deepcopy(raw.state_dict())
    old_param = p.detach().clone()
    if scaling:
        from accelerate.optimizer import AcceleratedOptimizer
        f.accelerator.scaler = torch.amp.GradScaler('cpu')
        f.optimizer = AcceleratedOptimizer(raw, scaler=f.accelerator.scaler)
    f.train_config.dpo_beta = 2.
    f.additional_logs = {}
    f.train_config.dpo_anchor_weight = 1e7
    f.train_config.dpo_sft_weight = 0.0

    def train_one(batch, accum_scale):
        drift = p.half().float().square().mean()
        loss = drift.relu() * f.train_config.dpo_anchor_weight
        assert loss.isfinite()
        f.accelerator.backward(loss * accum_scale)
        assert p.grad.isinf().any()
        f._dpo_metric_totals = torch.ones(6)
        return loss.detach()

    f.train_single_accumulation = train_one
    if scaling:
        iterate(f, 1, raw)
        assert f.optimizer.step_was_skipped
    else:
        with pytest.raises(FloatingPointError, match='Non-finite DPO gradient'):
            iterate(f, 1, raw)
    assert torch.equal(p, old_param)
    assert equal(raw.state_dict(), old_state)


def test_pair_weighted_dpo_window_resets_after_update(tmp_path):
    from toolkit.diffusion_dpo import accumulate_dpo_metrics
    f, _, raw = trainer(tmp_path, gas=2, rawr=False)
    f.train_config.dpo_beta = 2.
    f.train_config.dpo_anchor_weight = f.train_config.dpo_sft_weight = 0.
    f.additional_logs = {}
    train_one = f.train_single_accumulation

    def record(batch, accum_scale):
        weights, chosen, rejected = (
            ([2., 0.], [3., 100.], [5., 0.]) if f.step_num == 0 else ([1.], [-3.], [-4.]))
        weights, chosen, rejected = map(torch.tensor, (weights, chosen, rejected))
        accumulate_dpo_metrics(f, weights, rejected - chosen, chosen, rejected)
        return train_one(batch, accum_scale)

    f.train_single_accumulation = record
    iterate(f, 0, raw)
    assert not f.additional_logs
    iterate(f, 1, raw)
    assert f.additional_logs == pytest.approx({
        'dpo/chosen_delta': 1., 'dpo/rejected_delta': 2., 'dpo/margin': 1.,
        'dpo/accuracy': 2 / 3, 'dpo/chosen_pos_drift': 2., 'dpo/gain_gap': 1.})
    iterate(f, 2, raw)
    iterate(f, 3, raw)
    assert f.additional_logs['dpo/chosen_delta'] == -3.


def test_fused_backward_warns_by_default_and_refuses_in_strict_mode_for_active_dpo_penalty(tmp_path, monkeypatch):
    f, _, _ = trainer(tmp_path, rawr=False)
    optimizer = SimpleNamespace(steps_in_backward=True)
    f.train_config.loss_type = 'dpo'
    f.train_config.dpo_anchor_weight = f.train_config.dpo_sft_weight = 0.
    f._check_accumulation_support(optimizer)
    f.train_config.dpo_anchor_weight = 25.
    monkeypatch.delenv('AITK_STRICT_INVARIANTS', raising=False)
    f._check_accumulation_support(optimizer)
    monkeypatch.setenv('AITK_STRICT_INVARIANTS', '1')
    with pytest.raises(ValueError, match='fused backward-stepping optimizer'):
        f._check_accumulation_support(optimizer)
