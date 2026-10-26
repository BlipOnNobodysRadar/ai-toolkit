"""Production train-hook/save regressions with tiny CPU tensors.

Run with CUDA_VISIBLE_DEVICES='' ACCELERATE_USE_CPU=true. The integration
fixtures use real Accelerate, optimizer factory, SDTrainer update and base save.
Only diffusion forward/model serialization are replaced by a scalar fixture.
"""
import copy
import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from training_extras import rawr_checkpoint as rc
from training_extras.optim import create_optimizer, create_scheduler


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    # In a mixed suite an earlier GPU test may already have initialized CUDA; the patched entry points below still
    # forbid any new CUDA use here, so only a CUDA-free process can also assert that nothing initialized it.
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


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize('param_dtype,storage_dtype', [
    (torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16),
])
def test_saem_cpu_update_and_low_precision_state_resume(param_dtype, storage_dtype):
    p = torch.nn.Parameter(torch.full((3, 3), .5, dtype=param_dtype))
    kwargs = dict(torch_compile=False, beta1_warmup=8, sync_chunk_size=1,
                  state_storage_dtype=storage_dtype)
    opt = create_optimizer([p], 'easy:SimplifiedAdEMAMixExM', .01, kwargs)
    for _ in range(3):
        p.grad = torch.full_like(p, .1)
        opt.step()
    assert not torch.equal(p, torch.full_like(p, .5))
    saved = copy.deepcopy(opt.state_dict())
    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = create_optimizer([p2], 'easy:SimplifiedAdEMAMixExM', .01, kwargs)
    opt2.load_state_dict(saved)
    assert equal(opt.state_dict(), opt2.state_dict()), 'optimizer storage dtype changed on load'
    # BF16 state rounding uses RNG; compare with the same CPU random stream.
    for _ in range(3):
        rng = torch.get_rng_state()
        p.grad = torch.full_like(p, .2)
        opt.step()
        torch.set_rng_state(rng)
        p2.grad = torch.full_like(p2, .2)
        opt2.step()
        assert torch.equal(p, p2)
        assert equal(opt.state_dict(), opt2.state_dict())


def trainer(tmp_path, gas=2):
    base = importlib.import_module('jobs.process.BaseSDTrainProcess')
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    p = torch.nn.Parameter(torch.tensor([.25, -.5], dtype=torch.float64))
    optimizer = create_optimizer([p], 'adamw', .01)
    scheduler_params = dict(first_cycle_max_steps=5, warmup_steps=1,
                            min_lr=1e-5, gamma=.9, cycle_multiplier=1.5)
    config = rc.configuration(optimizer, 12, scheduler_params)
    scheduler = create_scheduler(optimizer, 'easy:RAWR', 12, scheduler_params)
    scheduler.step(0)
    accelerator = Accelerator(cpu=True)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    f = SimpleNamespace(
        accelerator=accelerator, optimizer=optimizer, lr_scheduler=scheduler,
        params=[p], sd=SimpleNamespace(is_multistage=False),
        train_config=SimpleNamespace(gradient_accumulation_steps=gas,
            gradient_accumulation=1, optimizer='adamw', max_grad_norm=0.,
            merge_network_on_save=False),
        model_config=SimpleNamespace(low_vram=False), steps_this_boundary=0,
        timer=lambda _: nullcontext(), adapter=None, ema=None, embedding=None,
        decorator=None, snr_gos=None, named_lora=False, is_fine_tuning=False,
        network=None, job=SimpleNamespace(name='tiny'), save_root=str(tmp_path),
        save_config=SimpleNamespace(dtype='float32'), meta={'training_info': {'step': 0}},
        clean_up_saves=lambda: None, post_save_hook=lambda _: None,
        end_of_training_loop=lambda: None, step_num=0, grad_accumulation_step=1,
        is_grad_accumulation_step=False, _checkpoint_completed_step=0,
        _rawr_checkpoint_config=config,
    )
    f.update_training_metadata = lambda: f.meta['training_info'].update(step=f.step_num)
    def train_one(target, accum_scale):
        loss = (p - target).square().mean()
        accelerator.backward(loss * accum_scale)
        return loss.detach()
    f.train_single_accumulation = train_one
    f.train = lambda target: SDTrainer.hook_train_loop(f, [target])
    f.save = lambda step=None: base.BaseSDTrainProcess.save(f, step)
    return f, p, scheduler_params


def advance(f, index):
    f.step_num = index
    f.is_grad_accumulation_step = f.grad_accumulation_step < f.train_config.gradient_accumulation_steps
    if not f.is_grad_accumulation_step:
        f.grad_accumulation_step = 0
    f.train(torch.tensor([index * .1 + .2, -.3], dtype=torch.float64))
    f._checkpoint_completed_step = index + 1


@pytest.mark.parametrize('stop', [1, 2, 3, 5])
@pytest.mark.parametrize('save_phase', ['periodic', 'end_step'])
def test_actual_save_hook_accumulation_continuation(tmp_path, stop, save_phase):
    a, pa, scheduler_params = trainer(tmp_path / 'a')
    for index in range(stop):
        advance(a, index)
        if index < stop - 1 or save_phase == 'end_step':
            a.grad_accumulation_step += 1
            a.step_num = index + 1
    a.save(a.step_num)
    saved = torch.load(tmp_path / 'a/optimizer.pt', map_location='cpu', weights_only=True)
    b, pb, _ = trainer(tmp_path / 'b')
    pb.data.copy_(pa.data)
    b.optimizer.load_state_dict(saved)
    captured = rc.capture(saved)
    # Actual base startup resets declared LR before RAWR construction.
    for g in b.optimizer.param_groups:
        g['lr'] = g['initial_lr'] = .01
    b.lr_scheduler = create_scheduler(b.optimizer, 'easy:RAWR', 12, scheduler_params)
    assert rc.restore(b.lr_scheduler, captured, b._rawr_checkpoint_config, stop)
    b.optimizer.zero_grad()
    b.grad_accumulation_step = rc.restore_training_state(
        b.optimizer, captured, b.train_config, stop)
    if save_phase == 'periodic':
        a.grad_accumulation_step += 1
    for index in range(stop, stop + 3):
        advance(a, index)
        advance(b, index)
        assert torch.equal(pa, pb)
        assert equal(a.optimizer.state_dict(), b.optimizer.state_dict())
        assert equal(a.lr_scheduler.state_dict(), b.lr_scheduler.state_dict())
        assert (pa.grad is None and pb.grad is None) or torch.equal(pa.grad, pb.grad)
        a.grad_accumulation_step += 1
        b.grad_accumulation_step += 1


@pytest.mark.parametrize('failure', ['write', 'replace'])
def test_actual_save_keeps_previous_optimizer_on_interrupted_write(tmp_path, monkeypatch, failure):
    f, p, scheduler_params = trainer(tmp_path)
    advance(f, 0)
    f.save(0)
    path = tmp_path / 'optimizer.pt'
    previous = path.read_bytes()
    if failure == 'write':
        def fail_save(obj, file, *args, **kwargs):
            if hasattr(file, 'write'):
                file.write(b'interrupted')
            else:
                from pathlib import Path
                Path(file).write_bytes(b'interrupted')
            raise OSError('injected serialization failure')
        monkeypatch.setattr(torch, 'save', fail_save)
    else:
        def fail_replace(*args):
            raise OSError('injected rename failure')
        monkeypatch.setattr(rc.os, 'replace', fail_replace)
    with pytest.raises(RuntimeError, match='Could not save RAWR'):
        f.save(0)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob('*.tmp'))
    # The retained file must also support the next real optimizer update.
    recovered = torch.load(path, map_location='cpu', weights_only=True)
    b, pb, _ = trainer(tmp_path / 'recovered')
    pb.data.copy_(p.data)
    b.optimizer.load_state_dict(recovered)
    captured = rc.capture(recovered)
    for g in b.optimizer.param_groups:
        g['lr'] = g['initial_lr'] = .01
    b.lr_scheduler = create_scheduler(b.optimizer, 'easy:RAWR', 12, scheduler_params)
    rc.restore(b.lr_scheduler, captured, b._rawr_checkpoint_config, 1)
    b.grad_accumulation_step = rc.restore_training_state(b.optimizer, captured, b.train_config, 1)
    f.grad_accumulation_step += 1
    advance(f, 1)
    advance(b, 1)
    assert torch.equal(p, pb)
    assert equal(f.optimizer.state_dict(), b.optimizer.state_dict())


@pytest.mark.parametrize('gas,version,stop,allowed', [
    (1, 0, 2, True), (2, 0, 2, False), (2, 1, 2, True), (2, 1, 3, False),
])
def test_older_checkpoint_boundary_policy(tmp_path, gas, version, stop, allowed):
    a, pa, scheduler_params = trainer(tmp_path / 'a', gas=gas)
    for index in range(stop):
        advance(a, index)
        a.grad_accumulation_step += 1
        a.step_num = index + 1
    state = copy.deepcopy(a.optimizer.state_dict())
    if version == 1:
        state = rc.add_state(state, a.lr_scheduler, a._rawr_checkpoint_config, stop)
    b, pb, _ = trainer(tmp_path / 'b', gas=gas)
    pb.data.copy_(pa.data)
    b.optimizer.load_state_dict(state)
    captured = rc.capture(state)
    for g in b.optimizer.param_groups:
        g['lr'] = g['initial_lr'] = .01
    b.lr_scheduler = create_scheduler(b.optimizer, 'easy:RAWR', 12, scheduler_params)
    if version == 0:
        with pytest.warns(RuntimeWarning, match='Legacy RAWR'):
            rc.restore(b.lr_scheduler, captured, b._rawr_checkpoint_config, stop)
    else:
        rc.restore(b.lr_scheduler, captured, b._rawr_checkpoint_config, stop)
    if not allowed:
        with pytest.raises(ValueError, match='partial gradients'):
            rc.restore_training_state(b.optimizer, captured, b.train_config, stop)
        return
    b.grad_accumulation_step = rc.restore_training_state(b.optimizer, captured, b.train_config, stop)
    for index in range(stop, stop + 3):
        advance(a, index)
        advance(b, index)
        assert torch.equal(pa, pb)
        assert equal(a.optimizer.state_dict(), b.optimizer.state_dict())
        a.grad_accumulation_step += 1
        b.grad_accumulation_step += 1


@pytest.mark.parametrize('corruption', ['config', 'position', 'groups', 'params', 'shape', 'dtype'])
def test_invalid_accumulation_state_rejected_before_gradient_changes(tmp_path, corruption):
    f, p, _ = trainer(tmp_path)
    advance(f, 0)
    state = rc.add_state(f.optimizer.state_dict(), f.lr_scheduler, f._rawr_checkpoint_config,
                         1, training_state=rc.training_state(f, 1))
    captured = rc.capture(state)
    saved = captured['payload']['training_state']
    if corruption == 'config':
        saved['gradient_accumulation_steps'] = 3
    elif corruption == 'position':
        saved['next_accumulation_step'] = 0
    elif corruption == 'groups':
        saved['gradients'] = []
    elif corruption == 'params':
        saved['gradients'][0] = []
    elif corruption == 'shape':
        saved['gradients'][0][0] = torch.zeros(3, dtype=p.dtype)
    elif corruption == 'dtype':
        saved['gradients'][0][0] = torch.zeros_like(p, dtype=torch.float16)
    old_grad = p.grad.clone()
    with pytest.raises(ValueError):
        rc.restore_training_state(f.optimizer, captured, f.train_config, 1)
    assert torch.equal(p.grad, old_grad)


def test_epoch_accumulation_resume_is_explicitly_unsupported(tmp_path):
    f, _, _ = trainer(tmp_path)
    advance(f, 0)
    state = rc.add_state(f.optimizer.state_dict(), f.lr_scheduler, f._rawr_checkpoint_config, 1)
    f.train_config.gradient_accumulation_steps = -1
    with pytest.raises(ValueError, match='dataset position'):
        rc.restore_training_state(f.optimizer, rc.capture(state), f.train_config, 1)
