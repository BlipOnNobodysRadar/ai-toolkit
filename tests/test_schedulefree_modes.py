"""Schedule-free optimizer train/eval mode at training start, on resume, and around evaluation (docs/STATE_MACHINES.md
M15, Codex C12). Real vendored optimizers built by the real factory on CPU fp32 parameters; the gradient is a
deterministic quadratic. BaseSDTrainProcess calls enter_training_mode right after loading optimizer state; the trainer
loop itself is not run here.
"""
import copy

import pytest
import torch

from training_extras.optim import create_optimizer, enter_training_mode, evaluation_optimizer, training_mode

GROUP_MODE = ['easy:NorMuonScheduleFree']  # mode in every param group (serialized with them)
# mode on the optimizer (not serialized). AMUSE has the same layout but compiles its update (torch.compile needs a
# C++ compiler this CPU environment lacks), so it is not exercised here.
SELF_MODE = ['easy:ADOPTScheduleFree', 'easy:FADOPTScheduleFree']
KWARGS = {}
TARGET = torch.tensor([[.3, -.7], [1.1, .2]])


def build(name, start=None):
    p = torch.nn.Parameter(torch.tensor([[1., 2.], [-.5, .25]]) if start is None else start.clone())
    return p, create_optimizer([p], name, .05, KWARGS.get(name))


def step(p, opt):
    opt.zero_grad(set_to_none=True)
    (p - TARGET).square().sum().backward()
    opt.step()


@pytest.mark.parametrize('name', GROUP_MODE + SELF_MODE)
def test_a_fresh_optimizer_can_step_after_enter_training_mode(name):
    p, opt = build(name)
    before = p.detach().clone()
    assert training_mode(opt) is False  # construction leaves it in eval mode (Codex schedulefree-probe)
    enter_training_mode(opt)  # FADOPT's step does not check the mode; the others raise until train()
    assert training_mode(opt) is True and torch.equal(p, before)  # a flag flip: no z state yet
    step(p, opt)
    step(p, opt)  # ADOPT's first step only initializes its second moment
    assert not torch.equal(p, before)


@pytest.mark.parametrize('name', GROUP_MODE + SELF_MODE)
def test_resume_from_a_train_coordinate_checkpoint_continues_exactly(name):
    # toolkit checkpoints are written from the training loop, in train mode: weights y plus the optimizer payload
    a, opt_a = build(name)
    enter_training_mode(opt_a)
    for _ in range(4):
        step(a, opt_a)
    weights, payload = a.detach().clone(), copy.deepcopy(opt_a.state_dict())
    for _ in range(4):
        step(a, opt_a)

    b, opt_b = build(name, start=weights)
    opt_b.load_state_dict(payload)
    enter_training_mode(opt_b)
    assert torch.equal(b, weights)
    for _ in range(4):
        step(b, opt_b)
    assert torch.equal(a, b)


@pytest.mark.parametrize('name', SELF_MODE)
def test_calling_train_on_a_resumed_self_mode_optimizer_would_move_the_weights(name):
    # negative control for the test above: the mode is not in the payload, so train() treats y as x and moves it
    a, opt_a = build(name)
    enter_training_mode(opt_a)
    for _ in range(4):
        step(a, opt_a)
    weights, payload = a.detach().clone(), copy.deepcopy(opt_a.state_dict())
    b, opt_b = build(name, start=weights)
    opt_b.load_state_dict(payload)
    assert training_mode(opt_b) is False
    opt_b.train()
    assert not torch.equal(b, weights)


@pytest.mark.parametrize('name', GROUP_MODE + SELF_MODE)
def test_evaluation_scope_restores_the_entry_mode_and_weights(name):
    p, opt = build(name)
    enter_training_mode(opt)
    for _ in range(4):
        step(p, opt)
    y = p.detach().clone()
    with evaluation_optimizer(opt):
        x = p.detach().clone()
        assert training_mode(opt) is False and not torch.equal(x, y)  # evaluation weights differ after 4 steps
        with evaluation_optimizer(opt):
            pass
        assert training_mode(opt) is False and torch.equal(p, x)  # the nested exit did not switch back
    assert training_mode(opt) is True and torch.allclose(p, y, rtol=0, atol=1e-6)


def test_radam_schedulefree_mode_is_entered_without_touching_weights():
    # its step() is CUDA-only (torch.cuda.current_device), so on CPU only the mode transition is checked; the first
    # step's "not in train mode" failure on CPU is Codex's schedulefree-probe.json
    p = torch.nn.Parameter(torch.ones(2))
    opt = create_optimizer([p], 'easy:RAdamScheduleFree', .001)
    assert training_mode(opt) is False
    enter_training_mode(opt)
    assert training_mode(opt) is True and torch.equal(p, torch.ones(2))
    assert opt.state_dict()['param_groups'][0]['train_mode'] is True  # serialized with the groups


def test_optimizers_without_a_mode_are_left_alone():
    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([p], lr=.1)
    enter_training_mode(opt)
    assert training_mode(opt) is None
    with evaluation_optimizer(opt):
        pass


# ---------------------------------------------------------------- evaluation scope: exact training values back

def run(name, dtype, validate_after=(), seed=0):
    """8 updates; an evaluation scope (as validation uses) after the listed steps. Returns weights, optimizer state and
    the RNG state at the end."""
    torch.manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(8, 8).to(dtype))
    target = torch.randn(8, 8)
    opt = create_optimizer([p], name, .01, KWARGS.get(name))
    enter_training_mode(opt)
    for i in range(8):
        opt.zero_grad(set_to_none=True)
        (p.float() - target).square().sum().backward()
        opt.step()
        if i in validate_after:
            with evaluation_optimizer(opt):
                (p.float() * 2).sum()  # the body reads the evaluation weights
    return p.detach().clone(), copy.deepcopy(opt.state_dict()), torch.get_rng_state()


def same_state(a, b):
    sa, sb = a['state'], b['state']
    return sa.keys() == sb.keys() and all(
        torch.equal(sa[k][f], sb[k][f]) if torch.is_tensor(sa[k][f]) else sa[k][f] == sb[k][f]
        for k in sa for f in sa[k])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('name', GROUP_MODE + SELF_MODE)
def test_repeated_evaluation_scopes_do_not_change_the_training_trajectory(name, dtype):
    plain = run(name, dtype)
    validated = run(name, dtype, validate_after=(1, 3, 5))
    assert torch.equal(plain[0], validated[0]) and same_state(plain[1], validated[1])
    assert torch.equal(plain[2], validated[2])  # eval()'s stochastic-rounding draws were given back


@pytest.mark.parametrize('name', GROUP_MODE + SELF_MODE)
def test_the_old_eval_train_round_trip_moved_bf16_weights(name):
    # negative control: what validation did before (eval() then train() on the live weights)
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8).to(torch.bfloat16))
    opt = create_optimizer([p], name, .01, KWARGS.get(name))
    enter_training_mode(opt)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        p.float().square().sum().backward()
        opt.step()
    y = p.detach().clone()
    opt.eval()
    opt.train()
    assert not torch.equal(p, y)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('name', GROUP_MODE + SELF_MODE)
def test_the_scope_shows_the_optimizers_own_evaluation_point(name, dtype):
    p, opt = build(name)
    p.data = p.data.to(dtype)
    enter_training_mode(opt)
    for _ in range(4):
        step(p, opt)
    twin_p = torch.nn.Parameter(p.detach().clone())
    twin = create_optimizer([twin_p], name, .05, KWARGS.get(name))
    twin.load_state_dict(copy.deepcopy(opt.state_dict()))
    enter_training_mode(twin)
    torch.manual_seed(7)
    twin.eval()
    torch.manual_seed(7)
    with evaluation_optimizer(opt):
        assert torch.equal(p, twin_p)


def test_parameters_storage_and_pending_gradients_survive_the_scope():
    p, opt = build('easy:ADOPTScheduleFree')
    enter_training_mode(opt)
    for _ in range(4):
        step(p, opt)
    p.grad = torch.full_like(p, .5)  # a pending accumulation window
    alias, ptr, y = p, p.data_ptr(), p.detach().clone()
    with evaluation_optimizer(opt):
        assert not torch.equal(p, y)
    assert alias is opt.param_groups[0]['params'][0] and p.data_ptr() == ptr
    assert torch.equal(p, y) and torch.equal(p.grad, torch.full_like(p, .5)) and training_mode(opt) is True


def test_a_failing_body_gives_the_training_values_back():
    p, opt = build('easy:NorMuonScheduleFree')
    enter_training_mode(opt)
    for _ in range(4):
        step(p, opt)
    y = p.detach().clone()
    with pytest.raises(RuntimeError, match='validation failed'):
        with evaluation_optimizer(opt):
            raise RuntimeError('validation failed')
    assert torch.equal(p, y) and training_mode(opt) is True


@pytest.mark.parametrize('name', GROUP_MODE + SELF_MODE)
def test_an_eval_that_fails_part_way_is_undone(name, monkeypatch):
    a = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))
    b = torch.nn.Parameter(torch.tensor([[5., 6.], [7., 8.]]))
    opt = create_optimizer([{'params': [a]}, {'params': [b], 'lr': .02}], name, .05, KWARGS.get(name))
    enter_training_mode(opt)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        (a.square().sum() + b.square().sum()).backward()
        opt.step()
    ya, yb = a.detach().clone(), b.detach().clone()
    real_lerp, calls = torch.Tensor.lerp_, [0]

    def lerp_(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:  # the first parameter was already converted
            raise RuntimeError('injected failure in eval()')
        return real_lerp(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'lerp_', lerp_)
    with pytest.raises(RuntimeError, match='injected failure'):
        with evaluation_optimizer(opt):
            pass
    monkeypatch.setattr(torch.Tensor, 'lerp_', real_lerp)
    assert calls[0] >= 2 and torch.equal(a, ya) and torch.equal(b, yb) and training_mode(opt) is True


def trained(name='easy:ADOPTScheduleFree', dtype=torch.float32, groups=1):
    params = [torch.nn.Parameter(torch.randn(4, 4, generator=torch.Generator().manual_seed(i)).to(dtype))
              for i in range(groups)]
    opt = create_optimizer([{'params': [p]} for p in params], name, .01, KWARGS.get(name))
    enter_training_mode(opt)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        sum(p.float().square().sum() for p in params).backward()
        opt.step()
    return params, opt


@pytest.mark.parametrize('name', GROUP_MODE + SELF_MODE)
def test_the_conversion_leaves_the_training_rng_where_it_was(name):
    params, opt = trained(name, torch.bfloat16)  # bf16: eval() rounds stochastically, drawing from the RNG
    torch.manual_seed(123)
    before = torch.get_rng_state()
    with evaluation_optimizer(opt):
        assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(torch.get_rng_state(), before)


def test_cpu_parameters_never_touch_a_cuda_rng(monkeypatch):
    params, opt = trained()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)  # a visible GPU that holds none of these weights

    def forbidden(*args, **kwargs):
        raise AssertionError('CUDA RNG touched for CPU parameters')
    for api in ('get_rng_state', 'get_rng_state_all', 'set_rng_state', 'set_rng_state_all'):
        monkeypatch.setattr(torch.cuda, api, forbidden)
    with evaluation_optimizer(opt):
        pass
    assert training_mode(opt) is True


def test_a_failed_restore_still_restores_the_others_and_does_not_claim_training(monkeypatch):
    (a, b), opt = trained(groups=2)
    ya, yb = a.detach().clone(), b.detach().clone()
    real_copy = torch.Tensor.copy_
    state = {'armed': False}

    def copy_(self, src, *args, **kwargs):
        if state['armed'] and self is a:
            raise RuntimeError('injected restore failure')
        return real_copy(self, src, *args, **kwargs)
    with pytest.raises(RuntimeError, match='injected restore failure') as caught:
        with evaluation_optimizer(opt):
            state['armed'] = True
            monkeypatch.setattr(torch.Tensor, 'copy_', copy_)
    monkeypatch.setattr(torch.Tensor, 'copy_', real_copy)
    assert any('could not restore' in note and 'reload' in note for note in caught.value.__notes__)
    assert torch.equal(b, yb) and not torch.equal(a, ya)  # the other parameter was still given back
    assert training_mode(opt) is False  # mixed weights are not called training weights


def test_validation_gives_back_the_networks_entry_activation(tmp_path):
    from training_extras import toolkit_validation
    from types import SimpleNamespace

    class Network:
        def __init__(self):
            self.is_active, self.multiplier = True, .7

        def __enter__(self):
            self.is_active = True

        def __exit__(self, *exc):
            self.is_active = False
    params, opt = trained()
    unet = torch.nn.Linear(1, 1)
    unet.train()
    trainer = SimpleNamespace(
        train_config=SimpleNamespace(validation_config=SimpleNamespace(seed=0, validation_sigmas=[.5])),
        device_torch=torch.device('cpu'), sd=SimpleNamespace(torch_dtype=torch.float32, unet=unet),
        network=Network(), optimizer=opt, _validation_cache={'native_batches': []}, additional_logs={},
        save_root=str(tmp_path), step_num=4)
    with pytest.warns(RuntimeWarning):  # no batches: the mean of nothing
        toolkit_validation.validate(trainer)
    assert trainer.network.is_active is True and trainer.network.multiplier == .7 and unet.training
    assert training_mode(opt) is True


@pytest.mark.parametrize('name', GROUP_MODE + SELF_MODE)
def test_a_partial_eval_whose_rollback_also_fails_is_never_marked_training(name, monkeypatch):
    (a, b), opt = trained(name, groups=2)
    ya, yb = a.detach().clone(), b.detach().clone()
    real_lerp, real_copy, calls = torch.Tensor.lerp_, torch.Tensor.copy_, [0]

    def lerp_(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError('injected failure in eval()')
        return real_lerp(self, *args, **kwargs)

    def copy_(self, src, *args, **kwargs):
        if self is a:
            raise RuntimeError('injected rollback failure')
        return real_copy(self, src, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'lerp_', lerp_)
    monkeypatch.setattr(torch.Tensor, 'copy_', copy_)
    with pytest.raises(RuntimeError, match='injected failure in eval') as caught:
        with evaluation_optimizer(opt):
            pass
    monkeypatch.setattr(torch.Tensor, 'lerp_', real_lerp)
    monkeypatch.setattr(torch.Tensor, 'copy_', real_copy)
    assert any('could not restore' in note for note in caught.value.__notes__)
    assert torch.equal(b, yb)  # the independent restore still ran
    assert training_mode(opt) is not True
