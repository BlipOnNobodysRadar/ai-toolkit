import copy
import io
import warnings

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from torch.optim import AdamW

from training_extras import rawr_checkpoint
from training_extras.optim import create_scheduler


LR_A = 0.02
LR_B = 0.05
TOTAL_STEPS = 40


def _scheduler_params(d, mult):
    return {
        "gamma": 0.9,
        "d": d,
        "cycle_multiplier": mult,
        "first_cycle_max_steps": 19,
        "warmup_steps": 2,
        "min_lr": 1e-6,
    }


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # other test files import toolkit modules that build the global accelerator
    # first; Accelerator(cpu=True) refuses an already-initialized state
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def _new_params():
    return [
        torch.nn.Parameter(torch.full((3,), 0.5, dtype=torch.float64)),
        torch.nn.Parameter(torch.full((3,), -0.25, dtype=torch.float64)),
    ]


def _new_optimizer(params, lr_a=LR_A, lr_b=LR_B):
    groups = [{"params": [params[0]], "lr": lr_a},
              {"params": [params[1]], "lr": lr_b}]
    opt = AdamW(groups, betas=(0.5, 0.7), eps=1e-10)
    for group in opt.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return opt


def _make_state(params, d, mult, total=TOTAL_STEPS, lr_a=LR_A, lr_b=LR_B):
    opt = _new_optimizer(params, lr_a, lr_b)
    params_dict = {"d": d, "mult": mult, "total": total}
    config = rawr_checkpoint.configuration(opt, total, _scheduler_params(d, mult))
    sched = create_scheduler(opt, "easy:RAWR", total, _scheduler_params(d, mult))
    sched.step(0)  # fresh startup to match trainer
    return params, opt, sched, config, params_dict


def _advance(params, opt, sched, n):
    for _ in range(n):
        for p in params:
            p.grad = torch.full_like(p, 0.1)
        opt.step()
        sched.step()


def _tensors_equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_tensors_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_tensors_equal(x, y) for x, y in zip(a, b))
    return a == b


def _full_opt_state(opt):
    return opt.state_dict()


def _compare_after(params_a, opt_a, sched_a, params_b, opt_b, sched_b, future=7, compare_scheduler=True):
    _advance(params_a, opt_a, sched_a, future)
    _advance(params_b, opt_b, sched_b, future)
    for pa, pb in zip(params_a, params_b):
        assert torch.equal(pa.detach(), pb.detach())
    assert _tensors_equal(_full_opt_state(opt_a), _full_opt_state(opt_b))
    ua = rawr_checkpoint.underlying_scheduler(sched_a)
    ub = rawr_checkpoint.underlying_scheduler(sched_b)
    if compare_scheduler:
        assert _tensors_equal(ua.state_dict(), ub.state_dict())


def _resume(loaded_opt_state, loaded_config, params2, params_dict, stop):
    opt2 = _new_optimizer(params2, params_dict["lr_a"], params_dict["lr_b"])
    config2 = rawr_checkpoint.configuration(
        opt2, params_dict["total"], _scheduler_params(params_dict["d"], params_dict["mult"]))
    declared = [g["lr"] for g in opt2.param_groups]
    opt2.load_state_dict(copy.deepcopy(loaded_opt_state))
    captured = rawr_checkpoint.capture(loaded_opt_state)
    # Trainer resets group lr and initial_lr to DECLARED LR values.
    for group, lr in zip(opt2.param_groups, declared):
        group["lr"] = lr
        group["initial_lr"] = lr
    sched2 = create_scheduler(opt2, "easy:RAWR", params_dict["total"],
                              _scheduler_params(params_dict["d"], params_dict["mult"]))
    assert rawr_checkpoint.restore(sched2, captured, config2, stop) is True
    return opt2, sched2


@pytest.mark.parametrize("stop", [0, 7, 18, 19, 20, 38])
@pytest.mark.parametrize("d,mult", [(0.45, 1.0), (0.45, 1.5), (0.9, 1.0), (0.9, 1.5)])
def test_exact_continuation_roundtrip(stop, d, mult):
    params, opt, sched, config, pdict = _make_state(_new_params(), d, mult)
    _advance(params, opt, sched, stop)
    state = rawr_checkpoint.add_state(opt.state_dict(), sched, config, stop)
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=True)

    params2 = [torch.nn.Parameter(p.detach().clone()) for p in params]
    pdict2 = dict(pdict, lr_a=LR_A, lr_b=LR_B)
    opt2, sched2 = _resume(loaded, config, params2, pdict2, stop)

    for pa, pb in zip(params, params2):
        assert torch.equal(pa.detach(), pb.detach())
    assert _tensors_equal(opt.state_dict(), opt2.state_dict())
    u = rawr_checkpoint.underlying_scheduler(sched)
    u2 = rawr_checkpoint.underlying_scheduler(sched2)
    assert _tensors_equal(u.state_dict(), u2.state_dict())

    _compare_after(params, opt, sched, params2, opt2, sched2, future=7)


@pytest.mark.parametrize("stop", [0, 7, 19, 38])
def test_accelerator_prepared_continuation(stop):
    params = _new_params()
    opt = _new_optimizer(params)
    config = rawr_checkpoint.configuration(opt, TOTAL_STEPS, _scheduler_params(0.45, 1.5))
    sched = create_scheduler(opt, "easy:RAWR", TOTAL_STEPS, _scheduler_params(0.45, 1.5))
    sched.step(0)
    acc = Accelerator(cpu=True)
    prepared_params = list(params)
    opt, sched = acc.prepare(opt, sched)
    _advance(prepared_params, opt, sched, stop)
    state = rawr_checkpoint.add_state(opt.state_dict(), sched, config, stop)

    params2 = [torch.nn.Parameter(p.detach().clone()) for p in prepared_params]
    pdict2 = {"d": 0.45, "mult": 1.5, "total": TOTAL_STEPS, "lr_a": LR_A, "lr_b": LR_B}
    opt2, sched2 = _resume(state, config, params2, pdict2, stop)
    acc2 = Accelerator(cpu=True)
    prepared2 = list(params2)
    opt2, sched2 = acc2.prepare(opt2, sched2)

    for pa, pb in zip(prepared_params, prepared2):
        assert torch.equal(pa.detach(), pb.detach())
    _compare_after(prepared_params, opt, sched, prepared2, opt2, sched2, future=7)


def test_repeat_resume_metadata_upgraded():
    params, opt, sched, config, pdict = _make_state(_new_params(), 0.45, 1.5)
    _advance(params, opt, sched, 7)
    state = rawr_checkpoint.add_state(opt.state_dict(), sched, config, 7)

    params2 = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt2, sched2 = _resume(state, config, params2, dict(pdict, lr_a=LR_A, lr_b=LR_B), 7)
    _advance(params2, opt2, sched2, 3)
    config2 = config
    state2 = rawr_checkpoint.add_state(opt2.state_dict(), sched2, config2, 10)
    payload2 = state2[rawr_checkpoint.KEY]
    assert payload2["version"] == 1
    assert payload2["trainer_step"] == 10

    params3 = [torch.nn.Parameter(p.detach().clone()) for p in params2]
    opt3, sched3 = _resume(loaded_opt_state=state2, loaded_config=config2, params2=params3,
                          params_dict=dict(pdict, lr_a=LR_A, lr_b=LR_B), stop=10)
    _compare_after(params2, opt2, sched2, params3, opt3, sched3, future=7)


def test_legacy_without_add_state_warns_and_matches():
    params, opt, sched, config, pdict = _make_state(_new_params(), 0.45, 1.5)
    _advance(params, opt, sched, 10)
    legacy = rawr_checkpoint.capture(opt.state_dict())
    assert legacy["payload"] is None

    params2 = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt2 = _new_optimizer(params2)
    config2 = rawr_checkpoint.configuration(opt2, TOTAL_STEPS, _scheduler_params(0.45, 1.5))
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    captured = rawr_checkpoint.capture(opt2.state_dict())
    for group in opt2.param_groups:
        group["initial_lr"] = group["lr"]
    sched2 = create_scheduler(opt2, "easy:RAWR", TOTAL_STEPS, _scheduler_params(0.45, 1.5))
    with pytest.warns(RuntimeWarning):
        assert rawr_checkpoint.restore(sched2, captured, config2, 10) is True

    for group_src, group_dst in zip(opt.param_groups, opt2.param_groups):
        for key in rawr_checkpoint.FIELDS:
            if key in group_src:
                assert group_dst.get(key) == group_src.get(key)
    _compare_after(params, opt, sched, params2, opt2, sched2, future=7, compare_scheduler=False)


@pytest.mark.parametrize("mutation", ["d", "lr", "step", "version", "corrupt"])
def test_changed_metadata_rejected(mutation):
    params, opt, sched, config, pdict = _make_state(_new_params(), 0.45, 1.5)
    _advance(params, opt, sched, 7)
    state = rawr_checkpoint.add_state(opt.state_dict(), sched, config, 7)
    state = copy.deepcopy(state)
    payload = state[rawr_checkpoint.KEY]

    params2 = _new_params()
    lr_a = LR_A
    d = 0.45
    trainer_step = 7
    if mutation == "d":
        d = 0.6
    elif mutation == "lr":
        lr_a = LR_A * 2.0
    elif mutation == "step":
        trainer_step = 8
    elif mutation == "version":
        payload["version"] = 2
    elif mutation == "corrupt":
        payload["scheduler"]["base_lrs"] = [99.0, 99.0]

    opt2 = _new_optimizer(params2, lr_a=lr_a, lr_b=LR_B)
    config2 = rawr_checkpoint.configuration(opt2, TOTAL_STEPS, _scheduler_params(d, 1.5))
    opt2.load_state_dict(state)
    captured = rawr_checkpoint.capture(state)
    for group in opt2.param_groups:
        group["initial_lr"] = group["lr"]
    sched2 = create_scheduler(opt2, "easy:RAWR", TOTAL_STEPS, _scheduler_params(d, 1.5))
    with pytest.raises(ValueError):
        rawr_checkpoint.restore(sched2, captured, config2, trainer_step)


def test_non_rawr_passthrough():
    params = _new_params()
    opt = _new_optimizer(params)
    sched = create_scheduler(opt, "cosine", TOTAL_STEPS, {"warmup_steps": 2})
    state = opt.state_dict()
    assert rawr_checkpoint.add_state(state, sched, None, 3) is state or \
        rawr_checkpoint.KEY not in rawr_checkpoint.add_state(state, sched, None, 3)
    captured = rawr_checkpoint.capture(state)
    assert captured["payload"] is None
    assert rawr_checkpoint.restore(sched, captured, None, 3) is False


def test_non_rawr_with_config_rejected():
    opt = _new_optimizer(_new_params())
    sched = create_scheduler(opt, "linear", TOTAL_STEPS, {})
    with pytest.raises(TypeError):
        rawr_checkpoint.add_state(opt.state_dict(), sched, {"x": 1}, 3)


def test_completed_step_helpers():
    assert rawr_checkpoint.completed_step(3, 4, {"rawr": True}) == 4
    assert rawr_checkpoint.completed_step(4, None, {"rawr": True}) == 4
    assert rawr_checkpoint.completed_step(4, 4, {"rawr": True}) == 4
    assert rawr_checkpoint.completed_step(3, None, None) == 3
    assert rawr_checkpoint.completed_step(3, 9, None) == 3


@pytest.mark.parametrize('corruption', ['missing_field', 'position', 'group_count',
                                      'scheduler_missing', 'scheduler_type', 'payload_type'])
def test_corrupt_checkpoint_rejected(corruption):
    params, opt, sched, config, _ = _make_state(_new_params(), .45, 1.5)
    _advance(params, opt, sched, 7)
    state = copy.deepcopy(rawr_checkpoint.add_state(opt.state_dict(), sched, config, 7))
    if corruption == 'missing_field':
        del state['param_groups'][0]['current_cycle_step']
    elif corruption == 'position':
        state['param_groups'][0]['current_cycle_step'] = -2
    elif corruption == 'group_count':
        state['param_groups'].pop()
    elif corruption == 'scheduler_missing':
        del state[rawr_checkpoint.KEY]['scheduler']
    elif corruption == 'scheduler_type':
        state[rawr_checkpoint.KEY]['scheduler'] = 'invalid'
    else:
        state[rawr_checkpoint.KEY] = None
    with pytest.raises(ValueError):
        captured = rawr_checkpoint.capture(state)
        rawr_checkpoint.restore(sched, captured, config, 7)


@pytest.mark.parametrize('change', [{'gamma': .8}, {'cycle_multiplier': 2},
                                    {'warmup_steps': 3}, {'first_cycle_max_steps': 20}])
def test_declared_schedule_change_rejected(change):
    params, opt, sched, config, _ = _make_state(_new_params(), .45, 1.5)
    state = rawr_checkpoint.add_state(opt.state_dict(), sched, config, 0)
    new_opt = _new_optimizer(_new_params())
    kwargs = dict(_scheduler_params(.45, 1.5), **change)
    new_config = rawr_checkpoint.configuration(new_opt, TOTAL_STEPS, kwargs)
    new_opt.load_state_dict(copy.deepcopy(state))
    new_sched = create_scheduler(new_opt, 'easy:RAWR', TOTAL_STEPS, kwargs)
    with pytest.raises(ValueError, match='configuration changed'):
        rawr_checkpoint.restore(new_sched, rawr_checkpoint.capture(state), new_config, 0)


def test_arbitrary_wrapper_cannot_silently_drop_rawr_state():
    _, opt, sched, config, _ = _make_state(_new_params(), .45, 1.5)
    class UnknownWrapper:
        def __init__(self, scheduler):
            self.scheduler = scheduler
    with pytest.raises(TypeError, match='unsupported scheduler wrapper'):
        rawr_checkpoint.add_state(opt.state_dict(), UnknownWrapper(sched), config, 0)
