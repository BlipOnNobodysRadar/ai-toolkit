"""Whole-epoch accumulation (gradient_accumulation_steps: -1) windows (docs/STATE_MACHINES.md M1, C13).

The real BaseSDTrainProcess._fetch_batch_list (the loop's batch fetch), _setup_accumulation_iteration, _run_train_step
and _end_loop_iteration on the real SDTrainer fixture of test_accumulation_window, over real torch DataLoaders (main
and regularization, no shuffle). Each batch is a target vector whose first element identifies it; the optimizer's step
pre-hook records which batches the update consumed. CPU state-machine evidence only: no diffusion model or GPU.
"""
import pytest
import torch
from torch.utils.data import DataLoader

from test_accumulation_window import cpu_only, trainer  # noqa: F401 (cpu_only is an autouse fixture)


def loader(ids, batch_size=1):
    return DataLoader([torch.tensor([float(i), 0.], dtype=torch.float64) for i in ids], batch_size=batch_size)


def run(tmp_path, main, steps, B=1, reg=None, oom_at=()):
    f, p, raw = trainer(tmp_path, gas=-1, rawr=False, optimizer='sgd')
    f.train_config.gradient_accumulation = B
    f.train_config.steps = steps
    f.training_seed = None
    f.data_loader = main
    f._data_epoch = f.epoch_num = 0
    assert f._main_loader_has_batches()
    f._epoch_cursor = 0
    consumed, windows = [], []
    real_train = f.train_single_accumulation

    def train_one(batch, accum_scale=1.0):
        consumed.append(int(batch.flatten()[0]))
        return real_train(batch, accum_scale)
    f.train_single_accumulation = train_one
    def record(*args):  # a step pre-hook must return None
        windows.append((f.step_num, list(consumed)))
        consumed.clear()
    raw.register_step_pre_hook(record)
    it, it_reg = iter(main), (iter(reg) if reg is not None else None)
    for step in range(steps):
        f.step_num = step
        f.is_grad_accumulation_step = True
        batches, it, it_reg = f._fetch_batch_list(step, main, it, reg, it_reg, False, False)
        f._setup_accumulation_iteration()
        f.oom_plan = {step: 'after_backward'} if step in oom_at else {}
        _, did_oom = f._run_train_step(step, batches, raw)
        if did_oom:
            consumed.clear()  # the window was discarded
        f._end_loop_iteration(step)
    return windows, f


def test_each_update_is_exactly_one_epoch(tmp_path):
    windows, f = run(tmp_path, loader(range(4)), steps=12)
    # the original took e0b0..e0b3 + e1b0 in the first update and shifted every later window by one batch
    assert windows == [(3, [0, 1, 2, 3]), (7, [0, 1, 2, 3]), (11, [0, 1, 2, 3])]
    assert f.epoch_num == 2  # the third epoch began at step 8


def test_a_run_of_exactly_one_epoch_updates_once(tmp_path):
    windows, _ = run(tmp_path, loader(range(4)), steps=4)
    assert windows == [(3, [0, 1, 2, 3])]  # the original made no update at all


def test_one_batch_epochs_update_every_step(tmp_path):
    windows, _ = run(tmp_path, loader([0]), steps=3)
    assert windows == [(0, [0]), (1, [0]), (2, [0])]


def test_the_final_partial_window_is_applied(tmp_path):
    windows, _ = run(tmp_path, loader(range(4)), steps=6)
    assert windows == [(3, [0, 1, 2, 3]), (5, [0, 1])]


def test_whole_epoch_accumulation_cannot_be_combined_with_batch_lists():
    # the supported contract: TrainConfig refuses gradient_accumulation > 1 with any gradient_accumulation_steps != 1,
    # -1 included, so whole-epoch windows are always one batch per iteration
    from toolkit.config_modules import TrainConfig
    with pytest.raises(ValueError, match='mutually exclusive'):
        TrainConfig(gradient_accumulation_steps=-1, gradient_accumulation=2)
    assert TrainConfig(gradient_accumulation_steps=-1).gradient_accumulation == 1


def test_regularization_batches_belong_to_the_window_they_are_fetched_in(tmp_path):
    reg = loader([100, 101])
    windows, _ = run(tmp_path, loader(range(4)), steps=8, reg=reg)
    # even steps fetch a regularization batch, odd steps a main one; the epoch ends with main batch 3 at step 7
    assert windows == [(7, [100, 0, 101, 1, 100, 2, 101, 3])]


def test_an_oom_discards_the_window_and_the_rest_of_the_epoch_is_a_shorter_one(tmp_path):
    windows, _ = run(tmp_path, loader(range(4)), steps=8, oom_at=(1,))
    assert windows == [(3, [2, 3]), (7, [0, 1, 2, 3])]


def test_whole_epoch_needs_a_sized_nonempty_main_loader(tmp_path):
    f, _, _ = trainer(tmp_path, gas=-1, rawr=False, optimizer='sgd')
    f.data_loader = loader([])
    assert not f._main_loader_has_batches()
    f.data_loader = None
    assert not f._main_loader_has_batches()

    class Unsized(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter([torch.zeros(2)])
    f.data_loader = DataLoader(Unsized())
    assert not f._main_loader_has_batches()


def test_a_loader_shorter_than_its_length_is_refused(tmp_path):
    class Short(DataLoader):
        def __len__(self):
            return 5  # yields only 4
    with pytest.raises(RuntimeError, match='needs an accurate length'):
        run(tmp_path, Short([torch.tensor([float(i), 0.], dtype=torch.float64) for i in range(4)]), steps=8)


def test_fixed_size_accumulation_is_unchanged_across_epochs(tmp_path):
    f, p, raw = trainer(tmp_path, gas=2, rawr=False, optimizer='sgd')
    f.train_config.steps = 6
    f.data_loader = main = loader(range(3))
    f._data_epoch = f.epoch_num = 0
    f._epoch_cursor = 0
    f.training_seed = None
    seen, windows = [], []
    real_train = f.train_single_accumulation
    def train_one(batch, accum_scale=1.0):
        seen.append(int(batch.flatten()[0]))
        return real_train(batch, accum_scale)
    f.train_single_accumulation = train_one

    def record(*args):
        windows.append(list(seen))
        seen.clear()
    raw.register_step_pre_hook(record)
    it = iter(main)
    for step in range(6):
        f.step_num = step
        f.is_grad_accumulation_step = True
        batches, it, _ = f._fetch_batch_list(step, main, it, None, None, False, False)
        f._setup_accumulation_iteration()
        f._run_train_step(step, batches, raw)
        f._end_loop_iteration(step)
    assert windows == [[0, 1], [2, 0], [1, 2]]  # windows of 2 regardless of epoch ends, as before


# ---------------------------------------------------------------- update magnitude (Codex epoch_weighting_probe)

def start():
    return torch.tensor([.25, -.5], dtype=torch.float64)


def target(i):
    return torch.tensor([float(i), 0.], dtype=torch.float64)


def test_each_whole_epoch_update_sums_its_epoch(tmp_path):
    windows, f = run(tmp_path, loader(range(4)), steps=4)
    (step, grad), = f.seen_updates
    expected = sum(start() - target(i) for i in range(4))  # d/dx mean((x - t)^2) over [1, 2] = (x - t); B = 1: summed
    assert step == 3 and torch.allclose(grad, expected, rtol=0, atol=1e-12)


def test_the_final_partial_window_is_summed_like_a_full_one(tmp_path):
    windows, f = run(tmp_path, loader(range(4)), steps=6)
    (_, first), (_, final) = f.seen_updates
    assert torch.allclose(first, sum(start() - target(i) for i in range(4)), rtol=0, atol=1e-12)
    after_first = start() - first  # SGD lr 1
    assert torch.allclose(final, sum(after_first - target(i) for i in range(2)), rtol=0, atol=1e-12)


def test_every_fetched_batch_is_cleaned_up(tmp_path):
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
    f, _, _ = trainer(tmp_path, gas=1, rawr=False, optimizer='sgd')
    cleaned = []

    def dto(i):
        d = DataLoaderBatchDTO.__new__(DataLoaderBatchDTO)
        d.cleanup = lambda: cleaned.append(i)
        return d
    f._cleanup_batch_list([dto(0), dto(1), None, dto(2)])
    assert cleaned == [0, 1, 2]  # the loop used to clean only the last batch it fetched
