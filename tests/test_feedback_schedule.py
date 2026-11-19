"""Focused CPU contracts for toolkit.feedback_schedule.

Production code is imported directly.  Trainer/accelerator doubles are thin,
explicit and typed; pair selection and batch loading are the only seams.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import uuid
from types import SimpleNamespace

import pytest

from toolkit import feedback_training
from toolkit.config_modules import TrainConfig
from toolkit.feedback_schedule import FeedbackSchedule


@dataclasses.dataclass
class FakeAccelerator:
    num_processes: int = 1
    process_index: int = 0
    is_main_process: bool = True
    optimizer_step_was_skipped: bool = False


@dataclasses.dataclass
class FakeTrainConfig:
    dpo_feedback: dict | None = None
    batch_size: int = 1
    gradient_accumulation: int = 1
    gradient_accumulation_steps: int = 1
    steps: int = 1000


class FakeTrainer:
    def __init__(
        self,
        *,
        every: int = 1,
        start_after: int = 0,
        batch_size: int = 1,
        gradient_accumulation: int = 1,
        gradient_accumulation_steps: int = 1,
        num_processes: int = 1,
        process_index: int = 0,
    ) -> None:
        self.train_config = FakeTrainConfig(
            dpo_feedback={"every": every, "start_after": start_after},
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        self.accelerator = FakeAccelerator(
            num_processes=num_processes,
            process_index=process_index,
        )
        self.additional_logs: dict[str, int] = {}
        self.is_grad_accumulation_step = True
        self._data_epoch = 0
        self._epoch_cursor = 0


@dataclasses.dataclass
class BatchDouble:
    pair_id: str


def install_feedback_doubles(monkeypatch, ids):
    """Patch the pair boundary, keeping schedule arithmetic production-real."""
    state = {"ids": list(ids)}
    vote_calls: list[list[str]] = []
    loaded: list[str] = []

    def voted_feedback_ids(_trainer):
        vote_calls.append(list(state["ids"]))
        return list(state["ids"])

    def load_feedback_batch(_trainer, pair_id):
        loaded.append(pair_id)
        return BatchDouble(pair_id)

    monkeypatch.setattr(feedback_training, "voted_feedback_ids", voted_feedback_ids)
    monkeypatch.setattr(feedback_training, "load_feedback_batch", load_feedback_batch)
    return SimpleNamespace(
        vote_calls=vote_calls,
        loaded=loaded,
        set_ids=lambda values: state.__setitem__("ids", list(values)),
        batches=lambda batch_list: [batch.pair_id for batch in batch_list],
    )


def finish_real_update(trainer, schedule, *, did_oom=False):
    trainer.is_grad_accumulation_step = False
    schedule.finish_iteration(did_oom)


# ---------------------------------------------------------------- configuration/dispatch


def test_train_config_normalizes_feedback_and_keeps_ordinary_mode_unchanged():
    ordinary = TrainConfig(loss_type="mse")
    assert ordinary.loss_type == "mse"
    assert ordinary.dpo_feedback is None

    hybrid = TrainConfig(loss_type="mse", dpo_feedback={"every": 4})
    assert hybrid.dpo_feedback == {"every": 4, "start_after": 0}

    with pytest.raises(ValueError, match="dpo_feedback"):
        TrainConfig(dpo_feedback={"every": 0})
    with pytest.raises(ValueError, match="dpo_feedback"):
        TrainConfig(dpo_feedback={"every": 10, "start_after": -1})
    with pytest.raises(ValueError, match="dpo_feedback"):
        TrainConfig(dpo_feedback={"every": 10, "extra": 1})
    with pytest.raises(ValueError, match="ordinary loss_type"):
        TrainConfig(loss_type="dpo", dpo_feedback={"every": 10})
    with pytest.raises(ValueError, match="dpo_beta"):
        TrainConfig(dpo_feedback={"every": 10}, dpo_beta=0)


def test_actual_diffusion_dpo_validation_dispatches_normal_and_feedback_modes():
    from toolkit.diffusion_dpo import validate_dpo_config

    def trainer(*, loss_type="mse", feedback=None, arch="moss_image"):
        return SimpleNamespace(
            train_config=TrainConfig(
                loss_type=loss_type,
                dpo_feedback=feedback,
                noise_scheduler="flowmatch",
                loss_target="noise",
            ),
            dataset_configs=[SimpleNamespace(
                rejected_path=None, is_reg=False, cache_text_embeddings=True,
            )],
            model_config=SimpleNamespace(arch=arch, model_kwargs={}, refiner_name_or_path=None),
            config={},
            network_config=SimpleNamespace(type="lora"),
            train_unet=True,
            train_text_encoder=False,
            adapter_config=None,
            embed_config=None,
            decorator_config=None,
            guidance_config=None,
            is_caching_text_embeddings=False,
        )

    # Ordinary training exits before MOSS/DPO-only requirements.
    assert validate_dpo_config(trainer(arch="sdxl")) is None
    # Hybrid feedback uses the DPO guard but still accepts an ordinary loss and
    # the ordinary chosen-only target dataset.
    assert validate_dpo_config(trainer(feedback={"every": 10})) is None
    with pytest.raises(ValueError, match="MOSS-Image"):
        validate_dpo_config(trainer(feedback={"every": 10}, arch="sdxl"))
    with pytest.raises(ValueError, match="rejected_path"):
        validate_dpo_config(trainer(loss_type="dpo"))


def test_no_votes_fall_back_to_ordinary_and_pin_that_window_across_microbatches(monkeypatch):
    trainer = FakeTrainer(every=1, gradient_accumulation_steps=2)
    doubles = install_feedback_doubles(monkeypatch, [])
    schedule = FeedbackSchedule(trainer)

    assert schedule.batches() is None
    doubles.set_ids(["pair-a"])  # a vote arriving between microbatches must not change this window
    assert schedule.batches() is None
    assert doubles.vote_calls == [[]]

    trainer.is_grad_accumulation_step = True
    schedule.finish_iteration(False)  # first microbatch, no optimizer update
    assert schedule.updates == 0 and schedule.preference_updates == 0
    finish_real_update(trainer, schedule)
    assert schedule.updates == 1 and schedule.preference_updates == 0
    assert trainer.additional_logs == {"feedback/updates": 0, "feedback/optimizer_updates": 1}

    batch_list = schedule.batches()
    assert doubles.batches(batch_list) == ["pair-a"]
    assert len(doubles.vote_calls) == 2


def test_new_votes_are_deferred_and_skipped_rows_disappear_next_window(monkeypatch):
    trainer = FakeTrainer(every=1, gradient_accumulation=2, gradient_accumulation_steps=2)
    doubles = install_feedback_doubles(monkeypatch, ["pair-a", "pair-b"])
    schedule = FeedbackSchedule(trainer)

    first = doubles.batches(schedule.batches())
    doubles.set_ids(["pair-b", "pair-c"])  # pair-a becomes skip; pair-c is newly voted
    second = doubles.batches(schedule.batches())
    assert first + second == ["pair-a", "pair-b", "pair-a", "pair-b"]
    assert doubles.vote_calls == [["pair-a", "pair-b"]]
    finish_real_update(trainer, schedule)
    assert schedule.preference_updates == 1 and schedule.pairs_used == 4

    next_window = doubles.batches(schedule.batches())
    assert next_window == ["pair-b", "pair-c"]
    assert doubles.vote_calls[-1] == ["pair-b", "pair-c"]


def test_deterministic_pair_sequence_rotates_by_completed_pairs(monkeypatch):
    trainer = FakeTrainer(every=1)
    doubles = install_feedback_doubles(monkeypatch, ["pair-a", "pair-b", "pair-c"])
    schedule = FeedbackSchedule(trainer)

    sequence = []
    for _ in range(4):
        sequence.extend(doubles.batches(schedule.batches()))
        finish_real_update(trainer, schedule)
    assert sequence == ["pair-a", "pair-b", "pair-c", "pair-a"]
    assert schedule.pairs_used == 4


# ---------------------------------------------------------------- real optimizer window arithmetic


def test_window_plan_uses_batch_size_gradient_accumulation_and_ranks(monkeypatch):
    trainer = FakeTrainer(
        every=1, batch_size=2, gradient_accumulation=4,
        gradient_accumulation_steps=3, num_processes=2, process_index=1,
    )
    ids = [f"pair-{i:02d}" for i in range(48)]
    doubles = install_feedback_doubles(monkeypatch, ids)
    schedule = FeedbackSchedule(trainer)

    loaded = []
    for iteration in range(3):
        batch_list = schedule.batches()
        assert len(batch_list) == 8
        loaded.extend(doubles.batches(batch_list))
        trainer.is_grad_accumulation_step = iteration < 2
        schedule.finish_iteration(False)
    assert len(loaded) == 24
    assert schedule.updates == 1 and schedule.preference_updates == 1
    assert schedule.pairs_used == 48
    assert schedule.plan is None and schedule.offset == 0


def test_whole_epoch_accumulation_uses_one_feedback_host_iteration(monkeypatch):
    trainer = FakeTrainer(every=1, batch_size=2, gradient_accumulation=3,
                          gradient_accumulation_steps=-1)
    ids = [f"pair-{i:02d}" for i in range(6)]
    doubles = install_feedback_doubles(monkeypatch, ids)
    schedule = FeedbackSchedule(trainer)

    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

    batch_list = schedule.batches()
    assert doubles.batches(batch_list) == ids
    # run() sets these for a whole-epoch feedback window, then the real setup
    # method leaves the one host iteration marked as an update.
    trainer.is_grad_accumulation_step = False
    run_state = SimpleNamespace(
        train_config=trainer.train_config, step_num=0,
        grad_accumulation_step=0, is_grad_accumulation_step=False,
    )
    BaseSDTrainProcess._setup_accumulation_iteration(run_state)
    assert run_state.is_grad_accumulation_step is False
    trainer.is_grad_accumulation_step = run_state.is_grad_accumulation_step
    schedule.finish_iteration(False)
    assert schedule.updates == 1 and schedule.preference_updates == 1
    assert schedule.pairs_used == 6


# ---------------------------------------------------------------- OOM / AMP / exceptions


def test_oom_resets_the_whole_window_without_counting_an_update(monkeypatch):
    trainer = FakeTrainer(every=1, gradient_accumulation_steps=2)
    doubles = install_feedback_doubles(monkeypatch, ["pair-a", "pair-b"])
    schedule = FeedbackSchedule(trainer)

    assert doubles.batches(schedule.batches()) == ["pair-a"]
    assert schedule.offset == 1
    schedule.finish_iteration(True)  # OOM anywhere in the window discards it
    assert schedule.updates == 0 and schedule.preference_updates == 0
    assert schedule.pairs_used == 0
    assert schedule.plan is None and schedule.offset == 0
    assert trainer.additional_logs == {"feedback/updates": 0, "feedback/optimizer_updates": 0}

    assert doubles.batches(schedule.batches()) == ["pair-a"]
    assert doubles.batches(schedule.batches()) == ["pair-b"]


def test_amp_skipped_optimizer_update_is_not_counted_and_window_retries(monkeypatch):
    trainer = FakeTrainer(every=1)
    doubles = install_feedback_doubles(monkeypatch, ["pair-a"])
    schedule = FeedbackSchedule(trainer)

    assert doubles.batches(schedule.batches()) == ["pair-a"]
    trainer.accelerator.optimizer_step_was_skipped = True
    finish_real_update(trainer, schedule)
    assert schedule.updates == 0 and schedule.preference_updates == 0
    assert schedule.pairs_used == 0
    assert trainer.additional_logs == {"feedback/updates": 0, "feedback/optimizer_updates": 0}

    trainer.accelerator.optimizer_step_was_skipped = False
    assert doubles.batches(schedule.batches()) == ["pair-a"]
    finish_real_update(trainer, schedule)
    assert schedule.updates == 1 and schedule.preference_updates == 1


def test_vote_poll_exception_propagates_without_starting_a_window(monkeypatch):
    trainer = FakeTrainer(every=1)

    def voted_feedback_ids(_trainer):
        raise RuntimeError("injected poll failure")

    monkeypatch.setattr(feedback_training, "voted_feedback_ids", voted_feedback_ids)
    schedule = FeedbackSchedule(trainer)
    with pytest.raises(RuntimeError, match="injected poll failure"):
        schedule.batches()
    assert schedule.plan is None and schedule.offset == 0
    assert schedule.updates == 0 and schedule.preference_updates == 0


def test_loader_exception_propagates_and_leaves_window_retryable(monkeypatch):
    trainer = FakeTrainer(every=1)
    ids = ["pair-a"]
    state = {"fail": True}
    vote_calls = []

    def voted_feedback_ids(_trainer):
        vote_calls.append(list(ids))
        return list(ids)

    def load_feedback_batch(_trainer, pair_id):
        if state["fail"]:
            raise RuntimeError("injected loader failure")
        return BatchDouble(pair_id)

    monkeypatch.setattr(feedback_training, "voted_feedback_ids", voted_feedback_ids)
    monkeypatch.setattr(feedback_training, "load_feedback_batch", load_feedback_batch)
    schedule = FeedbackSchedule(trainer)

    with pytest.raises(RuntimeError, match="injected loader failure"):
        schedule.batches()
    assert schedule.plan == ["pair-a"] and schedule.offset == 0
    assert schedule.updates == 0 and schedule.preference_updates == 0

    state["fail"] = False
    batch_list = schedule.batches()
    assert [batch.pair_id for batch in batch_list] == ["pair-a"]
    assert len(vote_calls) == 1  # no re-poll: the same planned window is retried


# ---------------------------------------------------------------- delayed / skipped rows at the real boundary


def _write_pair(pair_root, *, pair_id, reference_id, step, choice, write_vote=True):
    pair_dir = pair_root / pair_id
    pair_dir.mkdir(parents=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    (pair_dir / "pair.json").write_text(json.dumps({
        "version": 1,
        "id": pair_id,
        "step": step,
        "prompt": "old delayed row",
        "seeds": [1, 2],
        "width": 32,
        "height": 24,
        "reference_id": reference_id,
        "created_at": now,
    }), encoding="utf-8")
    if write_vote:
        (pair_dir / "vote.json").write_text(json.dumps({
            "version": 1, "id": pair_id, "choice": choice, "voted_at": now,
        }), encoding="utf-8")
    for name in ("a.png", "b.png", "embeddings.safetensors", "latents.safetensors"):
        (pair_dir / name).write_bytes(b"x")


def test_real_poll_keeps_old_step_rows_eligible_and_excludes_skip_and_pending(tmp_path, monkeypatch):
    reference_id = "a" * 64
    monkeypatch.setattr(feedback_training, "feedback_reference_id", lambda _trainer: reference_id)
    trainer = SimpleNamespace(save_root=str(tmp_path / "job"))
    root = tmp_path / "job" / "preferences"
    root.mkdir(parents=True)
    eligible = uuid.uuid4().hex
    skipped = uuid.uuid4().hex
    pending = uuid.uuid4().hex
    _write_pair(root, pair_id=eligible, reference_id=reference_id, step=1, choice="a")
    _write_pair(root, pair_id=skipped, reference_id=reference_id, step=1, choice="skip")
    _write_pair(root, pair_id=pending, reference_id=reference_id, step=1, choice="a", write_vote=False)

    assert feedback_training.voted_feedback_ids(trainer) == [eligible]


def test_real_poll_without_preferences_is_an_ordinary_offline_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_training, "feedback_reference_id", lambda _trainer: "b" * 64)
    trainer = SimpleNamespace(save_root=str(tmp_path / "job"))
    assert feedback_training.voted_feedback_ids(trainer) == []

# ---------------------------------------------------------------- schedule cadence and resume


def test_nine_ordinary_updates_then_one_feedback_update(monkeypatch):
    trainer = FakeTrainer(every=10, start_after=0)
    doubles = install_feedback_doubles(monkeypatch, ["pair-a"])
    schedule = FeedbackSchedule(trainer)

    kinds = []
    for _ in range(20):
        batch_list = schedule.batches()
        kinds.append("feedback" if batch_list is not None else "ordinary")
        finish_real_update(trainer, schedule)

    assert kinds[:9] == ["ordinary"] * 9
    assert kinds[9] == "feedback"
    assert kinds[10:19] == ["ordinary"] * 9
    assert kinds[19] == "feedback"
    assert [i + 1 for i, kind in enumerate(kinds) if kind == "feedback"] == [10, 20]
    assert schedule.updates == 20
    assert schedule.preference_updates == 2
    assert schedule.pairs_used == 2
    assert len(doubles.loaded) == 2


def _mid_window_state(monkeypatch, **trainer_kwargs):
    trainer = FakeTrainer(**trainer_kwargs)
    ids = [f"pair-{i}" for i in range(6)]
    doubles = install_feedback_doubles(monkeypatch, ids)
    schedule = FeedbackSchedule(trainer)
    assert [batch.pair_id for batch in schedule.batches()] == (
        schedule.plan[0:trainer.train_config.batch_size * trainer.train_config.gradient_accumulation]
    )
    trainer._data_epoch = 4
    trainer._epoch_cursor = 17
    state = schedule.state_dict()
    return trainer, ids, doubles, schedule, state


def test_rawr_resume_continues_the_saved_window_and_saved_data_position(monkeypatch):
    _, ids, _, schedule, state = _mid_window_state(
        monkeypatch, every=1, batch_size=2, gradient_accumulation=1, gradient_accumulation_steps=3,
    )
    assert state["plan"] == [ids[(i % len(ids))] for i in range(6)]
    assert state["offset"] == 2
    assert state["data_position"] == [4, 17]

    restored_trainer = FakeTrainer(
        every=1, batch_size=2, gradient_accumulation=1, gradient_accumulation_steps=3,
    )
    restored_trainer._rawr_scheduler_restored = True
    restored_trainer._saved_feedback_schedule = state
    restored_doubles = install_feedback_doubles(monkeypatch, ids)
    restored = FeedbackSchedule(restored_trainer)

    assert restored.saved_position == [4, 17]
    assert restored.plan == state["plan"] and restored.offset == 2
    assert restored_doubles.vote_calls == []  # an unfinished RAWR window never re-polls
    assert restored_doubles.batches(restored.batches()) == state["plan"][2:4]
    assert restored.offset == 4
    restored_trainer.is_grad_accumulation_step = True
    restored.finish_iteration(False)
    assert restored_doubles.batches(restored.batches()) == state["plan"][4:6]
    restored_trainer.is_grad_accumulation_step = False
    restored.finish_iteration(False)
    assert restored.updates == 1 and restored.preference_updates == 1
    assert restored.pairs_used == 6


def test_ordinary_resume_discards_unfinished_window_but_keeps_counters_and_position(monkeypatch):
    _, ids, _, schedule, state = _mid_window_state(
        monkeypatch, every=1, batch_size=2, gradient_accumulation=1, gradient_accumulation_steps=3,
    )
    restored_trainer = FakeTrainer(
        every=1, batch_size=2, gradient_accumulation=1, gradient_accumulation_steps=3,
    )
    restored_trainer._saved_feedback_schedule = state
    restored_doubles = install_feedback_doubles(monkeypatch, ids)
    restored = FeedbackSchedule(restored_trainer)

    assert restored.saved_position == [4, 17]
    assert restored.plan is None and restored.offset == 0
    assert restored.updates == schedule.updates == 0
    # Ordinary resume has no saved gradients, so it starts a fresh window from the primary position.
    assert restored_doubles.batches(restored.batches()) == state["plan"][0:2]
    assert restored_doubles.vote_calls == [ids]


def test_resume_preserves_completed_rotation_and_primary_position(monkeypatch):
    trainer = FakeTrainer(every=1, batch_size=1, gradient_accumulation=1)
    ids = ["pair-a", "pair-b", "pair-c"]
    doubles = install_feedback_doubles(monkeypatch, ids)
    schedule = FeedbackSchedule(trainer)
    assert doubles.batches(schedule.batches()) == ["pair-a"]
    finish_real_update(trainer, schedule)
    trainer._data_epoch, trainer._epoch_cursor = 9, 3
    state = schedule.state_dict()
    assert state["data_position"] == [9, 3]
    assert state["updates"] == state["preference_updates"] == state["pairs_used"] == 1

    restored_trainer = FakeTrainer(every=1, batch_size=1, gradient_accumulation=1)
    restored_trainer._saved_feedback_schedule = state
    restored_doubles = install_feedback_doubles(monkeypatch, ids)
    restored = FeedbackSchedule(restored_trainer)
    assert restored.saved_position == [9, 3]
    assert restored.pairs_used == 1
    assert restored_doubles.batches(restored.batches()) == ["pair-b"]


@pytest.mark.parametrize("mutation, message", [
    (lambda state: state.update(version=2), "Unsupported"),
    (lambda state: state.update(updates=True), "Invalid preference feedback checkpoint updates"),
    (lambda state: state.update(preference_updates=2), "Invalid preference feedback update counts"),
    (lambda state: state.update(data_position=[1]), "Invalid preference feedback data position"),
    (lambda state: state.update(plan=["pair-a"], offset=2), "Invalid preference feedback checkpoint offset"),
    (lambda state: state.update(plan=[1], offset=0), "Invalid preference feedback checkpoint plan"),
])
def test_invalid_saved_state_is_rejected(monkeypatch, mutation, message):
    _, _, _, _, state = _mid_window_state(
        monkeypatch, every=1, batch_size=2, gradient_accumulation=1, gradient_accumulation_steps=3,
    )
    mutation(state)
    trainer = FakeTrainer(every=1, batch_size=2, gradient_accumulation=1, gradient_accumulation_steps=3)
    trainer._rawr_scheduler_restored = True
    trainer._saved_feedback_schedule = state
    with pytest.raises(ValueError, match=message):
        FeedbackSchedule(trainer)


def test_rawr_resume_refuses_a_changed_accumulation_window_shape(monkeypatch):
    _, _, _, _, state = _mid_window_state(
        monkeypatch, every=1, batch_size=2, gradient_accumulation=1, gradient_accumulation_steps=3,
    )
    trainer = FakeTrainer(every=1, batch_size=2, gradient_accumulation=1, gradient_accumulation_steps=2)
    trainer._rawr_scheduler_restored = True
    trainer._saved_feedback_schedule = state
    with pytest.raises(ValueError, match="Cannot change feedback batch or world size"):
        FeedbackSchedule(trainer)


# ---------------------------------------------------------------- actual BaseSDTrainProcess call order


def test_base_process_accumulation_setup_and_training_info_integration(monkeypatch):
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

    trainer = FakeTrainer(
        every=1, batch_size=2, gradient_accumulation=2, gradient_accumulation_steps=3,
    )
    ids = [f"pair-{i:02d}" for i in range(12)]
    doubles = install_feedback_doubles(monkeypatch, ids)
    schedule = FeedbackSchedule(trainer)
    # run()'s per-iteration state before BaseSDTrainProcess._setup_accumulation_iteration.
    run_state = SimpleNamespace(
        train_config=trainer.train_config, step_num=0,
        grad_accumulation_step=1, is_grad_accumulation_step=True,
    )

    for iteration in range(3):
        batch_list = schedule.batches()
        assert len(batch_list) == 4
        BaseSDTrainProcess._setup_accumulation_iteration(run_state)
        trainer.is_grad_accumulation_step = run_state.is_grad_accumulation_step
        schedule.finish_iteration(False)
        BaseSDTrainProcess._end_loop_iteration(run_state, iteration)
        if iteration < 2:
            assert run_state.is_grad_accumulation_step is True
            assert schedule.updates == schedule.preference_updates == 0
    assert schedule.updates == 1 and schedule.preference_updates == 1
    assert schedule.pairs_used == 12
    assert len(doubles.loaded) == 12

    info = BaseSDTrainProcess.get_training_info(SimpleNamespace(
        step_num=3, epoch_num=0, _clean_share_cursor=None, _feedback_schedule=schedule,
    ))
    assert info["dpo_feedback"] == schedule.state_dict()
    plain = BaseSDTrainProcess.get_training_info(SimpleNamespace(
        step_num=3, epoch_num=0, _clean_share_cursor=None, _feedback_schedule=None,
    ))
    assert "dpo_feedback" not in plain