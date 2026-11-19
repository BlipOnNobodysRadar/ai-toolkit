"""Interleave complete preference optimizer windows with the ordinary image loader."""
from typing import Optional


class FeedbackSchedule:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.train_config.dpo_feedback
        self.updates = 0
        self.preference_updates = 0
        self.pairs_used = 0
        # None starts a new window. An empty plan pins an ordinary window even
        # if a vote arrives between its microbatches.
        self.plan: Optional[list] = None
        self.offset = 0
        self.saved_position = None
        saved = getattr(trainer, '_saved_feedback_schedule', None)
        if saved is not None:
            self._restore(saved)

    def _restore(self, saved):
        if not isinstance(saved, dict) or saved.get('version') != 1:
            raise ValueError('Unsupported preference feedback checkpoint state')
        for name in ('updates', 'preference_updates', 'pairs_used'):
            value = saved.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f'Invalid preference feedback checkpoint {name}')
            setattr(self, name, value)
        if self.preference_updates > self.updates:
            raise ValueError('Invalid preference feedback update counts')
        position = saved.get('data_position')
        if not isinstance(position, list) or len(position) != 2 or any(type(x) is not int or x < 0 for x in position):
            raise ValueError('Invalid preference feedback data position')
        self.saved_position = position
        # Only RAWR can restore the gradients that belong to an unfinished
        # window. Ordinary checkpoints resume with fresh gradients.
        if getattr(self.trainer, '_rawr_scheduler_restored', False) and saved.get('plan') is not None:
            plan, offset = saved['plan'], saved.get('offset')
            if not isinstance(plan, list) or not all(isinstance(x, str) for x in plan):
                raise ValueError('Invalid preference feedback checkpoint plan')
            if type(offset) is not int or not 0 <= offset <= len(plan):
                raise ValueError('Invalid preference feedback checkpoint offset')
            if saved.get('window_shape') != self._window_shape():
                raise ValueError('Cannot change feedback batch or world size inside a restored accumulation window')
            self.plan, self.offset = plan, offset

    def _window_shape(self):
        train = self.trainer.train_config
        return [train.batch_size, train.gradient_accumulation, train.gradient_accumulation_steps,
                self.trainer.accelerator.num_processes]

    def state_dict(self):
        return {
            'version': 1, 'updates': self.updates, 'preference_updates': self.preference_updates,
            'pairs_used': self.pairs_used, 'plan': self.plan, 'offset': self.offset,
            'window_shape': self._window_shape(),
            'data_position': [self.trainer._data_epoch, self.trainer._epoch_cursor],
        }

    def _start_window(self):
        train = self.trainer.train_config
        since_start = self.updates + 1 - self.config['start_after']
        if since_start <= 0 or since_start % self.config['every']:
            return []
        from toolkit.feedback_training import voted_feedback_ids

        accelerator = self.trainer.accelerator
        ids = voted_feedback_ids(self.trainer) if accelerator.is_main_process else []
        # All ranks must select the same objective, including when a vote arrives
        # between reads or one rank gets to this window first.
        if accelerator.num_processes > 1:
            from accelerate.utils import broadcast_object_list
            ids = broadcast_object_list([ids], from_process=0)[0]
        if not ids:
            return []
        count = train.batch_size * train.gradient_accumulation * max(1, train.gradient_accumulation_steps)
        count *= accelerator.num_processes
        return [ids[(self.pairs_used + i) % len(ids)] for i in range(count)]

    def batches(self):
        """Return one host iteration's preference microbatches, or None for the ordinary loader."""
        if self.plan is None:
            self.plan = self._start_window()
        if not self.plan:
            return None
        from toolkit.feedback_training import FeedbackError, load_feedback_batch

        train = self.trainer.train_config
        accel = self.trainer.accelerator
        count = train.batch_size * train.gradient_accumulation * accel.num_processes
        block = self.plan[self.offset:self.offset + count]
        if len(block) != count:
            raise ValueError('Preference feedback window and gradient accumulation are out of sync')
        batches = []
        for pair_id in block[accel.process_index::accel.num_processes]:
            try:
                batch = load_feedback_batch(self.trainer, pair_id)
            except FeedbackError as exc:
                raise RuntimeError(
                    f"Could not use voted pair {pair_id}: {exc}. Check the archived decision, "
                    "original reference model and pair files before resuming this window."
                ) from exc
            batch._feedback_pair_id = pair_id
            batches.append(batch)
        self.offset += count
        return batches

    def finish_iteration(self, did_oom):
        trainer = self.trainer
        if not did_oom and trainer.is_grad_accumulation_step:
            return
        if not did_oom and not trainer.accelerator.optimizer_step_was_skipped:
            self.updates += 1
            if self.plan:
                self.preference_updates += 1
                self.pairs_used += self.offset
        trainer.additional_logs['feedback/updates'] = self.preference_updates
        trainer.additional_logs['feedback/optimizer_updates'] = self.updates
        self.plan, self.offset = None, 0
