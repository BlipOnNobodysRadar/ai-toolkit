import random
import tempfile
from datetime import datetime
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Union

import torch
import yaml

from jobs.process.BaseProcess import BaseProcess
from toolkit import save_root_lease
from toolkit.config import config_flag, config_for_backup

if TYPE_CHECKING:
    from jobs import TrainJob, BaseJob, ExtensionJob
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm


class BaseTrainProcess(BaseProcess):

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.process_id: int
        self.config: OrderedDict
        self.writer: 'SummaryWriter'
        self.job: Union['TrainJob', 'BaseJob', 'ExtensionJob']
        self.progress_bar: 'tqdm' = None

        self.training_seed = self.get_conf('training_seed', self.job.training_seed if hasattr(self.job, 'training_seed') else None)
        # if training seed is set, use it
        if self.training_seed is not None:
            torch.manual_seed(self.training_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.training_seed)
            random.seed(self.training_seed)

        self.progress_bar = None
        self.writer = None
        self.training_folder = self.get_conf('training_folder',
                                             self.job.training_folder if hasattr(self.job, 'training_folder') else None)
        self.save_root = os.path.join(self.training_folder, self.name)
        self.step = 0
        self.first_step = 0
        self.log_dir = self.get_conf('log_dir', self.job.log_dir if hasattr(self.job, 'log_dir') else None)
        self._refuse_distributed()
        # before anything is written to the folder, and before any resume reads it (docs/STATE_MACHINES.md,
        # OPUS-ONEWRITER-01). Owned by the job: BaseJob releases it if a later constructor fails.
        self._save_root_lease = save_root_lease.acquire(self.save_root, owner=self.job, holder=self)
        self.setup_tensorboard()
        self.save_training_config()

    def _refuse_distributed(self):
        # Data loaders are not sharded, resume metadata is read on the main rank only, per-step reseeding gives every
        # rank the same draws, and every rank would take this folder's lease (OPUS-DIST-01). Refuse before writing.
        from toolkit.accelerator import get_accelerator
        processes = get_accelerator().num_processes
        if processes > 1:
            raise ValueError(f"Training with {processes} processes (accelerate launch / torchrun) is not supported: "
                             f"each process would train on the same batches and write the same output folder. Launch "
                             f"one process")

    def release_save_root(self):
        """Give up this process's share of the save-root lease (idempotent): its run is over."""
        save_root_lease.release(getattr(self, '_save_root_lease', None), self)
        self._save_root_lease = None

    def run(self):
        super().run()
        # implement in child class
        # be sure to call super().run() first
        pass

    # def print(self, message, **kwargs):
    def print(self, *args):
        if self.progress_bar is not None:
            self.progress_bar.write(' '.join(map(str, args)))
            self.progress_bar.update()
        else:
            print(*args)

    def setup_tensorboard(self):
        if self.log_dir:
            from torch.utils.tensorboard import SummaryWriter
            now = datetime.now()
            time_str = now.strftime('%Y%m%d-%H%M%S')
            summary_name = f"{self.name}_{time_str}"
            summary_dir = os.path.join(self.log_dir, summary_name)
            self.writer = SummaryWriter(summary_dir)

    def save_training_config(self):
        os.makedirs(self.save_root, exist_ok=True)
        if not config_flag(self.get_conf('logging.save_config', False)):
            return
        save_dif = os.path.join(self.save_root, 'config.yaml')
        try:
            snapshot = config_for_backup(self.job.raw_config)
        except yaml.YAMLError:
            self.print("Skipped local config snapshot because the unexpanded template could not be parsed.")
            return
        fd, temporary = tempfile.mkstemp(dir=self.save_root)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                yaml.safe_dump(snapshot, f, sort_keys=False)
            os.replace(temporary, save_dif)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
