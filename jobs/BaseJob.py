import importlib
import os
from collections import OrderedDict
from typing import List

from jobs.process import BaseProcess
from toolkit import save_root_lease


class BaseJob:

    def __init__(self, config: OrderedDict):
        if not config:
            raise ValueError('config is required')
        self.process: List[BaseProcess]

        self.config = config['config']
        self.raw_config = config
        self.job = config['job']
        self.name = self.get_conf('name', required=True)
        if 'meta' in config:
            self.meta = config['meta']
        else:
            self.meta = OrderedDict()

    def get_conf(self, key, default=None, required=False):
        if key in self.config:
            return self.config[key]
        elif required:
            raise ValueError(f'config file error. Missing "config.{key}" key')
        else:
            return default

    def run(self):
        print("")
        print(f"#############################################")
        print(f"# Running job: {self.name}")
        print(f"#############################################")
        print("")
        # implement in child class
        # be sure to call super().run() first
        pass

    def run_processes(self):
        # A UI row tracks the current process. Caption counts and trainer steps are different units.
        ui = None
        if len(self.process) > 1 and os.environ.get("IS_AI_TOOLKIT_UI") == "1" and os.environ.get("AITK_JOB_ID"):
            from toolkit.ui_job_status import UIJobStatus
            ui = UIJobStatus(self.process[0].get_conf("sqlite_db_path", "./aitk_db.db"))
        for index, process in enumerate(self.process):
            self.active_process = process
            if ui is not None:
                if ui.should_stop() or ui.should_return_to_queue():
                    raise KeyboardInterrupt()
                # Steps belong to this process. Its trainer replaces zero with the checkpoint position on load.
                info = f"Process {index + 1}/{len(self.process)} · {process.get_conf('type')}"
                ui.begin_process(info, process.get_conf("train.steps", 0))
                print(info)
            process.run()

    def load_processes(self, process_dict: dict):
        # only call if you have processes in this job type
        if 'process' not in self.config:
            raise ValueError('config file is invalid. Missing "config.process" key')
        if len(self.config['process']) == 0:
            raise ValueError('config file is invalid. "config.process" must be a list of processes')

        module = importlib.import_module('jobs.process')

        # add the processes
        self.process = []
        try:
            for i, process in enumerate(self.config['process']):
                if 'type' not in process:
                    raise ValueError(f'config file is invalid. Missing "config.process[{i}].type" key')

                # check if dict key is process type
                if process['type'] in process_dict:
                    if isinstance(process_dict[process['type']], str):
                        ProcessClass = getattr(module, process_dict[process['type']])
                    else:
                        # it is the class
                        ProcessClass = process_dict[process['type']]
                    self.process.append(ProcessClass(i, self, process))
                else:
                    raise ValueError(f'config file is invalid. Unknown process type: {process["type"]}')
        except BaseException:
            # a constructor that failed after BaseTrainProcess took its save-root lease (or a later process failing)
            # must not leave this half-built job holding output folders
            self.release_save_roots()
            raise

    def release_save_roots(self):
        save_root_lease.release_owner(self)

    def cleanup(self):
        # if you implement this in child clas,
        # be sure to call super().cleanup() LAST
        self.release_save_roots()
        del self
