"""A finished job releases what the process-wide Accelerator holds (deep dive, 1 Nov 2026).

prepare() keeps every model, optimizer and scheduler in the Accelerator's lists; run() only called end_training(),
so with several configs in one run.py process the next job started with the previous one's still alive.
"""
import gc
import importlib
import inspect
import weakref
from types import SimpleNamespace

import torch
from accelerate import Accelerator

from test_rawr_trainer_continuity import cpu_only  # noqa: F401  (cpu_only is autouse)


def test_release_drops_prepared_objects():
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    model, prepared_optimizer, prepared_scheduler = accelerator.prepare(model, optimizer, scheduler)
    model(torch.ones(1, 2)).sum().backward()
    prepared_optimizer.step()
    refs = [weakref.ref(x) for x in (model, optimizer, scheduler)]
    job = SimpleNamespace(accelerator=accelerator, modules_being_trained=[model], optimizer=prepared_optimizer,
                          lr_scheduler=prepared_scheduler, params=[{'params': list(model.parameters())}])
    del model, optimizer, scheduler, prepared_optimizer, prepared_scheduler
    gc.collect()
    assert all(r() is not None for r in refs)  # the Accelerator's lists keep them alive
    base.release_accelerator_state(job)
    gc.collect()
    assert all(r() is None for r in refs)
    assert accelerator._models == [] and accelerator._optimizers == [] and accelerator._schedulers == []


def test_run_releases_after_done_hook():
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    source = inspect.getsource(base.run)
    assert source.rindex('self.release_accelerator_state()') > source.rindex('self.done_hook()')
