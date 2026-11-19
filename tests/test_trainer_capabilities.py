"""Alternative trainers refuse accumulation settings their hook_train_loop ignores (docs/STATE_MACHINES.md M27).

The six hook_train_loop overrides below zero gradients and call optimizer.step() on every call, so
gradient_accumulation_steps != 1 silently trained at a different effective batch size; TrainSliderProcess also uses
only batch[0], dropping gradient_accumulation > 1 batches. The real classes and the real
BaseSDTrainProcess._run_training entry; model setup is replaced by a recorder (the refusal must come first).
"""
import importlib
from types import SimpleNamespace

import pytest

ALTERNATIVES = [
    ('jobs.process.TrainSliderProcess', 'TrainSliderProcess'),
    ('jobs.process.TrainSliderProcessOld', 'TrainSliderProcessOld'),
    ('jobs.process.TrainSDRescaleProcess', 'TrainSDRescaleProcess'),
    ('extensions_built_in.concept_replacer.ConceptReplacer', 'ConceptReplacer'),
    ('extensions_built_in.image_reference_slider_trainer.ImageReferenceSliderTrainerProcess',
     'ImageReferenceSliderTrainerProcess'),
    ('extensions_built_in.ultimate_slider_trainer.UltimateSliderTrainerProcess', 'UltimateSliderTrainerProcess'),
]


def instance(module, name, gas=1, batches=1):
    cls = getattr(importlib.import_module(module), name)
    t = cls.__new__(cls)
    t.train_config = SimpleNamespace(gradient_accumulation_steps=gas, gradient_accumulation=batches)
    return t


def test_slider_configs_reach_the_slider_trainer():
    from jobs.TrainJob import process_dict
    assert process_dict['slider'] == 'TrainSliderProcess'  # config/examples/train_slider.example.yml uses it


@pytest.mark.parametrize('module,name', ALTERNATIVES)
def test_accumulation_steps_are_refused_before_any_model_loads(module, name, monkeypatch):
    from jobs.process.BaseSDTrainProcess import BaseTrainProcess
    t = instance(module, name, gas=4)
    loaded = []
    monkeypatch.setattr(BaseTrainProcess, 'run', lambda self: None)
    t.hook_before_model_load = lambda: loaded.append(True)
    with pytest.raises(ValueError, match='steps the optimizer on every iteration'):
        t._run_training()
    assert loaded == []


@pytest.mark.parametrize('module,name', ALTERNATIVES)
def test_the_default_setting_is_accepted(module, name):
    instance(module, name)._check_trainer_accumulation_capability()


@pytest.mark.parametrize('module,name', ALTERNATIVES)
def test_batch_lists_are_refused_only_where_the_trainer_drops_them(module, name):
    t = instance(module, name, batches=3)
    if name == 'TrainSliderProcess':
        with pytest.raises(ValueError, match='first batch of each iteration'):
            t._check_trainer_accumulation_capability()
    else:
        t._check_trainer_accumulation_capability()  # not shown to mishandle lists: unchanged


@pytest.mark.parametrize('gas,batches', [(4, 1), (-1, 1), (1, 3)])
def test_sdtrainer_keeps_every_accumulation_setting(gas, batches):
    instance('extensions_built_in.sd_trainer.SDTrainer', 'SDTrainer', gas, batches)._check_trainer_accumulation_capability()
