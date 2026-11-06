"""A failed preview must not leave a process-wide scheduler override installed.

Execute the actual install/use/cleanup region from each generate_images method,
with lightweight CPU objects, and inject failure after the real per-image sigma
assignment. Heavy model-loading setup is outside this regression's scope.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from toolkit.sample_sigmas import install_sample_sigma_override


def _sample_region(relative_path):
    tree = ast.parse((Path(__file__).resolve().parents[1] / relative_path).read_text(encoding='utf-8'))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'generate_images')
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'unwrap_sigma_override' for t in n.targets))
    end = next(i for i in range(start + 1, len(fn.body))
               if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                      and n.func.id == 'unwrap_sigma_override' for n in ast.walk(fn.body[i])))
    region = ast.parse('def probe(self, pipeline, image_configs, network):\n    pass\n').body[0]
    region.body = copy.deepcopy(fn.body[start:end + 1])
    namespace = {'torch': torch, 'tqdm': lambda it, **kwargs: it,
                 'install_sample_sigma_override': install_sample_sigma_override,
                 'unwrap_step_hooks': lambda: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[region], type_ignores=[])), relative_path, 'exec'), namespace)
    return namespace['probe']


@pytest.mark.parametrize('relative_path', ['toolkit/models/base_model.py', 'toolkit/stable_diffusion_model.py'])
@pytest.mark.parametrize('inherited', [False, True])
def test_failed_preview_restores_scheduler_class(relative_path, inherited):
    class FlowMatchFixture:
        config = SimpleNamespace(num_train_timesteps=1000, use_flow_sigmas=True)

        def set_timesteps(self, steps):
            self.timesteps = torch.arange(steps)
            self.sigmas = torch.ones(steps + 1)

    class Child(FlowMatchFixture):
        pass

    cls = Child if inherited else FlowMatchFixture
    own_method = cls.__dict__.get('set_timesteps')
    original = cls.set_timesteps
    scheduler = cls()
    holder = SimpleNamespace(adapter=object(), refiner_unet=None, noise_scheduler=scheduler)

    class BadPreview:
        sigmas = [1.0, 0.5]

        @property
        def adapter_image_path(self):
            scheduler.set_timesteps(9)
            assert len(scheduler.timesteps) == 2  # the override really was active
            raise RuntimeError('preview failure')

    class Network:
        multiplier = 1.0
        is_active = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    try:
        with pytest.raises(RuntimeError, match='preview failure'):
            _sample_region(relative_path)(holder, SimpleNamespace(scheduler=scheduler), [BadPreview()], Network())
        assert cls.set_timesteps is original
        assert cls.__dict__.get('set_timesteps') is own_method
        assert holder.sample_sigmas is None
        scheduler.set_timesteps(7)
        assert len(scheduler.timesteps) == 7
    finally:
        # Keep the deliberate unfixed-baseline run isolated too.
        if own_method is None:
            if 'set_timesteps' in cls.__dict__:
                del cls.set_timesteps
        else:
            cls.set_timesteps = own_method
