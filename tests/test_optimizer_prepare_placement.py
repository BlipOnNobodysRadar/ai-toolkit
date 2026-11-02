"""prepare() leaves the optimizer state where its optimizer keeps it (deep dive A/B, 1 Nov 2026).

accelerate's default device placement round-trips the whole optimizer state through state_dict(), a move to the
accelerator device and load_state_dict(). On an H3 resume that copied 4.6 GiB of restored, CPU-offloaded moments onto
the GPU before the optimizer moved them back. torch's load_state_dict already casts state to each param's device;
automagic's lr_mask skips that cast, so its restore now follows the param's device itself.
"""
import importlib
import io
import types
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType


@pytest.fixture(autouse=True)
def fresh_accelerator_state():
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


class RecordingAccelerator:
    def __init__(self, distributed_type=DistributedType.NO, device='meta'):
        self.calls = []
        self.distributed_type, self.device = distributed_type, torch.device(device)

    def prepare(self, *args, device_placement=None):
        self.calls.append((args, device_placement))
        return args[0]


def _job(distributed_type):
    base = importlib.import_module('jobs.process.BaseSDTrainProcess').BaseSDTrainProcess
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=torch.tensor(1e-3))
    job = SimpleNamespace(accelerator=RecordingAccelerator(distributed_type), modules_being_trained=[], adapter=None,
                          sd=SimpleNamespace(vae=None, unet=None, text_encoder=None, refiner_unet=None, network=None),
                          optimizer=optimizer, lr_scheduler=None)
    job.prepare_optimizer_without_state_placement = types.MethodType(base.prepare_optimizer_without_state_placement, job)
    base.prepare_accelerator(job)
    return job, optimizer


def test_prepare_accelerator_skips_state_placement_but_moves_tensor_hyperparameters():
    job, optimizer = _job(DistributedType.NO)
    assert job.accelerator.calls == [((optimizer,), [False])]
    group = optimizer.param_groups[0]
    assert group['lr'].device.type == 'meta'  # the accelerator device (meta stands in for cuda here)
    assert all(p.device.type == 'cpu' for p in group['params'])


@pytest.mark.parametrize('distributed_type', [DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM])
def test_deepspeed_and_megatron_keep_accelerate_placement(distributed_type):
    job, optimizer = _job(distributed_type)
    assert job.accelerator.calls == [((optimizer,), None)]  # a custom placement raises there
    assert optimizer.param_groups[0]['lr'].device.type == 'cpu'


def test_unplaced_optimizer_does_not_round_trip_its_state():
    accelerator = Accelerator(cpu=True)
    calls = []

    class Spy(torch.optim.AdamW):
        def state_dict(self):
            calls.append('state_dict')
            return super().state_dict()

        def load_state_dict(self, state_dict):
            calls.append('load_state_dict')
            super().load_state_dict(state_dict)

    model = torch.nn.Linear(2, 2)
    prepared = accelerator.prepare(Spy(model.parameters(), lr=1e-3), device_placement=[False])
    assert isinstance(prepared, AcceleratedOptimizer) and calls == []
    AcceleratedOptimizer(Spy(model.parameters(), lr=1e-3), device_placement=True)  # control: the old path
    assert calls == ['state_dict', 'load_state_dict']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cpu_held_state_stays_on_cpu_and_gpu_default_would_move_it():
    accelerator = Accelerator()
    param = torch.nn.Parameter(torch.ones(4, 4, device='cuda'))

    def stepped_with_cpu_state():
        opt = torch.optim.AdamW([param], lr=1e-3)
        param.grad = torch.ones_like(param)
        opt.step()
        for state in opt.state.values():  # what a CPU-offloading optimizer holds after its own load
            state['exp_avg'], state['exp_avg_sq'] = state['exp_avg'].cpu(), state['exp_avg_sq'].cpu()
        return opt

    kept = accelerator.prepare(stepped_with_cpu_state(), device_placement=[False])
    assert all(s['exp_avg'].device.type == 'cpu' for s in kept.state.values())
    moved = accelerator.prepare(stepped_with_cpu_state())  # control: accelerate's default placement
    assert all(s['exp_avg'].device.type == 'cuda' for s in moved.state.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_automagic_lr_mask_follows_param_device_from_cpu_checkpoint():
    from toolkit.optimizers.automagic import Automagic
    param = torch.nn.Parameter(torch.randn(8, 8, device='cuda'))
    opt = Automagic([param], lr=1e-6)
    for _ in range(2):
        param.grad = torch.randn_like(param)
        opt.step()
    buffer = io.BytesIO()
    torch.save(opt.state_dict(), buffer)
    buffer.seek(0)
    restored = Automagic([param], lr=1e-6)
    restored.load_state_dict(torch.load(buffer, weights_only=True, map_location='cpu'))  # the RAWR resume path
    assert restored.state[param]['lr_mask'].quantized.device.type == 'cuda'
    param.grad = torch.randn_like(param)
    restored.step()  # mixed devices would raise here
