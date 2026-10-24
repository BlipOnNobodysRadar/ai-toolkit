"""CPU/offline observer for the production tiny SDTrainer, no replacement updates."""
import os
for key, value in {
    'CUDA_VISIBLE_DEVICES': '', 'ACCELERATE_USE_CPU': 'true',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'NO_ALBUMENTATIONS_UPDATE': '1',
    'AITK_HF_DOWNLOADER': 'default', 'PYTHONDONTWRITEBYTECODE': '1',
}.items():
    os.environ[key] = value
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
def no_cuda(*args, **kwargs):
    raise AssertionError('CUDA initialization forbidden by CPU probe')
torch.cuda._lazy_init = no_cuda
torch.cuda.init = no_cuda
torch.cuda.is_available = lambda: False
torch.accelerator.current_accelerator = lambda *a, **kw: None
import numpy as np
np.random.seed(1234)
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--gas', type=int, default=1)
parser.add_argument('--stop', type=int)
parser.add_argument('--save-every', type=int, default=2)
parser.add_argument('--controlled', action='store_true')
parser.add_argument('--fixtures', required=True, type=Path)
parser.add_argument('--evidence-root', required=True, type=Path)
parser.add_argument('--crash-before-commit', type=int)
parser.add_argument('--learnable-snr', action='store_true')
parser.add_argument('--default-export-dtype', action='store_true')
args = parser.parse_args()
evidence = args.evidence_root / args.name
evidence.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load((args.fixtures / 'template.yaml').read_text())
config['config']['name'] = args.name
p = config['config']['process'][0]
p['training_folder'] = str(evidence / 'output')
p['model']['name_or_path'] = str(args.fixtures / 'tiny-sd')
p['datasets'][0]['folder_path'] = str(args.fixtures / ('controlled-data' if args.controlled else 'tiny-data'))
p['datasets'][0]['num_workers'] = 0
p['save']['save_every'] = args.save_every
p['save']['max_step_saves_to_keep'] = 20
p['train']['steps'] = 8
p['train']['gradient_accumulation_steps'] = args.gas
p['train']['optimizer_params']['beta1_warmup'] = 8 // args.gas
if args.learnable_snr:
    p['train']['learnable_snr_gos'] = True
if args.default_export_dtype:
    p['save'].pop('dtype', None)
cfg_path = evidence / 'config.yaml'
cfg_path.write_text(yaml.safe_dump(config))

from toolkit.job import get_job
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from training_extras.rawr_checkpoint import underlying_scheduler
from training_extras.rawr_stochastic import snr_state, rng_state

trace_path = evidence / ('resume.jsonl' if (evidence / 'trace.jsonl').exists() else 'trace.jsonl')
def params(self):
    return [x for g in self.optimizer.param_groups for x in g['params']]
def record(kind, self):
    row = dict(kind=kind, step=self.step_num, counter=self.grad_accumulation_step,
               accumulating=self.is_grad_accumulation_step,
               scheduler=underlying_scheduler(self.lr_scheduler).state_dict(),
               rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
               param_sum=sum(x.detach().double().sum().item() for x in params(self)),
               grad_norm=sum(x.grad.double().norm().item() for x in params(self) if x.grad is not None),
               optimizer_steps=[g.get('step', 0) for g in self.optimizer.param_groups],
               state_dtypes=sorted({str(v.dtype) for s in self.optimizer.state.values() for v in s.values() if torch.is_tensor(v)}),
               cuda_initialized=torch.cuda.is_initialized())
    with trace_path.open('a') as f:
        f.write(json.dumps(row, default=str) + '\n')

original_train = SDTrainer.hook_train_loop
def snapshot(self):
    return {'params': [x.detach().cpu().clone() for x in params(self)],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'snr': snr_state(self), 'rng': rng_state(),
            'gradients': [None if x.grad is None else x.grad.detach().cpu().clone() for x in params(self)]}

def observed_train(self, batch):
    record('before', self)
    torch.save(snapshot(self),
               evidence / f'{trace_path.stem}-before-{self.step_num}.pt')
    # Optional isolation of stochastic input: reset RNG per loop, feed the same
    # copied image. Still use actual model/noise/backward/optimizer/scheduler.
    if args.controlled:
        torch.manual_seed(4321 + self.step_num)
        random.seed(4321 + self.step_num)
        np.random.seed(4321 + self.step_num)
    result = original_train(self, batch)
    record('after', self)
    torch.save(snapshot(self),
               evidence / f'{trace_path.stem}-step-{self.step_num}.pt')
    return result
SDTrainer.hook_train_loop = observed_train

class AcceptedStop(Exception):
    pass
original_save = BaseSDTrainProcess.save
def observed_save(self, step=None):
    record('before_save', self)
    if args.crash_before_commit is not None and step == args.crash_before_commit:
        from training_extras import rawr_pair
        def crash(*args):
            # This is an owned fixture process. Exit without Python cleanup,
            # after model/optimizer writes but before the manifest replacement.
            print('INJECTED_PROCESS_EXIT_BEFORE_MANIFEST', flush=True)
            os._exit(87)
        rawr_pair._write_manifest = crash
    original_save(self, step)
    record('saved', self)
    if args.stop is not None and step == args.stop:
        raise AcceptedStop()
BaseSDTrainProcess.save = observed_save
status = 'failed'
try:
    job = get_job(str(cfg_path))
    job.run()
    job.cleanup()
    status = 'completed'
except AcceptedStop:
    status = 'stopped_after_save'
finally:
    result = {'status': status, 'cuda_initialized': torch.cuda.is_initialized(),
              'torch': torch.__version__, 'cpu_threads': torch.get_num_threads()}
    (evidence / f'{trace_path.stem}-result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
