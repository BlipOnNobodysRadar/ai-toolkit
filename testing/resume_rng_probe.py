"""Resume-RNG probe: rawr_checkpoint_acceptance.py plus a per-step log of drawn images, timesteps and noise (draws.jsonl).

Only observation and an optional stop AFTER a successful normal save are added.
The production training, checkpoint serialization, model loading, optimizer
loading, accelerator preparation and startup path all execute normally.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'

parser=argparse.ArgumentParser()
parser.add_argument('--repo',required=True)
parser.add_argument('--config',required=True)
parser.add_argument('--evidence',required=True)
parser.add_argument('--stop-after-save',type=int)
parser.add_argument('--manual-save-at',type=int)
args=parser.parse_args()
sys.path.insert(0,args.repo)
os.chdir(args.repo)
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from toolkit.job import get_job
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from training_extras.rawr_checkpoint import underlying_scheduler,FIELDS,KEY

evidence=Path(args.evidence)
evidence.mkdir(parents=True,exist_ok=False)
def record(kind,p,**extra):
    scheduler=underlying_scheduler(p.lr_scheduler)
    row={'kind':kind,'step':p.step_num,'restored':getattr(p,'_rawr_scheduler_restored',False),
         'accelerator_device':str(p.accelerator.device),
         'groups':[{k:g[k] for k in FIELDS if k in g} for g in p.optimizer.param_groups],
         'scheduler':scheduler.state_dict(),**extra}
    with (evidence/'trace.jsonl').open('a') as f:
        f.write(json.dumps(row,default=str)+'\n')

original_pgtb=BaseSDTrainProcess.process_general_training_batch
def observed_pgtb(self,batch):
    out=original_pgtb(self,batch)
    noisy,noise,timesteps=out[0],out[1],out[2]
    files=[os.path.basename(getattr(fi,"path","?")) for fi in getattr(batch,"file_items",[])]
    row={"kind":"draw","step":self.step_num,"files":files,"timesteps":[round(float(t),3) for t in timesteps.flatten().tolist()],"noise_sum":round(float(noise.float().sum()),5),"noise_first":round(float(noise.flatten()[0]),6)}
    with (evidence/"draws.jsonl").open("a") as f: f.write(json.dumps(row)+chr(10))
    return out
BaseSDTrainProcess.process_general_training_batch=observed_pgtb
original_train=SDTrainer.hook_train_loop
def observed_train(self,batch):
    record('before_update',self)
    value=original_train(self,batch)
    record('after_update',self,loss=value)
    return value
SDTrainer.hook_train_loop=observed_train
original_save=BaseSDTrainProcess.save
class AcceptedStop(Exception): pass
def observed_save(self,step=None):
    original_save(self,step)
    opt_path=Path(self.save_root)/'optimizer.pt'
    opt=torch.load(opt_path,weights_only=True,map_location='cpu')
    if KEY in opt:  # RAWR runs: the optimizer payload must match the model metadata
        expected=self.meta['training_info']['step']
        assert opt[KEY]['trainer_step']==expected
    record('saved',self,payload=opt.get(KEY),completed_steps=self.meta['training_info'].get('completed_steps'))
    shutil.copy2(opt_path,evidence/f'optimizer-{self.step_num}.pt')
    if args.stop_after_save is not None and step==args.stop_after_save:
        raise AcceptedStop('Stopped after successful production checkpoint save')
BaseSDTrainProcess.save=observed_save
original_end=BaseSDTrainProcess.end_step_hook
def observed_end(self):
    original_end(self)
    if args.manual_save_at is not None and self.step_num==args.manual_save_at:
        self.save(self.step_num)
BaseSDTrainProcess.end_step_hook=observed_end
try:
    job=get_job(args.config)
    job.run()
    job.cleanup()
    status='completed'
except AcceptedStop:
    status='stopped_after_save'
summary={'status':status,'device':'cpu' if os.environ.get('CUDA_VISIBLE_DEVICES')=='' else 'default',
         'cuda_initialized':torch.cuda.is_initialized()}
if torch.cuda.is_initialized():
    summary.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
(evidence/'result.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
