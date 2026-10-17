"""Opt-in loss composition with independent, resumable weighting state."""
import os
import torch
from .loss_bridge import LossBridge
from .edm import EDMWeighting
from .element_weighting import ElementWeighting
from .vendor.sd_scripts.dynamic_loss_weighting import DynamicLossWeighter

class TrainingExtras(torch.nn.Module):
    def __init__(self, config, scheduler, is_flow, prediction_type, device, total_steps):
        super().__init__()
        self.config = config
        allowed = {'loss','delta','delta_schedule','scale','loss_float64','x0_direct','anchor','hf','frequency','topology','dynamic','edm2','wavelet','token_mining'}
        unknown = set(config) - allowed
        if unknown: raise ValueError(f'Unknown training_extras keys: {sorted(unknown)}')
        if config.get('loss') == 'psnr_loss' and (config.get('edm2',{}).get('enabled') or config.get('token_mining',{}).get('enabled') or config.get('dynamic',{}).get('mode','none') != 'none'):
            raise ValueError('Negative PSNR objectives cannot be combined with EDM2, token mining or dynamic auxiliary weighting')
        self.elements = ElementWeighting({k:v for k,v in config.items() if k in ("wavelet","token_mining")},scheduler,is_flow)
        self.bridge = LossBridge(config, scheduler, is_flow, prediction_type)
        cfg = dict(config.get('edm2', {}), total_steps=total_steps)
        self.edm = EDMWeighting(cfg, scheduler, device)
        if self.edm.model is not None:
            self.edm.model.timesteps_in_sigma = False
        self.dynamic = DynamicLossWeighter(**config.get('dynamic', {}))

    def augment(self, per_sample, pred, clean, noisy, timesteps, parameters=()):
        value, logs = self.bridge.augment(per_sample, pred, clean, noisy, timesteps)
        if self.dynamic.mode != 'none' and self.bridge._aux_enabled():
            aux = value - per_sample
            weight = self.dynamic.compute_weight(per_sample.mean(), aux.mean(), list(parameters))
            value = per_sample + weight * aux
            logs['loss/dynamic_weight'] = weight
        value, weighted_logs = self.edm.weight(value, timesteps)
        logs.update(weighted_logs)
        return value, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in logs.items()}

    def step(self): self.edm.step()

    def checkpoint(self, path):
        payload = {'config':self.config, 'module':self.state_dict(), 'dynamic':self.dynamic.state_dict()}
        if self.edm.enabled:
            payload.update(optimizer=self.edm.aux_optimizer.state_dict(), scheduler=self.edm.aux_scheduler.state_dict())
        torch.save(payload, str(path)+'.tmp')
        os.replace(str(path)+'.tmp', path)
        if self.edm.enabled:
            import csv
            model=self.edm.model;training=model.training
            try:
                model.eval()
                with torch.no_grad():
                    t=torch.arange(model.num_train_timesteps,device=self.edm.device)
                    logvar=model._forward(t).detach()
                    weight=model.lambda_weights/torch.exp(logvar)
                curve=os.path.join(os.path.dirname(path),'edm2-weighting.csv')
                with open(curve+'.tmp','w',newline='',encoding='utf-8') as f:
                    writer=csv.writer(f);writer.writerow(['timestep','sigma','log_variance','prediction_weight'])
                    writer.writerows(zip(t.cpu().tolist(),(t/model.num_train_timesteps).cpu().tolist(),logvar.cpu().tolist(),weight.cpu().tolist()))
                os.replace(curve+'.tmp',curve)
            finally:model.train(training)

    def restore(self, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['config'] != self.config: raise ValueError('Loss configuration differs from saved training extras state')
        self.load_state_dict(data['module'])
        self.dynamic.load_state_dict(data['dynamic'])
        if self.edm.enabled:
            self.edm.aux_scheduler.load_state_dict(data['scheduler'])
            self.edm.aux_optimizer.load_state_dict(data['optimizer'])
