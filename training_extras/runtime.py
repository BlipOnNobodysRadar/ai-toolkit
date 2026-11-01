"""Opt-in loss composition with independent, resumable weighting state."""
import os
import warnings
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

    def checkpoint(self, path, curve_dir=None, step=None):
        # step: completed training steps this state belongs to (what a resume restores into step_num);
        # restore refuses state from another checkpoint (a stale or mismatched resume).
        payload = {'config':self.config, 'module':self.state_dict(), 'dynamic':self.dynamic.state_dict(),
                   'step':None if step is None else int(step)}
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
                # The curve is a report for the save folder, even when the state lives in a RAWR pair slot.
                curve=os.path.join(curve_dir if curve_dir is not None else os.path.dirname(path),'edm2-weighting.csv')
                with open(curve+'.tmp','w',newline='',encoding='utf-8') as f:
                    writer=csv.writer(f);writer.writerow(['timestep','sigma','log_variance','prediction_weight'])
                    writer.writerows(zip(t.cpu().tolist(),(t/model.num_train_timesteps).cpu().tolist(),logvar.cpu().tolist(),weight.cpu().tolist()))
                os.replace(curve+'.tmp',curve)
            finally:model.train(training)

    def restore(self, path, expected_step=None):
        """Restore, checking the step stamp when expected_step is given (a committed RAWR pair's own copy)."""
        data = torch.load(path, map_location='cpu', weights_only=True)
        saved = data.get('step')
        if expected_step is not None:
            if saved is None:
                warnings.warn(f'{path} has no step stamp (legacy file); cannot certify it matches resume step {expected_step}', RuntimeWarning)
            elif int(saved) != int(expected_step):
                raise ValueError(f'training_extras state step {saved} does not match resume step {expected_step}')
        self._apply(data)

    def restore_for_step(self, path, step_num, strict=True, log=print):
        """Restore only state saved at the step being resumed. Returns 'restored', 'legacy' or 'stale'.

        A fresh start (step 0) moves a leftover file aside instead of reviving an old run's EDM2 state;
        a file from another step raises when strict (warns and restores otherwise); a file written
        before steps were recorded is restored with a warning.
        """
        data = torch.load(path, map_location='cpu', weights_only=True)
        saved = data.get('step')
        if saved is not None and int(saved) == int(step_num):
            self._apply(data); return 'restored'
        if int(step_num) == 0:
            stale = f'{path}.stale-{int(os.path.getmtime(path))}'
            os.replace(path, stale)
            log(f'training_extras: fresh start but {path} holds state from step {saved if saved is not None else "unknown"}; moved it to {stale} and starting EDM2/dynamic weighting fresh')
            return 'stale'
        if saved is None:
            log(f'training_extras: {path} has no recorded step (written before step tracking); restoring it for step {step_num} unchecked')
            self._apply(data); return 'legacy'
        msg = (f'training_extras: {path} was saved at step {saved} but training resumes at step {step_num}. '
               'Delete or rename training_extras.pt to reset EDM2/dynamic loss weighting and resume.')
        if strict: raise ValueError(msg)
        log(msg + ' Restoring anyway because train.start_step overrides the step.')
        self._apply(data); return 'restored'

    def _apply(self, data):
        if data['config'] != self.config: raise ValueError('Loss configuration differs from saved training extras state')
        self.load_state_dict(data['module'])
        self.dynamic.load_state_dict(data['dynamic'])
        if self.edm.enabled:
            self.edm.aux_scheduler.load_state_dict(data['scheduler'])
            self.edm.aux_optimizer.load_state_dict(data['optimizer'])
