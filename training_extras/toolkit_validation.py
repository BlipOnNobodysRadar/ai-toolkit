"""Fixed held-out batches through the host's native image/video/audio path."""
import copy, json, random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from .optim import evaluation_optimizer

@contextmanager
def fixed_rng(seed, device):
    py, np_state = random.getstate(), np.random.get_state()
    devices = [device] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py); np.random.set_state(np_state)

def setup(trainer):
    from toolkit.data_loader import get_dataloader_from_datasets
    cfg = trainer.train_config.validation_config
    if trainer.train_config.train_text_encoder or getattr(trainer.sd,'is_edit',False) or any(d.get('do_i2v',False) or d.get('control_path') for d in cfg.datasets):
        raise ValueError('Cached held-out validation requires a frozen text encoder and plain image/video conditioning')
    configs = []
    for raw in cfg.datasets:
        data = dict(raw)
        data.update(random_crop=False,shuffle_tokens=False,caption_dropout_rate=0,flip_x=False,
            flip_y=False,augments=[],augmentations=[],num_workers=0,cache_latents=True,
            cache_latents_to_disk=False)
        if Path(data['folder_path']).resolve() in {Path(d.folder_path).resolve() for d in trainer.datasets}:
            raise ValueError('Training and validation folders must be separate')
        configs.append(data)
    cache = []
    device, dtype = trainer.device_torch, trainer.sd.torch_dtype
    trainer.sd.set_device_state_preset('cache_latents')
    try:
        with torch.no_grad(), fixed_rng(cfg.seed, device):
            loader = get_dataloader_from_datasets(configs, 1, trainer.sd)
            for i, batch in enumerate(loader):
                if i >= cfg.max_batches: break
                _, noise, _, prompts, _ = trainer.process_general_training_batch(batch)
                trainer.sd.text_encoder_to(device)
                embeds = trainer.sd.encode_prompt(prompts).detach().to('cpu', dtype=torch.float32)
                batch.latents = batch.latents.detach().to('cpu', dtype=torch.float32)
                noise = noise.detach().to('cpu', dtype=torch.float32)
                cache.append((batch,noise,embeds))
    finally:
        trainer.sd.restore_device_state()
    if not cache: raise ValueError('Validation dataset has no usable batches')
    trainer._validation_cache = {'native_batches':cache}

def validate(trainer, phase='periodic'):
    cfg = trainer.train_config.validation_config
    device, dtype = trainer.device_torch, trainer.sd.torch_dtype
    network = trainer.network
    was_training = trainer.sd.unet.training
    old_multiplier = network.multiplier if network is not None else None
    results = []
    try:
        trainer.sd.unet.eval()
        if network is not None: network.multiplier = 1.0
        from toolkit.stable_diffusion_model import BlankNetwork
        with torch.no_grad(), evaluation_optimizer(trainer.optimizer), network if network is not None else BlankNetwork():
            for item, (cached, noise_cpu, embeds_cpu) in enumerate(trainer._validation_cache['native_batches']):
                for sigma in cfg.validation_sigmas:
                    if not 0 <= sigma <= 1: raise ValueError('Validation sigmas must be in [0,1]')
                    with fixed_rng(cfg.seed + item, device):
                        batch = copy.copy(cached)
                        batch.latents = cached.latents.clone().to(device, dtype=dtype)
                        noise = noise_cpu.to(device,dtype=dtype)
                        t = float(sigma) * trainer.sd.noise_scheduler.config.num_train_timesteps
                        if not trainer.sd.is_flow_matching: t = min(round(t),trainer.sd.noise_scheduler.config.num_train_timesteps-1)
                        timesteps = torch.full((batch.latents.shape[0],),t,device=device,
                            dtype=torch.float32 if trainer.sd.is_flow_matching else torch.long)
                        noisy = trainer.sd.add_noise(batch.latents,noise,timesteps).detach()
                        pred = trainer.sd.predict_noise(latents=noisy,conditional_embeddings=embeds_cpu.clone().to(device,dtype=dtype),
                            timestep=timesteps,guidance_scale=1.0,batch=batch,
                            guidance_embedding_scale=trainer.train_config.cfg_scale,
                            bypass_guidance_embedding=trainer.train_config.bypass_guidance_embedding)
                        if hasattr(trainer.sd,'get_loss_target'):
                            target = trainer.sd.get_loss_target(noise=noise,batch=batch,timesteps=timesteps)
                        elif trainer.sd.is_flow_matching: target = noise - batch.latents
                        elif trainer.sd.prediction_type == 'v_prediction':
                            target = trainer.sd.noise_scheduler.get_velocity(batch.latents,noise,timesteps)
                        else: target = noise
                        row = {'sigma':sigma,'item':item,'video_mse':float((pred.float()-target.float()).square().mean())}
                        if hasattr(pred,'get') and pred.get('audio') is not None:
                            row['audio_mse'] = float((pred.get('audio').float()-pred.get('audio_target').float()).square().mean())
                        results.append(row)
        trainer.additional_logs['val/loss'] = float(np.mean([r['video_mse'] for r in results]))
        for sigma in cfg.validation_sigmas:
            trainer.additional_logs[f'val/raw_mse/sigma_{sigma:g}'] = float(np.mean([r['video_mse'] for r in results if r['sigma']==sigma]))
        audio = [r['audio_mse'] for r in results if 'audio_mse' in r]
        if audio: trainer.additional_logs['val/audio_mse'] = float(np.mean(audio))
        path = Path(trainer.save_root) / 'validation.jsonl'
        with path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'step':trainer.step_num,'phase':phase,'raw_metrics':results})+'\n')
    finally:
        if network is not None: network.multiplier = old_multiplier
        trainer.sd.unet.train(was_training)
