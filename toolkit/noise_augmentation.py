"""Training noise augmentation for BCHW images and BCTHW videos.

Pyramid noise follows sd-scripts' spatial algorithm, including its global std
normalization. Video frames get independent spatial pyramids, with no temporal
interpolation. Input perturbation deliberately leaves the loss noise untouched.
Local generators keep augmentation out of the data/timestep RNG streams.
"""
import random
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def augmentation_seed(seed: int, step: int, microstep: int, stream: int, *, batch_index=0) -> int:
    return (int(seed) + 104729 * int(step) + 1009 * int(microstep)
            + 1000003 * int(stream) + 2147483647 * int(batch_index)) % (2**63 - 1)


def begin_noise_augmentation_batch(trainer):
    # List-based accumulation can prepare several batches at the same host step.
    # Separate them without changing ordinary one-batch-per-step seeds.
    key = (trainer.step_num, trainer.grad_accumulation_step)
    previous = getattr(trainer, '_noise_augmentation_batch_key', None)
    index = getattr(trainer, '_noise_augmentation_batch_index', 0) + 1 if key == previous else 0
    trainer._noise_augmentation_batch_key = key
    trainer._noise_augmentation_batch_index = index


@contextmanager
def disable_noise_augmentation(trainer):
    """Held-out noise must remain identical between augmented and control runs."""
    previous = getattr(trainer, '_disable_noise_augmentation', False)
    trainer._disable_noise_augmentation = True
    try:
        yield
    finally:
        trainer._disable_noise_augmentation = previous


def multires_noise_like(noise, iterations, discount, *, seed):
    if iterations == 0:
        return noise
    if noise.ndim not in (4, 5) or not noise.is_floating_point():
        raise ValueError('Multires noise requires floating BCHW or BCTHW latents')
    generator = torch.Generator(device=noise.device).manual_seed(seed)
    sizes = random.Random(seed)
    height, width = noise.shape[-2:]
    result = noise.float().clone()
    for i in range(iterations):
        ratio = sizes.random() * 2 + 2
        h = max(1, int(height / ratio**i))
        w = max(1, int(width / ratio**i))
        coarse = torch.randn((*noise.shape[:-2], h, w), device=noise.device,
                             dtype=torch.float32, generator=generator)
        # Flatten only leading dimensions: identical spatial treatment for video.
        resized = F.interpolate(coarse.reshape(-1, 1, h, w),
                                size=(height, width), mode='bilinear',
                                align_corners=False).reshape(noise.shape)
        result.add_(resized, alpha=discount**i)
        if h == 1 or w == 1:
            break
    std = result.std().clamp_min(torch.finfo(torch.float32).eps)
    return (result / std).to(noise.dtype)


def input_perturbed_noise(noise, gamma, *, seed, random_strength=False):
    """Return model-input noise; callers retain `noise` for the loss target."""
    if gamma == 0:
        return noise
    generator = torch.Generator(device=noise.device).manual_seed(seed)
    if random_strength:
        gamma *= torch.rand((), device=noise.device, generator=generator).item()
    delta = torch.randn(noise.shape, device=noise.device, dtype=torch.float32,
                        generator=generator)
    return (noise.float() + gamma * delta).to(noise.dtype)
