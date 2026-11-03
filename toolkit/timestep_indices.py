import torch


def last_train_index(num_train_timesteps: int, noise_scheduler) -> int:
    """Highest timestep index the trainer may draw: the end of the grid the scheduler actually built, which can be
    shorter than num_train_timesteps (lognorm_blend at 30 builds 29 entries)."""
    timesteps = getattr(noise_scheduler, 'timesteps', None)
    if timesteps is None:
        return num_train_timesteps - 1
    return min(num_train_timesteps, len(timesteps)) - 1


def balanced_timestep_indices(min_noise_steps: int, max_noise_steps: int, batch_size: int, flowmatch: bool, device=None):
    """Uniform training-timestep indices for content_or_style 'balanced'.

    Flowmatch draws from [min, max] inclusive, like the 'content'/'style' branches: the trainer clips max to the grid's
    last index (last_train_index), which holds its lowest sigma, and torch.randint's exclusive upper bound used to leave
    that index out. Other schedulers keep their legacy draw, randint(min + 1, max - 1), i.e. [min + 1, max - 2].
    """
    if min_noise_steps == max_noise_steps:
        return torch.full((batch_size,), min_noise_steps, device=device, dtype=torch.long)
    if flowmatch:
        return torch.randint(min_noise_steps, max_noise_steps + 1, (batch_size,), device=device)
    return torch.randint(min_noise_steps + 1, max_noise_steps - 1, (batch_size,), device=device)
