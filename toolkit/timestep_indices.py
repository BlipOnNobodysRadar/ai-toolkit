import torch


def balanced_timestep_indices(min_noise_steps: int, max_noise_steps: int, batch_size: int, flowmatch: bool, device=None):
    """Uniform training-timestep indices for content_or_style 'balanced'.

    Flowmatch draws from [min, max] inclusive, like the 'content'/'style' branches: the trainer clips max to the grid's
    last index, which holds its lowest sigma, and torch.randint's exclusive upper bound used to leave that index out.
    Other schedulers keep their legacy open interval (min, max).
    """
    if min_noise_steps == max_noise_steps:
        return torch.full((batch_size,), min_noise_steps, device=device, dtype=torch.long)
    if flowmatch:
        return torch.randint(min_noise_steps, max_noise_steps + 1, (batch_size,), device=device)
    return torch.randint(min_noise_steps + 1, max_noise_steps - 1, (batch_size,), device=device)
