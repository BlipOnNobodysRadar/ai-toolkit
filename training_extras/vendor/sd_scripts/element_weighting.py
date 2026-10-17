import torch
from typing import Optional

def apply_token_mining(
    loss: torch.Tensor,
    sigmas: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    sigma_gate: bool = True,
) -> torch.Tensor:
    """Token-level hard-example mining for per-element (spatial) losses.

    Computes a per-token difficulty map from the *detached* per-element loss,
    converts it to multiplicative weights, and reweights the loss so that hard
    spatial tokens (edges, textures) contribute more gradient than easy ones
    (flat regions). Weights are detached, so the model cannot inflate its own
    mining weights.

    Weight construction per sample:
        w_i = clamp((L_i / median(L)) ** alpha, min_weight, max_weight)
    followed by renormalization to mean 1 per sample, so the overall loss scale
    matches the plain mean reduction.

    When ``sigma_gate`` is enabled and ``sigmas`` are provided, mining strength
    is gated by g(sigma) = clip(4*sigma*(1-sigma), 0, 1): full strength at
    mid-schedule, disabled at the sigma extremes (where per-token loss
    variation is mostly irreducible noise). The gate blends weights toward
    uniform: w = 1 + g*(w-1), again renormalized to mean 1.

    Args:
        loss: Per-element loss, shape (B, C, ...) with >= 3 dims (e.g. (B,C,H,W)).
            Tensors with fewer than 3 dims are returned unchanged.
        sigmas: Optional per-sample flow-matching sigmas, shape (B,) or (B,1,...).
        alpha: Difficulty exponent. Higher values concentrate more weight on
            hard tokens.
        min_weight / max_weight: Clamp bounds for the mining weights, relative
            to uniform (1.0).
        sigma_gate: Enable the sigma-dependent strength gate.

    Returns:
        Weighted loss tensor, same shape and dtype as ``loss``.
    """
    if loss.ndim < 3:
        return loss

    with torch.no_grad():
        per_token = loss.detach().to(torch.float32).mean(dim=1)  # (B, ...) e.g. (B,H,W)
        flat = per_token.flatten(1)  # (B, N)
        med = flat.median(dim=1, keepdim=True).values.clamp(min=1e-12)
        w = (flat / med) ** alpha
        w = w.clamp(min_weight, max_weight)
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-12)

        if sigma_gate and sigmas is not None:
            s = sigmas.detach().reshape(sigmas.shape[0], -1)[:, 0].to(torch.float32).clamp(0.0, 1.0)
            g = (4.0 * s * (1.0 - s)).clamp(0.0, 1.0).unsqueeze(1)  # (B, 1)
            w = 1.0 + g * (w - 1.0)
            w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-12)

        w = w.view(per_token.shape).unsqueeze(1)  # (B, 1, ...)

    return loss * w.to(dtype=loss.dtype, device=loss.device)

class HaarDWT(torch.nn.Module):
    """One-level zero-padded Haar transform, layout B,C,3,H,W."""
    def __init__(self):
        super().__init__()
        self.register_buffer('filters',torch.tensor([[[1,1],[1,1]],[[-1,-1],[1,1]],[[-1,1],[-1,1]],[[1,-1],[-1,1]]],dtype=torch.float32).unsqueeze(1)/2)
    def forward(self,x):
        c=x.shape[1]
        x=torch.nn.functional.pad(x,(0,x.shape[-1]%2,0,x.shape[-2]%2))
        y=torch.nn.functional.conv2d(x,self.filters.to(x).repeat(c,1,1,1),stride=2,groups=c)
        y=y.reshape(x.shape[0],c,4,y.shape[-2],y.shape[-1])
        return y[:,:,0],[y[:,:,1:]]


def setup_wavelet_dwt(device):return HaarDWT().to(device).eval()


def compute_wavelet_attention_map(
    latent: torch.Tensor,
    dwt,
) -> torch.Tensor:
    """
    Compute a wavelet-based spatial saliency map from a noisy latent tensor.

    This implements Section 3.2 of the LWD paper: applies a single-level Haar DWT
    to the latent, aggregates high-frequency subband energy (LH, HL, HH),
    upsamples, and normalizes to [0, 1].

    The map highlights regions with strong structural detail (textures, edges,
    transitions) and is used to create a time-dependent binary mask that focuses
    the denoising loss on detail-rich spatial regions.

    Args:
        latent: Noisy latent tensor of shape (B, C, H, W)
        dwt: DWTForward module (from setup_wavelet_dwt)

    Returns:
        attn_map: Normalized attention map of shape (B, H, W), values in [0, 1]
    """
    B, C, H, W = latent.shape

    # Cast to float32 for wavelet processing (wavelets may not support fp16/bf16)
    original_dtype = latent.dtype
    latent_f32 = latent.to(dtype=torch.float32)

    # Apply 2D Haar DWT: LL low-frequency, high[0] contains (LH, HL, HH)
    # Each subband has shape (B, num_directions, C, H//2, W//2)
    _LL, high = dwt(latent_f32)
    LH = high[0][:, :, 0]  # (B, C, H//2, W//2)
    HL = high[0][:, :, 1]  # (B, C, H//2, W//2)
    HH = high[0][:, :, 2]  # (B, C, H//2, W//2)

    # Compute per-location high-frequency energy, averaged across channels
    # E(i,j) = (1/C) * sum_c [LH^2 + HL^2 + HH^2]  (Eq. 3 in paper)
    LH_energy = LH.pow(2).mean(dim=1)  # (B, H//2, W//2)
    HL_energy = HL.pow(2).mean(dim=1)  # (B, H//2, W//2)
    HH_energy = HH.pow(2).mean(dim=1)  # (B, H//2, W//2)
    HF_energy = LH_energy + HL_energy + HH_energy  # (B, H//2, W//2)

    # Upsample to original spatial resolution via bilinear interpolation
    attn_map_raw = torch.nn.functional.interpolate(
        HF_energy.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(1)  # (B, H, W)

    # Min-max normalize per sample to [0, 1]
    attn_min = attn_map_raw.amin(dim=(1, 2), keepdim=True)
    attn_max = attn_map_raw.amax(dim=(1, 2), keepdim=True)
    attn_map = (attn_map_raw - attn_min) / (attn_max - attn_min + 1e-8)  # (B, H, W)

    # Restore original dtype
    attn_map = attn_map.to(dtype=original_dtype)

    return attn_map

def get_wavelet_mask(
    A: torch.Tensor,
    l: float,
    T: int,
    timesteps: torch.Tensor,
    flow_matching: bool = False,
) -> torch.Tensor:
    """
    Compute the time-dependent binary mask for LWD wavelet masking.

    Implements Eq. 6 from the paper:
        M_t(i, j) = 1  if  T * (A_wavelet(i,j) + l) >= t
                   = 0  otherwise

    This ensures that:
    - All spatial regions receive at least l*T steps of supervision
    - High-frequency regions (A close to 1) receive up to (1+l)*T steps
    - Smooth regions (A close to 0) receive l*T steps (the minimum)

    Timestep scale handling:
    Eq. 6 is scale-invariant (M_t = 1 iff (A+l) >= t/T). Flow-matching
    trainers (Flux/SD3/Anima/Lumina/Hunyuan) divide timesteps by 1000 before
    returning them (t in [0, 1]), whereas DDPM trainers keep them in [0, T].
    The caller must pass ``flow_matching=True`` when timesteps are in [0, 1]
    so they are rescaled to [0, T] before the comparison. Without this,
    flow-matching trainers would always produce an all-ones mask
    (wavelet_mask_ratio stuck at 1.0).

    Args:
        A: Wavelet attention map of shape (B, H, W), values in [0, 1]
        l: Lower bound on supervision fraction (paper optimal: 0.3)
        T: Total number of diffusion timesteps (e.g., 1000)
        timesteps: Current timestep per sample, shape (B,). In [0, T] for
            DDPM trainers, or [0, 1] for flow-matching trainers (when
            ``flow_matching=True``).
        flow_matching: If True, timesteps are in [0, 1] and are rescaled
            to [0, T] before applying Eq. 6. Defaults to False (DDPM
            convention, timesteps already in [0, T]).

    Returns:
        mask: Binary mask of shape (B, 1, H, W), values {0.0, 1.0}
    """
    B, H, W = A.shape
    device = A.device

    # Ensure consistent dtype for computation
    original_dtype = A.dtype
    A_f32 = A.to(dtype=torch.float32, device=device)

    # Normalize timesteps to the [0, T] scale.
    # Flow-matching trainers (Flux/SD3/Anima/Lumina/Hunyuan) divide timesteps by
    # 1000 before returning them (t in [0, 1]), whereas DDPM trainers keep them in
    # [0, T]. Eq. 6 is scale-invariant: M_t = 1 iff T*(A+l) >= t, i.e. iff
    # (A+l) >= t/T. When flow_matching=True, rescale t from [0, 1] to [0, T] so
    # the comparison is correct. Without this, flow-matching trainers would
    # always produce an all-ones mask (wavelet_mask_ratio stuck at 1.0).
    t_f32 = timesteps.to(dtype=torch.float32, device=device)
    if flow_matching:
        t_scaled = t_f32 * float(T)
    else:
        t_scaled = t_f32

    # Broadcast timesteps to spatial dims: (B, 1, 1) -> (B, H, W)
    t_matrix = t_scaled.view(B, 1, 1).expand(B, H, W)

    # Compute threshold: T * (A + l)  (Eq. 6)
    thresholds = T * (A_f32 + l)  # (B, H, W)

    # Generate binary mask
    mask = (thresholds >= t_matrix).to(dtype=original_dtype)  # (B, H, W)

    # Add channel dimension for broadcasting with loss tensor (B, C, H, W)
    return mask.unsqueeze(1)