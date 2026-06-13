"""Packing / sampling helpers for Ideogram 4.

This module holds the glue that turns image latents + Qwen3-VL text features into
the single packed sequence the transformer consumes, plus a minimal flow-matching
sampling pipeline used to render preview images during training.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from transformers.masking_utils import create_causal_mask

from .transformer import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Transformer2DModel,
)


# ---------------------------------------------------------------------------
# Latent (un)patchification.
#
# The VAE produces (B, ae_ch=32, H/8, W/8) latents. The transformer works on
# tokens of dim ae_ch * patch**2 = 128. We store the patchified latent in a 4-D
# (B, 128, gh, gw) layout so the rest of ai-toolkit (noise, add_noise, loss) can
# treat it like an ordinary image latent. The channel ordering here matches the
# reference Ideogram 4 decode exactly: 128 = (patch_h, patch_w, ae_ch) with ae_ch
# the fastest-varying axis.
# ---------------------------------------------------------------------------


def patchify_latents(z: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """(B, ae_ch, H8, W8) -> (B, ae_ch * patch**2, gh, gw)."""
    b, ae_ch, h8, w8 = z.shape
    ph = pw = patch_size
    gh, gw = h8 // ph, w8 // pw
    z = z.view(b, ae_ch, gh, ph, gw, pw)
    # -> (B, ph, pw, ae_ch, gh, gw) then merge (ph, pw, ae_ch) -> channels
    z = z.permute(0, 3, 5, 1, 2, 4).reshape(b, ph * pw * ae_ch, gh, gw)
    return z


def unpatchify_latents(z: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """(B, ae_ch * patch**2, gh, gw) -> (B, ae_ch, H8, W8)."""
    b, c, gh, gw = z.shape
    ph = pw = patch_size
    ae_ch = c // (ph * pw)
    z = z.view(b, ph, pw, ae_ch, gh, gw)
    # -> (B, ae_ch, gh, ph, gw, pw) then merge spatial
    z = z.permute(0, 3, 4, 1, 5, 2).reshape(b, ae_ch, gh * ph, gw * pw)
    return z


# ---------------------------------------------------------------------------
# Qwen3-VL hidden-state extraction.
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_qwen3_vl_features(
    text_encoder,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_2d: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen3-VL and concat the hidden states from the activation layers.

    Returns a (B, L, hidden_size * num_layers) tensor (in the encoder's dtype),
    zeroed at non-text (padding) positions.
    """
    language_model = text_encoder.language_model

    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    causal_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states

    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    batch_size, seq_len = token_ids.shape
    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))  # (B, L, H, num_taps)
    stacked = stacked.reshape(batch_size, seq_len, -1)

    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask
    return stacked


# ---------------------------------------------------------------------------
# Packing + velocity prediction.
# ---------------------------------------------------------------------------


def pad_text_features(
    features_list: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of per-sample (Lt_i, D) features into a batch.

    Captions are stored at their natural length (one tensor per batch item) and
    only padded to the batch max here, right before the model call. Returns
    ``(features (B, Lt, D), attention_mask (B, Lt))``; the mask is 1 for real
    tokens and 0 for padding (which the transformer masks out anyway).
    """
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[-1]
    batch_size = len(features_list)

    features = torch.zeros(batch_size, max_len, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, f in enumerate(features_list):
        n = f.shape[0]
        features[i, :n] = f.to(device, dtype)
        mask[i, :n] = 1
    return features, mask


def predict_velocity(
    transformer: Ideogram4Transformer2DModel,
    latents: torch.Tensor,  # (B, 128, gh, gw)
    t: torch.Tensor,  # (B,) toolkit flow time in [0, 1] (1 = pure noise)
    llm_features: torch.Tensor,  # (B, Lt, llm_dim)
    text_mask: torch.Tensor,  # (B, Lt) 1 for real text tokens
) -> torch.Tensor:
    """Run the transformer on the packed [text | image] sequence.

    ``t`` is in the ai-toolkit flow-matching convention: ``t=1`` is pure noise,
    ``t=0`` is clean, and the returned velocity is ``noise - clean`` (matching the
    toolkit scheduler / loss target).

    Ideogram's transformer uses the opposite convention internally (``t=1`` is
    clean) and predicts ``clean - noise``, so we feed it ``1 - t`` and negate its
    output. Returns the velocity reshaped to the (B, 128, gh, gw) latent layout.
    """
    device = latents.device
    b, c, gh, gw = latents.shape
    num_image_tokens = gh * gw
    num_text_tokens = llm_features.shape[1]
    seq_len = num_text_tokens + num_image_tokens

    # image latents -> tokens (row-major: h outer, w inner)
    image_tokens = latents.permute(0, 2, 3, 1).reshape(b, num_image_tokens, c)

    # The mask may arrive as a float (PromptEmbeds.to casts it to the embed
    # dtype); work in long so cumsum positions stay exact for long prompts.
    text_mask_bool = text_mask.to(device) > 0
    text_mask_long = text_mask_bool.long()

    # noise tokens: text region is zeroed (masked out anyway)
    x = torch.cat(
        [
            torch.zeros(b, num_text_tokens, c, device=device, dtype=image_tokens.dtype),
            image_tokens,
        ],
        dim=1,
    )

    # llm features: image region is zero
    llm_full = torch.cat(
        [
            llm_features,
            torch.zeros(
                b,
                num_image_tokens,
                llm_features.shape[-1],
                device=device,
                dtype=llm_features.dtype,
            ),
        ],
        dim=1,
    )

    # indicator: real text -> 3, image -> 2, text pad -> 0
    indicator = torch.zeros(b, seq_len, dtype=torch.long, device=device)
    indicator[:, :num_text_tokens] = text_mask_long * LLM_TOKEN_INDICATOR
    indicator[:, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    # segment ids: real text + image -> 1, text pad -> -1 (its own padding segment)
    segment_ids = torch.ones(b, seq_len, dtype=torch.long, device=device)
    segment_ids[:, :num_text_tokens] = torch.where(
        text_mask_bool,
        torch.ones_like(text_mask_long),
        torch.full_like(text_mask_long, SEQUENCE_PADDING_INDICATOR),
    )

    # position ids (t, h, w)
    # text positions: 0..num_real-1 at the real slots (relative; pad -> 0)
    text_pos = (text_mask_long.cumsum(dim=-1) - 1).clamp(min=0)  # (B, Lt)
    text_pos_3d = text_pos.unsqueeze(-1).expand(-1, -1, 3)

    h_idx = torch.arange(gh, device=device).view(-1, 1).expand(gh, gw).reshape(-1)
    w_idx = torch.arange(gw, device=device).view(1, -1).expand(gh, gw).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
    image_pos_3d = image_pos.unsqueeze(0).expand(b, -1, -1)

    position_ids = torch.cat([text_pos_3d, image_pos_3d], dim=1)

    # Flip into the model's time convention (t=1 -> clean).
    model_t = 1.0 - t

    out = transformer(
        llm_features=llm_full,
        x=x,
        t=model_t,
        position_ids=position_ids,
        segment_ids=segment_ids,
        indicator=indicator,
    )

    image_velocity = out[:, num_text_tokens:]  # (B, Li, 128)
    image_velocity = image_velocity.reshape(b, gh, gw, c).permute(0, 3, 1, 2)
    # Model predicts clean - noise; negate to return toolkit velocity (noise - clean).
    return -image_velocity


# ---------------------------------------------------------------------------
# Asymmetric unconditional pass + guidance-aware prediction.
#
# Ideogram 4 ships a *separate* unconditional transformer (same architecture,
# different weights) and samples with classic CFG between the two networks:
#
#     v = v_uncond + g * (v_cond - v_uncond)
#
# The reference pipeline's negative branch is "asymmetric": it drops the text
# tokens entirely (the sequence is image tokens only) and feeds zeroed llm
# features. The helpers below reproduce that exact branch and expose the guided
# combination as a differentiable training operator: gradients flow through the
# conditional prediction only, the unconditional pass is always no-grad.
#
# Why train against the guided combination? A LoRA trained on the conditional
# branch alone gets its residual amplified by ``g`` at inference (g=7 for most
# of the official schedule), which fries outputs unless the LoRA is also
# applied to the unconditional model — which in turn cancels the differential
# and mutes the LoRA. Regressing ``v_uncond + g * (v_cond_lora - v_uncond)``
# against the flow target instead bakes the inference-time operator into
# training, so the resulting LoRA is meant to be applied to the conditional
# model only, at the guidance scale it was trained for.
# ---------------------------------------------------------------------------


def predict_velocity_unconditional(
    transformer: Ideogram4Transformer2DModel,
    latents: torch.Tensor,  # (B, 128, gh, gw)
    t: torch.Tensor,  # (B,) toolkit flow time in [0, 1] (1 = pure noise)
) -> torch.Tensor:
    """Run the asymmetric unconditional branch: image tokens only, zeroed text.

    Mirrors the reference pipeline's negative pass, which slices the text region
    off entirely (no padding tokens) and passes zeroed llm features over the
    image positions. Reuses :func:`predict_velocity` with a zero-length text
    segment, which produces exactly that packing: an image-only sequence, all
    ``OUTPUT_IMAGE_INDICATOR`` tokens, a single shared segment, and image-only
    MRoPE positions.
    """
    batch_size = latents.shape[0]
    llm_dim = transformer.config.llm_features_dim
    empty_features = torch.zeros(
        batch_size, 0, llm_dim, device=latents.device, dtype=latents.dtype
    )
    empty_mask = torch.zeros(batch_size, 0, dtype=torch.long, device=latents.device)
    return predict_velocity(transformer, latents, t, empty_features, empty_mask)


def combine_guided_velocity(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    guidance: "torch.Tensor | float",
    grad_rescale: bool = True,
) -> torch.Tensor:
    """``v = v_uncond + g * (v_cond - v_uncond)`` with optional gradient rescale.

    ``guidance`` may be a python float or a per-sample tensor of shape ``(B,)``.
    The combination is computed in float32 and cast back to ``v_cond.dtype``.
    ``v_uncond`` is always detached; gradients flow through ``v_cond`` only.

    With ``grad_rescale=True`` (default) the *forward value is unchanged* but
    the gradient w.r.t. ``v_cond`` is divided by ``g``. Without it, the loss
    gradient picks up an extra factor of ``g`` (d v/d v_cond = g), which acts
    like multiplying the learning rate by the guidance scale. Rescaling keeps
    effective step sizes comparable to ordinary (non-guided) training so
    existing LR recipes carry over.
    """
    if torch.is_tensor(guidance):
        g = guidance.to(device=v_cond.device, dtype=torch.float32)
    else:
        g = torch.tensor(float(guidance), device=v_cond.device, dtype=torch.float32)
    if g.dim() == 0:
        g = g.view(1)
    while g.dim() < v_cond.dim():
        g = g.unsqueeze(-1)

    v_uncond_f = v_uncond.detach().to(torch.float32)
    v = g * v_cond.to(torch.float32) + (1.0 - g) * v_uncond_f

    if grad_rescale:
        v_over_g = v / g
        v = v_over_g + (v - v_over_g).detach()

    return v.to(v_cond.dtype)


def guidance_aware_prediction(
    cond_transformer: Ideogram4Transformer2DModel,
    uncond_transformer: Ideogram4Transformer2DModel,
    latents: torch.Tensor,  # (B, 128, gh, gw)
    t: torch.Tensor,  # (B,) toolkit flow time in [0, 1]
    llm_features: torch.Tensor,  # (B, Lt, llm_dim)
    text_mask: torch.Tensor,  # (B, Lt)
    guidance: "torch.Tensor | float",
    grad_rescale: bool = True,
) -> torch.Tensor:
    """Differentiable guided velocity: cond pass with grad, uncond pass without."""
    v_cond = predict_velocity(cond_transformer, latents, t, llm_features, text_mask)
    with torch.no_grad():
        v_uncond = predict_velocity_unconditional(
            uncond_transformer, latents.detach(), t
        )
    return combine_guided_velocity(v_cond, v_uncond, guidance, grad_rescale=grad_rescale)


# ---------------------------------------------------------------------------
# Minimal sampling pipeline (for training previews).
# ---------------------------------------------------------------------------


class Ideogram4Pipeline:
    """Lightweight flow-matching sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # ``model`` is the Ideogram4Model so we can reuse its encode/decode and
        # latent helpers without duplicating state.
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[Image.Image]:
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer
        patch = model.patch_size

        # Use a fresh scheduler so we never mutate the training scheduler's state.
        scheduler = model.get_train_scheduler()
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        ae_scale = model.vae_scale_factor  # 8
        gh = height // (ae_scale * patch)
        gw = width // (ae_scale * patch)
        latent_channels = transformer.config.in_channels

        # When the separate unconditional transformer is loaded
        # (guidance_aware_training / load_unconditional_model), previews use
        # the *real* inference operator: the asymmetric image-only negative
        # branch on the dedicated unconditional model, exactly like the
        # reference pipeline and the dual-model ComfyUI workflow. Otherwise
        # fall back to classic CFG on the conditional model with the (usually
        # empty) negative prompt.
        uncond_transformer = getattr(model, "unconditional_transformer", None)
        use_uncond_model = uncond_transformer is not None and getattr(
            model, "sample_with_unconditional", True
        )
        do_cfg = guidance_scale != 1.0 and (
            use_uncond_model or unconditional_embeds is not None
        )

        if latents is None:
            shape = (1, latent_channels, gh, gw)
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=torch.float32
            )
        latents = latents.to(device, dtype=torch.float32)
        latents = latents * scheduler.init_noise_sigma

        cond_feats, cond_mask = pad_text_features(
            conditional_embeds.text_embeds, device, dtype
        )
        if do_cfg and not use_uncond_model:
            uncond_feats, uncond_mask = pad_text_features(
                unconditional_embeds.text_embeds, device, dtype
            )
        if do_cfg and use_uncond_model:
            ensure_fn = getattr(model, "_ensure_unconditional_on_device", None)
            if ensure_fn is not None:
                ensure_fn()

        for t in timesteps:
            t01 = (t / 1000.0).to(device).expand(latents.shape[0])
            v_cond = predict_velocity(
                transformer, latents.to(dtype), t01, cond_feats, cond_mask
            )
            if do_cfg:
                if use_uncond_model:
                    v_uncond = predict_velocity_unconditional(
                        uncond_transformer, latents.to(dtype), t01
                    )
                else:
                    v_uncond = predict_velocity(
                        transformer, latents.to(dtype), t01, uncond_feats, uncond_mask
                    )
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            latents = scheduler.step(
                v.to(torch.float32), t, latents, return_dict=False
            )[0]

        images = model.decode_latents(latents, device=device, dtype=dtype)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
