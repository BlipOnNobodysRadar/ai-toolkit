"""Offline preference training for flow-matching image models."""

from copy import copy

import torch
import torch.nn.functional as F

from toolkit.prompt_utils import concat_prompt_embeds


def validate_dpo_config(trainer):
    train = trainer.train_config
    datasets = trainer.dataset_configs
    if train.loss_type != 'dpo':
        if any(dataset.rejected_path for dataset in datasets):
            raise ValueError("datasets[].rejected_path requires train.loss_type: dpo")
        return

    model = trainer.model_config
    # ModelConfig strips the UI variant suffix, including on hand-written YAML.
    selected_arch = getattr(trainer, 'config', {}).get('model', {}).get('arch')
    if (model.arch != 'moss_image' or model.model_kwargs.get('edit')
            or selected_arch in ('moss_image:edit', 'moss_image:base_edit')):
        raise ValueError('DPO currently supports MOSS-Image text-to-image training')
    if train.noise_scheduler != 'flowmatch' or train.loss_target != 'noise':
        raise ValueError("MOSS DPO requires noise_scheduler: flowmatch and the default flow target")
    if (trainer.network_config is None or trainer.network_config.type != 'lora'
            or not train.train_unet or train.train_text_encoder
            or trainer.adapter_config or trainer.embed_config or trainer.decorator_config):
        raise ValueError('DPO requires a transformer LoRA with a frozen base and text encoder')
    if model.refiner_name_or_path or trainer.guidance_config:
        raise ValueError('DPO does not support a refiner or a guidance process')
    if not datasets or not all(d.rejected_path and not d.is_reg for d in datasets):
        raise ValueError('Every DPO dataset must provide a rejected_path for its chosen folder')
    if (train.unload_text_encoder or trainer.is_caching_text_embeddings) and not all(
            d.cache_text_embeddings for d in datasets):
        raise ValueError('DPO needs cached captions for every dataset when unloading the text encoder')

    # These modes change the reference weights, condition the noise on one arm,
    # or use an auxiliary loss/conditioning branch that this objective replaces.
    incompatible = (
        'merge_network_on_save', 'short_and_long_captions', 'single_item_batching',
        'dynamic_noise_offset', 'do_signal_correction_noise', 'do_batch_noise_correction',
        'do_blank_stabilization', 'blended_blur_noise', 'training_extras',
        'do_cfg', 'do_random_cfg', 'do_guidance_loss', 'do_differential_guidance',
        'diff_output_preservation', 'blank_prompt_preservation', 'inverted_mask_prior',
        'min_snr_gamma', 'snr_gamma', 'learnable_snr_gos', 'do_fft_loss', 't0_loss_target',
        'train_turbo', 'do_prior_divergence', 'diffusion_feature_extractor_path',
        'adapter_assist_name_or_path',
        'match_noise_norm', 'correct_pred_norm', 'target_norm_std', 'do_signal_amplification',
    )
    for key in incompatible:
        if getattr(train, key, None):
            raise ValueError(f'train.{key} is incompatible with DPO')
    if train.optimal_noise_pairing_samples > 1 or train.pred_scaler != 1 or train.target_noise_multiplier != 1:
        raise ValueError('DPO requires ordinary noise pairing and unit target and prediction scales')

    from toolkit.data_loader import validate_dpo_config_options
    for dataset in datasets:
        validate_dpo_config_options(dataset)


def prediction_errors(prediction, target):
    """Mean squared velocity error per image, before any batch reduction."""
    if prediction.shape != target.shape:
        raise ValueError(f'DPO prediction shape {prediction.shape} differs from target {target.shape}')
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def preference_logits(policy_errors, reference_errors, beta):
    """Chosen and rejected halves share pair order. A positive logit favors chosen."""
    chosen_delta, rejected_delta = (policy_errors - reference_errors.detach()).chunk(2)
    return (0.5 * beta) * (rejected_delta - chosen_delta)


def paired_batch(batch):
    # The loader batch stays reusable. General preprocessing writes its encoded
    # latents onto this shallow view; captions and item weights remain B-wide.
    paired = copy(batch)
    if batch.latents is None:
        chosen, rejected = batch.tensor, batch.rejected_tensor
    else:
        chosen, rejected = batch.latents, batch.rejected_latents
    if chosen is None or rejected is None or chosen.shape != rejected.shape:
        raise ValueError('DPO requires a matching chosen and rejected tensor for every pair')
    if len(batch.file_items) != chosen.shape[0]:
        raise ValueError('DPO needs one chosen caption per image pair')
    if batch.latents is None:
        paired.tensor = torch.cat((chosen, rejected))
    else:
        paired.tensor = None
        paired.latents = torch.cat((chosen, rejected))
    return paired


@torch.no_grad()
def reference_errors(trainer, noisy, timesteps, conditioning, target):
    network = trainer.network
    previous_active = network.is_active
    try:
        network.is_active = False
        return prediction_errors(trainer.predict_noise(noisy, timesteps, conditioning), target)
    finally:
        network.is_active = previous_active


def train_dpo_accumulation(trainer, batch, accum_scale):
    with torch.no_grad():
        paired = paired_batch(batch)
        noisy, noise, timesteps, prompts, _ = trainer.process_general_training_batch(
            paired, shared_pair_noise=True)
        target = trainer.sd.get_loss_target(noise=noise, batch=paired)
        embeds = batch.prompt_embeds
        if embeds is None:
            if trainer.train_config.unload_text_encoder or trainer.is_caching_text_embeddings:
                raise ValueError('Missing cached caption embedding for a DPO pair')
            embeds = trainer.sd.encode_prompt(prompts)
        conditioning = concat_prompt_embeds([embeds, embeds])
        with trainer.timer('dpo_reference'):
            frozen_errors = reference_errors(trainer, noisy, timesteps, conditioning, target)

    network = trainer.network
    previous_active, previous_multiplier = network.is_active, network.multiplier
    try:
        network.multiplier = batch.get_network_weight_list() * 2
        # Checkpoint recomputation must see the policy LoRA during backward.
        with network:
            with trainer.timer('dpo_policy'):
                policy_errors = prediction_errors(trainer.predict_noise(noisy, timesteps, conditioning), target)
                logits = preference_logits(policy_errors, frozen_errors, trainer.train_config.dpo_beta)
                weights = logits.new_tensor(batch.loss_multiplier_list)
                loss = (-F.logsigmoid(logits) * weights).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite DPO loss')
            with trainer.timer('backward'):
                trainer.accelerator.backward(loss * accum_scale)
    finally:
        network.is_active = previous_active
        network.multiplier = previous_multiplier

    trainer.additional_logs['dpo/margin'] = logits.detach().mean().item()
    trainer.additional_logs['dpo/accuracy'] = (logits.detach() > 0).float().mean().item()
    return loss.detach()
