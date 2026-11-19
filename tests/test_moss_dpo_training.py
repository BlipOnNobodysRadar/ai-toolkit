"""CPU acceptance contracts for MOSS-Image DPO through the real SDTrainer.

The model is a tiny but real MossImageTransformer with the production
CustomFlowMatchEulerDiscreteScheduler and MossImageModel.get_noise_prediction.
Only the VAE and text encoder are light fakes for the on-the-fly encoding path;
the native Toolkit LoRA and gradient checkpointing are real.
"""

import math
from contextlib import nullcontext
from copy import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from extensions_built_in.diffusion_models.moss_image.src import transformer as moss_transformer
from extensions_built_in.diffusion_models.moss_image.src.transformer import (
    MossImageParams,
    MossImageTransformer,
)
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import DatasetConfig, NetworkConfig, TrainConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.diffusion_dpo import validate_dpo_config
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.prompt_utils import concat_prompt_embeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

TINY = MossImageParams(
    dim=32,
    ffn_dim=64,
    num_heads=4,
    num_layers=2,
    text_dim=16,
    freq_dim=16,
    latent_channels=4,
)
BATCH_SIZE = 2
TEXT_LEN = 5
LATENT_SHAPE = (4, 4, 4)


@pytest.mark.parametrize("beta", [0, -1, math.inf, math.nan, True])
def test_dpo_rejects_invalid_beta(beta):
    with pytest.raises(ValueError, match="finite positive"):
        TrainConfig(loss_type="dpo", dpo_beta=beta)


@pytest.mark.parametrize("variant", ["moss_image:edit", "moss_image:base_edit"])
def test_dpo_rejects_edit_variant_even_after_arch_normalization(variant):
    trainer = SimpleNamespace(
        config={"model": {"arch": variant}},
        model_config=SimpleNamespace(arch="moss_image", model_kwargs={}),
        train_config=TrainConfig(loss_type="dpo"),
        dataset_configs=[],
    )
    with pytest.raises(ValueError, match="text-to-image"):
        validate_dpo_config(trainer)


class _Timer:
    def __call__(self, *args, **kwargs):
        return nullcontext()

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass


class _Backward:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def backward(self, loss):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected DPO backward failure")
        loss.backward()


class _FakeVAE(torch.nn.Module):
    """Pixel (B,3,H,W) -> deterministic (B,4,4,4) latent."""

    dtype = torch.float32

    def encode(self, images):
        pooled = F.adaptive_avg_pool2d(images.float(), (4, 4))
        return torch.cat((pooled, pooled.mean(dim=1, keepdim=True)), dim=1)


class _FakeTextEncoder(torch.nn.Module):
    dtype = torch.float32

    def encode(self, prompts):
        out = []
        for prompt in prompts:
            digest = sum(ord(ch) for ch in prompt)
            out.append(
                torch.randn(
                    TEXT_LEN,
                    TINY.text_dim,
                    generator=torch.Generator().manual_seed(10_000 + digest),
                )
            )
        return out


def _make_model():
    torch.manual_seed(1)
    transformer = MossImageTransformer(TINY)
    transformer.enable_gradient_checkpointing(True)
    transformer.train()

    model = MossImageModel.__new__(MossImageModel)
    model.model = transformer
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.is_flow_matching = True
    model.is_multistage = False
    model.is_edit = False
    model.kv_cache = False
    model.arch = "moss_image"
    model.model_config = SimpleNamespace(arch='moss_image', model_kwargs={}, refiner_name_or_path=None)
    model.use_old_lokr_format = False
    model.lora_keys_use_comfy_prefix = True
    model.is_transformer = True
    model.target_lora_modules = ["MossImageTransformer"]
    model.noise_scheduler = CustomFlowMatchEulerDiscreteScheduler(
        shift=5.0, num_train_timesteps=1000, use_dynamic_shifting=False
    )
    model.vae = _FakeVAE().eval().requires_grad_(False)
    model.text_encoder = _FakeTextEncoder().eval().requires_grad_(False)
    model.vae_torch_dtype = torch.float32
    model.te_torch_dtype = torch.float32
    model.x0_pred = False

    # Real image-encoder hook for the on-the-fly (rejected_tensor) arm. The
    # cached-latent test bypasses this, just like production cached training.
    def encode_images(image_list, device=None, dtype=None):
        images = image_list.to(device=device, dtype=torch.float32)
        return model.vae.encode(images).to(device=device, dtype=dtype or torch.float32)

    model.encode_images = encode_images

    def encode_prompt(prompts, *args, **kwargs):
        if not isinstance(prompts, list):
            prompts = [prompts]
        return AdvancedPromptEmbeds(text_embeds=model.text_encoder.encode(prompts))

    model.encode_prompt = encode_prompt
    return model, transformer


def _make_network(model, transformer):
    network_config = NetworkConfig(
        type="lora", linear=4, linear_alpha=4, transformer_only=True
    )
    transformer.requires_grad_(False)
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=transformer,
        lora_dim=4,
        alpha=4,
        multiplier=1.0,
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type="lora",
        transformer_only=True,
        is_transformer=True,
        base_model=model,
        target_lin_modules=["MossImageTransformer"],
    )
    network.model_ref = model
    network.apply_to(None, transformer, False, True)
    network.to("cpu", dtype=torch.float32)
    network.enable_gradient_checkpointing()
    assert transformer.gradient_checkpointing is True
    return network, network_config


def _make_trainer(model, transformer, network, network_config):
    train_config = TrainConfig(
        loss_type="dpo",
        dpo_beta=100,
        noise_scheduler="flowmatch",
        loss_target="noise",
        timestep_type="sigmoid",
        content_or_style="balanced",
        gradient_checkpointing=True,
        num_train_timesteps=1000,
        min_denoising_steps=0,
        max_denoising_steps=999,
        do_cfg=False,
        do_random_cfg=False,
        train_unet=True,
        train_text_encoder=False,
    )
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.sd = model
    trainer.train_config = train_config
    trainer.network = network
    trainer.network_config = network_config
    trainer.adapter = None
    trainer.adapter_config = None
    trainer.embedding = None
    trainer.embed_config = None
    trainer.decorator = None
    trainer.decorator_config = None
    trainer.guidance_config = None
    trainer.trigger_word = None
    trainer.do_long_prompts = False
    trainer.model_config = model.model_config
    trainer.device_torch = torch.device("cpu")
    trainer.is_caching_text_embeddings = False
    trainer.dataset_configs = [
        DatasetConfig(folder_path="chosen", rejected_path="rejected", buckets=True)
    ]
    trainer.additional_logs = {}
    trainer.accelerator = _Backward()
    trainer.timer = _Timer()
    return trainer


def _make_latent_batch(seed=200):
    chosen = torch.randn(
        BATCH_SIZE, *LATENT_SHAPE, generator=torch.Generator().manual_seed(seed)
    )
    rejected = torch.randn(
        BATCH_SIZE, *LATENT_SHAPE, generator=torch.Generator().manual_seed(seed + 1)
    )
    prompt_embeds = AdvancedPromptEmbeds(
        text_embeds=[tensor.clone() for tensor in _FakeTextEncoder().encode(
            ["chosen caption zero", "chosen caption one"]
        )]
    )
    batch = DataLoaderBatchDTO.__new__(DataLoaderBatchDTO)
    batch.file_items = [
        SimpleNamespace(
            is_reg=False,
            caption=f"chosen caption {i}",
            network_weight=1.0,
            loss_multiplier=1.0,
        )
        for i in range(BATCH_SIZE)
    ]
    batch.latents = chosen
    batch.rejected_latents = rejected
    batch.tensor = None
    batch.rejected_tensor = None
    batch.prompt_embeds = prompt_embeds
    batch.loss_multiplier_list = [1.0] * BATCH_SIZE
    batch.mask_tensor = None
    batch.unconditional_tensor = None
    batch.unconditional_latents = None
    batch.unaugmented_tensor = None
    batch.sigmas = None
    return batch


def _make_tensor_batch(seed=300):
    chosen = torch.randn(
        BATCH_SIZE,
        3,
        8,
        8,
        generator=torch.Generator().manual_seed(seed),
    )
    rejected = torch.randn(
        BATCH_SIZE,
        3,
        8,
        8,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    batch = DataLoaderBatchDTO.__new__(DataLoaderBatchDTO)
    batch.file_items = [
        SimpleNamespace(
            is_reg=False,
            caption=f"chosen caption {i}",
            network_weight=1.0,
            loss_multiplier=1.0,
        )
        for i in range(BATCH_SIZE)
    ]
    batch.latents = None
    batch.rejected_latents = None
    batch.tensor = chosen
    batch.rejected_tensor = rejected
    batch.prompt_embeds = None
    batch.loss_multiplier_list = [1.0] * BATCH_SIZE
    batch.mask_tensor = None
    batch.unconditional_tensor = None
    batch.unconditional_latents = None
    batch.unaugmented_tensor = None
    batch.sigmas = None
    return batch


def _parameter_snapshot(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _assert_base_frozen_and_unchanged(model, transformer, before):
    assert all(not parameter.requires_grad for parameter in transformer.parameters())
    assert all(parameter.grad is None for parameter in transformer.parameters())
    after = _parameter_snapshot(transformer)
    assert after.keys() == before.keys()
    for name in before:
        assert torch.equal(after[name], before[name]), f"frozen base changed: {name}"


def _per_image_errors(prediction, target):
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def _paired_preference_logits(trainer, batch, seed):
    """Replay the real paired forward and return one logit per preference pair.

    This measures the same-draw margin directly from the model instead of
    relying on a particular log-key spelling inside the trainer.
    """
    paired = copy(batch)
    if batch.latents is None:
        paired.tensor = torch.cat((batch.tensor, batch.rejected_tensor))
        paired.latents = None
    else:
        paired.latents = torch.cat((batch.latents, batch.rejected_latents))
        paired.tensor = None

    torch.manual_seed(seed)
    with torch.no_grad():
        noisy, noise, timesteps, prompts, _ = trainer.process_general_training_batch(
            paired, shared_pair_noise=True
        )
        target = trainer.sd.get_loss_target(noise=noise, batch=paired)
        embeds = batch.prompt_embeds
        if embeds is None:
            embeds = trainer.sd.encode_prompt(prompts)
        conditioning = concat_prompt_embeds([embeds, embeds])

        network = trainer.network
        previous_active, previous_multiplier = network.is_active, network.multiplier
        try:
            network.is_active = False
            reference = _per_image_errors(
                trainer.predict_noise(noisy, timesteps, conditioning), target
            )
            network.multiplier = batch.get_network_weight_list() * 2
            with network:
                policy = _per_image_errors(
                    trainer.predict_noise(noisy, timesteps, conditioning), target
                )
            chosen_delta, rejected_delta = (policy - reference).chunk(2)
            logits = 0.5 * trainer.train_config.dpo_beta * (rejected_delta - chosen_delta)
        finally:
            network.is_active = previous_active
            network.multiplier = previous_multiplier
    return logits.detach()


def test_dpo_trainer_acceptance_with_real_moss_cpu(monkeypatch):
    model, transformer = _make_model()
    network, network_config = _make_network(model, transformer)
    trainer = _make_trainer(model, transformer, network, network_config)
    validate_dpo_config(trainer)
    assert isinstance(model.noise_scheduler, CustomFlowMatchEulerDiscreteScheduler)

    batch = _make_latent_batch()
    assert batch.rejected_latents is not None
    base_before = _parameter_snapshot(transformer)
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    assert trainable

    # Observe the real scheduler redraw, the real MOSS prediction entry point,
    # and checkpoint recomputation without replacing any of them.
    process_calls = []
    real_process = trainer.process_general_training_batch

    def observe_process(batch_arg, **kwargs):
        result = real_process(batch_arg, **kwargs)
        process_calls.append(result)
        return result

    monkeypatch.setattr(trainer, "process_general_training_batch", observe_process)

    prediction_calls = []
    real_prediction = model.get_noise_prediction

    def observe_prediction(*args, **kwargs):
        prediction_calls.append(dict(kwargs))
        return real_prediction(*args, **kwargs)

    monkeypatch.setattr(model, "get_noise_prediction", observe_prediction)

    checkpoint_calls = []
    real_checkpoint = moss_transformer.checkpoint

    def observe_checkpoint(*args, **kwargs):
        checkpoint_calls.append(torch.is_grad_enabled())
        return real_checkpoint(*args, **kwargs)

    monkeypatch.setattr(moss_transformer, "checkpoint", observe_checkpoint)

    scheduler_calls = []
    real_set_train_timesteps = model.noise_scheduler.set_train_timesteps

    def observe_set_train_timesteps(*args, **kwargs):
        scheduler_calls.append(dict(kwargs))
        return real_set_train_timesteps(*args, **kwargs)

    monkeypatch.setattr(
        model.noise_scheduler, "set_train_timesteps", observe_set_train_timesteps
    )

    optimizer = torch.optim.SGD(trainable, lr=1e-2)
    initial_logits = _paired_preference_logits(trainer, batch, seed=4242)
    assert torch.equal(initial_logits, torch.zeros_like(initial_logits))
    initial_margin = initial_logits.mean().item()
    # The replay above is measurement only; inspect the actual training pass.
    process_calls.clear()
    prediction_calls.clear()
    checkpoint_calls.clear()
    scheduler_calls.clear()
    torch.manual_seed(4242)
    loss = SDTrainer.train_single_accumulation(trainer, batch, accum_scale=1.0)
    initial_loss = loss.detach()

    # Zero adapter means policy == reference: standard DPO starts at log(2).
    assert torch.isfinite(initial_loss)
    assert initial_margin == pytest.approx(0.0, abs=1e-7)
    assert initial_loss.item() == pytest.approx(math.log(2.0), abs=1e-6)
    assert checkpoint_calls, "gradient checkpointing did not recompute a block"
    assert any(checkpoint_calls), "checkpoint ran without an active training graph"
    assert len(scheduler_calls) == 1
    assert scheduler_calls[0]["timestep_type"] == "sigmoid"

    # One paired reference forward and one paired policy forward, each over
    # chosen then rejected, with chosen conditioning/noise/timestep repeated.
    assert len(prediction_calls) == 2
    for call in prediction_calls:
        assert call["latent_model_input"].shape[0] == 2 * BATCH_SIZE
        assert call["timestep"].reshape(-1).shape[0] == 2 * BATCH_SIZE
        assert len(call["text_embeddings"].text_embeds) == 2 * BATCH_SIZE
    first, second = prediction_calls
    assert torch.equal(first["latent_model_input"], second["latent_model_input"])
    assert torch.equal(first["timestep"], second["timestep"])
    first_embeds = first["text_embeddings"].text_embeds
    second_embeds = second["text_embeddings"].text_embeds
    assert all(torch.equal(a, b) for a, b in zip(first_embeds, second_embeds))
    chosen_embeds = batch.prompt_embeds.text_embeds
    for i in range(BATCH_SIZE):
        assert torch.equal(first_embeds[i], chosen_embeds[i])
        assert torch.equal(first_embeds[BATCH_SIZE + i], chosen_embeds[i])

    assert len(process_calls) == 1
    noisy_latents, noise, timesteps, prompts, _ = process_calls[0]
    assert noisy_latents.shape[0] == 2 * BATCH_SIZE
    assert noise.shape[0] == 2 * BATCH_SIZE
    assert timesteps.shape[0] == 2 * BATCH_SIZE
    assert torch.equal(noise[:BATCH_SIZE], noise[BATCH_SIZE:])
    assert torch.equal(timesteps[:BATCH_SIZE], timesteps[BATCH_SIZE:])
    assert len(prompts) == BATCH_SIZE

    # The adapter received a finite, non-zero gradient; the base did not move.
    gradients = [parameter.grad for parameter in trainable]
    assert any(
        gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in gradients
    )
    assert all(
        gradient is None or torch.isfinite(gradient).all() for gradient in gradients
    )
    _assert_base_frozen_and_unchanged(model, transformer, base_before)
    assert network.is_active is False
    assert network.multiplier == 1.0
    assert torch.is_grad_enabled()

    # Optimize on the same pair, then replay the exact same noise/timestep draw.
    adapter_before = _parameter_snapshot(network)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    adapter_after = _parameter_snapshot(network)
    assert any(
        not torch.equal(adapter_after[name], value) for name, value in adapter_before.items()
    )
    improved_logits = _paired_preference_logits(trainer, batch, seed=4242)
    assert torch.isfinite(improved_logits).all()
    assert improved_logits.mean().item() > initial_margin + 1e-4
    _assert_base_frozen_and_unchanged(model, transformer, base_before)
    assert network.is_active is False
    assert torch.is_grad_enabled()


def test_dpo_rejected_tensor_path_with_fake_vae_cpu():
    model, transformer = _make_model()
    network, network_config = _make_network(model, transformer)
    trainer = _make_trainer(model, transformer, network, network_config)
    validate_dpo_config(trainer)
    batch = _make_tensor_batch()
    assert batch.rejected_tensor is not None
    base_before = _parameter_snapshot(transformer)

    torch.manual_seed(5151)
    loss = SDTrainer.train_single_accumulation(trainer, batch, accum_scale=1.0)
    assert loss.item() == pytest.approx(math.log(2.0), abs=1e-6)
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in trainable
    )
    _assert_base_frozen_and_unchanged(model, transformer, base_before)
    assert network.is_active is False
    assert torch.is_grad_enabled()


def test_dpo_swap_flips_margin_and_batch_ordering():
    model, transformer = _make_model()
    network, network_config = _make_network(model, transformer)
    trainer = _make_trainer(model, transformer, network, network_config)
    validate_dpo_config(trainer)

    # Give the adapter a nonzero delta so the preference margin is informative.
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            if "lora_up" in name:
                parameter.normal_(0.0, 0.02)

    batch = _make_latent_batch(seed=707)
    logits = _paired_preference_logits(trainer, batch, seed=6060)
    assert torch.isfinite(logits).all()
    assert abs(logits.mean().item()) > 1e-5

    swapped = _make_latent_batch(seed=707)
    swapped.latents = batch.rejected_latents.clone()
    swapped.rejected_latents = batch.latents.clone()
    swapped.prompt_embeds = batch.prompt_embeds.clone()
    swapped_logits = _paired_preference_logits(trainer, swapped, seed=6060)

    # For a paired batch (chosen rows first, rejected second), swapping the
    # arms must flip each pair's implicit preference while preserving order.
    torch.testing.assert_close(swapped_logits, -logits, rtol=0.0, atol=1e-4)


def test_dpo_failure_restores_grad_mode_gradients_and_network_scope(monkeypatch):
    model, transformer = _make_model()
    network, network_config = _make_network(model, transformer)
    trainer = _make_trainer(model, transformer, network, network_config)
    validate_dpo_config(trainer)
    batch = _make_latent_batch(seed=808)

    # Existing accumulation gradients must not be disturbed by the failed pass.
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    sentinel = trainable[0]
    sentinel.grad = torch.ones_like(sentinel)
    previous_grad = sentinel.grad.detach().clone()

    network.is_active = False
    network.multiplier = 0.25
    trainer.accelerator = _Backward()
    trainer.accelerator.fail = True

    with pytest.raises(RuntimeError, match="injected DPO backward failure"):
        torch.manual_seed(9090)
        SDTrainer.train_single_accumulation(trainer, batch, accum_scale=1.0)

    assert trainer.accelerator.calls == 1
    assert network.is_active is False
    assert network.multiplier == 0.25
    assert torch.is_grad_enabled()
    assert torch.equal(sentinel.grad, previous_grad)
    assert all(parameter.grad is None for parameter in transformer.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
