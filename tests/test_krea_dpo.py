"""CPU acceptance contracts for tiny genuine Krea-2 Raw T2I DPO.

Real/native: SingleMMDiTConfig, SingleStreamDiT, Krea2Model.get_noise_prediction
and get_loss_target, ai-toolkit LoRASpecialNetwork, torch Adam, and
SDTrainer.train_single_accumulation/train_dpo_accumulation.

Synthetic and clearly marked: deterministic text embeddings and VAE placeholder;
the generate_single_image pipeline boundary is also a stub. No synthetic
diffusion model is used.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.krea2 import krea2 as krea2_module
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from extensions_built_in.diffusion_models.krea2.src.pipeline import (
    Krea2Pipeline,
    pad_text_features,
    predict_velocity,
)
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import DatasetConfig, NetworkConfig, TrainConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.diffusion_dpo import (
    paired_batch,
    prediction_errors,
    preference_logits,
    validate_dpo_config,
)
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.prompt_utils import concat_prompt_embeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

# Structurally native tiny Krea-2 MMDiT. headdim=16 gives three non-empty,
# even RoPE axes as SingleStreamDiT.__init__ requires.
TINY = SingleMMDiTConfig(
    features=64, tdim=16, txtdim=16, heads=4, kvheads=2, multiplier=2,
    layers=1, patch=2, channels=4, txtlayers=1, txtheads=4, txtkvheads=2,
)
BATCH_SIZE = 2
LATENT_SHAPE = (4, 4, 4)
TEXT_LENGTHS = [5, 7]
BETA = 100.0


class _Timer:
    """Same lightweight trainer timer contract used by test_moss_dpo_training."""

    def __call__(self, *args, **kwargs):
        return nullcontext()

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass


class _Backward:
    """Same trainer backward shim used by test_moss_dpo_training."""

    def __init__(self):
        self.calls = 0

    def backward(self, loss):
        self.calls += 1
        loss.backward()


class _SyntheticTextEncoder(torch.nn.Module):
    """SYNTHETIC Qwen3-VL stand-in: deterministic random vectors, no weights."""

    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def encode(self, prompts):
        out = []
        for i, prompt in enumerate(prompts):
            digest = sum(ord(ch) for ch in prompt)
            generator = torch.Generator().manual_seed(10_000 + digest + i)
            out.append(torch.randn(
                TEXT_LENGTHS[i % len(TEXT_LENGTHS)],
                TINY.txtlayers * TINY.txtdim,
                generator=generator,
            ))
        return out


class _SyntheticVAE(torch.nn.Module):
    """SYNTHETIC VAE stand-in; carries the shape/config contract only."""

    dtype = torch.float32

    class _Config:
        z_dim = TINY.channels
        latents_mean = [0.0] * TINY.channels
        latents_std = [1.0] * TINY.channels

    def __init__(self):
        super().__init__()
        self.config = self._Config()
        self.device = torch.device("cpu")


class _StubDecoderPipeline:
    """Boundary stub for generate_single_image's decoder/pipeline return path."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        cond = kwargs["conditional_embeds"]
        uncond = kwargs.get("unconditional_embeds")
        self.calls.append({
            "conditional_shapes": [tuple(x.shape) for x in cond.text_embeds],
            "unconditional_shapes": None if uncond is None else [
                tuple(x.shape) for x in uncond.text_embeds
            ],
            "height": kwargs.get("height"),
            "width": kwargs.get("width"),
        })
        from PIL import Image
        return [Image.new("RGB", (kwargs["width"], kwargs["height"]))]


def _make_model():
    torch.manual_seed(1)
    transformer = SingleStreamDiT(TINY)
    transformer.requires_grad_(False)
    transformer.train()

    model = Krea2Model.__new__(Krea2Model)
    model.model = transformer
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.is_flow_matching = True
    model.is_transformer = True
    model.is_multistage = False
    model.is_edit = False
    model.kv_cache = False
    model.arch = "krea2"
    model.patch_size = TINY.patch
    model.vae_scale_factor = 8
    model.max_text_length = 32
    model.model_config = SimpleNamespace(
        arch="krea2", model_kwargs={}, refiner_name_or_path=None, low_vram=False
    )
    model.use_old_lokr_format = False
    model.lora_keys_use_comfy_prefix = True
    model.target_lora_modules = ["SingleStreamDiT"]
    model.noise_scheduler = Krea2Model.get_train_scheduler()
    model.vae = _SyntheticVAE()
    model.text_encoder = _SyntheticTextEncoder()
    model.vae_torch_dtype = torch.float32
    model.te_torch_dtype = torch.float32
    model.x0_pred = False
    model.pipeline = Krea2Pipeline(model)
    return model, transformer


def _make_network(model, transformer):
    network_config = NetworkConfig(
        type="lora", linear=4, linear_alpha=4, transformer_only=True
    )
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
        target_lin_modules=["SingleStreamDiT"],
    )
    network.model_ref = model
    network.apply_to(None, transformer, False, True)
    network.to("cpu", dtype=torch.float32)
    return network, network_config


def _make_trainer(model, network, network_config):
    train_config = TrainConfig(
        loss_type="dpo",
        dpo_beta=BETA,
        noise_scheduler="flowmatch",
        loss_target="noise",
        timestep_type="sigmoid",
        content_or_style="balanced",
        gradient_checkpointing=False,
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
    trainer.device = torch.device("cpu")
    trainer.is_caching_text_embeddings = False
    trainer.dataset_configs = [
        DatasetConfig(folder_path="chosen", rejected_path="rejected", buckets=True)
    ]
    trainer.additional_logs = {}
    trainer.accelerator = _Backward()
    trainer.timer = _Timer()
    return trainer


def _make_latent_batch(model):
    chosen = torch.randn(
        BATCH_SIZE, *LATENT_SHAPE, generator=torch.Generator().manual_seed(200)
    )
    rejected = torch.randn(
        BATCH_SIZE, *LATENT_SHAPE, generator=torch.Generator().manual_seed(201)
    )
    captions = ["chosen caption zero", "chosen caption one"]
    batch = DataLoaderBatchDTO.__new__(DataLoaderBatchDTO)
    batch.file_items = [
        SimpleNamespace(
            is_reg=False, caption=captions[i], network_weight=1.0, loss_multiplier=1.0
        )
        for i in range(BATCH_SIZE)
    ]
    batch.latents = chosen
    batch.rejected_latents = rejected
    batch.tensor = None
    batch.rejected_tensor = None
    # SYNTHETIC TE output, deliberately variable length across the batch.
    batch.prompt_embeds = AdvancedPromptEmbeds(
        text_embeds=[x.clone() for x in model.text_encoder.encode(captions)]
    )
    batch.loss_multiplier_list = [1.0] * BATCH_SIZE
    batch.mask_tensor = None
    batch.unconditional_tensor = None
    batch.unconditional_latents = None
    batch.unaugmented_tensor = None
    batch.sigmas = None
    return batch


def _snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_unchanged(module, before):
    after = _snapshot(module)
    assert after.keys() == before.keys()
    for name in before:
        assert torch.equal(after[name], before[name]), f"frozen base changed: {name}"


def _fixed_pair_metrics(trainer, noisy, timesteps, conditioning, target):
    """Replay reference and policy on the exact same fixed noise/timestep."""
    network = trainer.network
    previous_active, previous_multiplier = network.is_active, network.multiplier
    try:
        with torch.no_grad():
            network.is_active = False
            reference = prediction_errors(
                trainer.predict_noise(noisy, timesteps, conditioning), target
            )
            network.multiplier = [1.0] * noisy.shape[0]
            with network:
                policy = prediction_errors(
                    trainer.predict_noise(noisy, timesteps, conditioning), target
                )
            logits = preference_logits(policy, reference, trainer.train_config.dpo_beta)
    finally:
        network.is_active = previous_active
        network.multiplier = previous_multiplier
    return reference.detach(), policy.detach(), logits.detach()


def test_krea2_t2i_dpo_cpu_acceptance():
    torch.set_num_threads(1)
    model, transformer = _make_model()
    network, network_config = _make_network(model, transformer)
    trainer = _make_trainer(model, network, network_config)
    validate_dpo_config(trainer)

    assert isinstance(model.noise_scheduler, CustomFlowMatchEulerDiscreteScheduler)
    assert type(model).__name__ == "Krea2Model"
    assert model.arch == "krea2"
    assert type(transformer).__name__ == "SingleStreamDiT"
    assert type(transformer.config).__name__ == "SingleMMDiTConfig"
    assert type(network).__name__ == "LoRASpecialNetwork"
    assert model.target_lora_modules == ["SingleStreamDiT"]

    batch = _make_latent_batch(model)
    cond = concat_prompt_embeds([batch.prompt_embeds, batch.prompt_embeds])
    assert [int(x.shape[0]) for x in cond.text_embeds] == TEXT_LENGTHS + TEXT_LENGTHS

    # AdvancedPromptEmbeds variable-length pair duplication reaches the native
    # Krea get_noise_prediction padding/mask path unchanged.
    latents = torch.randn(
        len(cond.text_embeds), *LATENT_SHAPE,
        generator=torch.Generator().manual_seed(33),
    )
    timesteps = torch.tensor([300.0, 600.0, 300.0, 600.0])
    captured = []
    real_pad = krea2_module.pad_text_features

    def observe_pad(features_list, device, dtype):
        context, mask = real_pad(features_list, device, dtype)
        captured.append(([int(x.shape[0]) for x in features_list], mask.clone()))
        return context, mask

    krea2_module.pad_text_features = observe_pad
    try:
        with torch.no_grad():
            prediction = model.get_noise_prediction(latents, timesteps, cond)
            context, mask = pad_text_features(cond.text_embeds, torch.device("cpu"), torch.float32)
            native_prediction = predict_velocity(
                model.transformer, latents, timesteps / 1000.0, context, mask
            )
    finally:
        krea2_module.pad_text_features = real_pad

    assert prediction.shape == latents.shape
    assert torch.isfinite(prediction).all()
    assert len(captured) == 1 and captured[0][0] == TEXT_LENGTHS + TEXT_LENGTHS
    expected_mask = [[1] * n + [0] * (max(TEXT_LENGTHS) - n) for n in TEXT_LENGTHS + TEXT_LENGTHS]
    assert captured[0][1].tolist() == expected_mask
    torch.testing.assert_close(prediction, native_prediction, atol=1e-6, rtol=1e-5)

    # Native generate_single_image entry: only the decoder/pipeline boundary is
    # stubbed. It must pass the single conditional embedding through with its
    # original (variable-length-aware) 2D shape.
    stub = _StubDecoderPipeline()
    cond_one = AdvancedPromptEmbeds(text_embeds=[batch.prompt_embeds.text_embeds[0]])
    uncond_one = AdvancedPromptEmbeds(text_embeds=[torch.zeros_like(cond_one.text_embeds[0])])
    gen_config = SimpleNamespace(
        width=32, height=32, guidance_scale=4.5, num_inference_steps=1,
        sigmas=None, latents=None, ctrl_img=None, ctrl_img_1=None,
        ctrl_img_2=None, ctrl_img_3=None,
    )
    with torch.no_grad():
        image = model.generate_single_image(
            stub, gen_config, cond_one, uncond_one, torch.Generator().manual_seed(7), {}
        )
    assert image.size == (32, 32)
    assert stub.calls[0]["conditional_shapes"] == [(TEXT_LENGTHS[0], TINY.txtlayers * TINY.txtdim)]
    assert stub.calls[0]["unconditional_shapes"] == [(TEXT_LENGTHS[0], TINY.txtlayers * TINY.txtdim)]

    base_before = _snapshot(transformer)
    trainable = [p for p in network.parameters() if p.requires_grad]
    assert trainable
    adapter_before = _snapshot(network)
    optimizer = torch.optim.Adam(trainable, lr=1e-2)

    process_calls = []
    prediction_calls = []
    real_process = trainer.process_general_training_batch
    real_prediction = model.get_noise_prediction

    def observe_process(batch_arg, **kwargs):
        result = real_process(batch_arg, **kwargs)
        process_calls.append(result)
        return result

    def observe_prediction(*args, **kwargs):
        prediction_calls.append((
            kwargs["latent_model_input"].detach().clone(),
            kwargs["timestep"].detach().clone(),
            [int(x.shape[0]) for x in kwargs["text_embeddings"].text_embeds],
        ))
        return real_prediction(*args, **kwargs)

    trainer.process_general_training_batch = observe_process
    model.get_noise_prediction = observe_prediction
    try:
        loss = SDTrainer.train_single_accumulation(trainer, batch, accum_scale=1.0)
    finally:
        trainer.process_general_training_batch = real_process
        model.get_noise_prediction = real_prediction

    # Zero LoRA means policy == reference, so the DPO loss starts at log(2).
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=1e-6)
    assert len(process_calls) == 1
    assert len(prediction_calls) == 2
    first, second = prediction_calls
    assert torch.equal(first[0], second[0])  # same fixed noisy latents
    assert torch.equal(first[1], second[1])  # same timestep
    assert first[2] == TEXT_LENGTHS + TEXT_LENGTHS

    noisy, noise, timesteps, _, _ = process_calls[0]
    paired = paired_batch(batch)
    target = model.get_loss_target(noise=noise, batch=paired)
    t01 = timesteps.to(torch.float32) / 1000.0
    torch.testing.assert_close(
        noisy,
        (1.0 - t01.view(-1, 1, 1, 1)) * paired.latents
        + t01.view(-1, 1, 1, 1) * noise,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(target, noise - paired.latents)
    assert torch.equal(noise[:BATCH_SIZE], noise[BATCH_SIZE:])
    assert torch.equal(timesteps[:BATCH_SIZE], timesteps[BATCH_SIZE:])

    initial_reference, initial_policy, initial_logits = _fixed_pair_metrics(
        trainer, noisy, timesteps, cond, target
    )
    assert initial_logits.mean().item() == pytest.approx(0.0, abs=1e-7)
    assert torch.equal(initial_reference, initial_policy)

    grads = [p.grad for p in trainable]
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert trainer.accelerator.calls == 1
    assert all(p.grad is None for p in transformer.parameters())
    _assert_unchanged(transformer, base_before)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    adapter_after = _snapshot(network)
    assert any(
        not torch.equal(adapter_after[name], value)
        for name, value in adapter_before.items()
    )
    _assert_unchanged(transformer, base_before)

    post_reference, _, post_logits = _fixed_pair_metrics(
        trainer, noisy, timesteps, cond, target
    )
    assert torch.isfinite(post_logits).all()
    assert post_logits.mean().item() > initial_logits.mean().item() + 1e-5
    assert torch.equal(initial_reference, post_reference)

    # Reverse chosen/rejected at the exact same noise draw. The same conditioned
    # policy must flip the margin sign and exactly negate the original logit.
    swapped_clean = torch.cat((batch.rejected_latents, batch.latents), dim=0)
    swapped_noise = torch.cat((noise[BATCH_SIZE:], noise[:BATCH_SIZE]), dim=0)
    swapped_noisy = (
        (1.0 - t01.view(-1, 1, 1, 1)) * swapped_clean
        + t01.view(-1, 1, 1, 1) * swapped_noise
    )
    swapped_target = swapped_noise - swapped_clean
    _, _, swapped_logits = _fixed_pair_metrics(
        trainer, swapped_noisy, timesteps, cond, swapped_target
    )
    assert swapped_logits.mean().item() < 0.0
    torch.testing.assert_close(swapped_logits, -post_logits, atol=1e-5, rtol=1e-5)
    assert torch.equal(initial_reference, post_reference)
    _assert_unchanged(transformer, base_before)

