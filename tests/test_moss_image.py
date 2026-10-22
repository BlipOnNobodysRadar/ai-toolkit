"""MOSS-Image checks that need no weights: checkpoint naming, LoRA targeting and
key layout on a tiny transformer."""

import os

import huggingface_hub
import pytest

import toolkit.models.v2.resolver as resolver
from extensions_built_in.diffusion_models.moss_image.src import checkpoints


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "MODELS_PATH", str(tmp_path))

    def offline(*args, **kwargs):
        raise AssertionError("hub download attempted")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    return tmp_path


def _resolve(repo_id, qtype):
    return os.path.normpath(checkpoints.comfy_transformer_file(repo_id, qtype=qtype))


def _touch(root, rel):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return os.path.normpath(path)


@pytest.mark.parametrize(
    "repo_id, variant",
    [(checkpoints.BASE_REPO, "base"), (checkpoints.SFT_REPO, "sft"), (checkpoints.COMFY_REPO, "sft")],
)
def test_repack_int8_names_resolve_locally(models_dir, repo_id, variant):
    # the repack's int8 files lost their "_attn" suffix on 2026-10-22
    path = _touch(models_dir, f"diffusion_models/moss_image_{variant}_int8_convrot.safetensors")
    assert _resolve(repo_id, "convrot8") == path


def test_old_attn_names_still_resolve(models_dir):
    path = _touch(models_dir, "diffusion_models/moss_image_base_int8_convrot_attn.safetensors")
    assert _resolve(checkpoints.BASE_REPO, "convrot8") == path


def test_bf16_base_wins_without_convrot(models_dir):
    _touch(models_dir, "diffusion_models/moss_image_base_int8_convrot.safetensors")
    bf16 = _touch(models_dir, "diffusion_models/moss_image_base_bf16.safetensors")
    # int8 only stands in for bf16 when it attaches as-is (convrot requested)
    assert _resolve(checkpoints.BASE_REPO, None) == bf16
    assert _resolve(checkpoints.BASE_REPO, "qfloat8") == bf16


def test_repack_listing_names():
    # Comfy-Org/MOSS-Image files as listed by the hub API on 2026-10-22
    listed = {
        "diffusion_models/moss_image_base_bf16.safetensors",
        "diffusion_models/moss_image_base_int8_convrot.safetensors",
        "diffusion_models/moss_image_sft_bf16.safetensors",
        "diffusion_models/moss_image_sft_int8_convrot.safetensors",
        "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
    }
    for repo_files, _ in checkpoints.COMFY_TRANSFORMER_FILES.values():
        assert set(repo_files) <= listed
    assert set(checkpoints.VENDOR_MIRROR.values()) <= listed
    assert checkpoints.COMFY_TEXT_ENCODER_FILES[0][1] in listed


# ---- timestep_type 'logit_normal' (MOSS-Image pre-training distribution) ----

import math

import torch

from toolkit.config_modules import TrainConfig
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


def _scheduler():
    return CustomFlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000, use_dynamic_shifting=False)


def test_logit_normal_config_defaults_and_kwargs():
    cfg = TrainConfig()
    assert cfg.flow_timestep_kwargs("sigmoid") == {}
    assert cfg.flow_timestep_kwargs("shift") == {}
    cfg = TrainConfig(timestep_type="logit_normal", logit_normal_mean=1.09, logit_normal_std=0.95,
                      logit_normal_uniform=0.01)
    assert cfg.flow_timestep_kwargs(cfg.timestep_type) == {
        "logit_mean": 1.09, "logit_std": 0.95, "uniform_mix": 0.01, "cutoff": 1.0}


@pytest.mark.parametrize("bad", [{"logit_normal_std": 0}, {"logit_normal_std": -1},
                                 {"logit_normal_uniform": 1.5}, {"logit_normal_mean": float("nan")},
                                 {"logit_normal_cutoff": 0}, {"logit_normal_cutoff": 1.2}])
def test_logit_normal_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        TrainConfig(timestep_type="logit_normal", **bad)


def test_logit_normal_matches_moss_pretraining_median():
    torch.manual_seed(0)
    t = _scheduler().set_train_timesteps(20000, "cpu", timestep_type="logit_normal", logit_mean=1.09, logit_std=0.95)
    sigma = t / 1000
    assert torch.all(t[:-1] >= t[1:]), "sorted descending like the other types"
    assert 0 < sigma.min() and sigma.max() < 1
    # median of sigmoid(N(1.09, .95)) is sigmoid(1.09) = 0.748; shift 5 uniform would be 0.833
    assert abs(sigma.median().item() - 1 / (1 + math.exp(-1.09))) < 0.01


def test_logit_normal_cutoff_matches_tech_report_sampler():
    """Truncated at 0.95 and stretched: the SimpleTuner / KohakuMoss MOSS sampler (median sigma ~0.78)."""
    torch.manual_seed(4)
    sigma = _scheduler().set_train_timesteps(
        40000, "cpu", timestep_type="logit_normal", logit_mean=1.09, logit_std=0.95, cutoff=0.95) / 1000
    assert 0 <= sigma.min() and sigma.max() <= 1
    # reference: SimpleTuner's binned CDF over [0, 0.95] mapped onto the 1000-step grid
    edges = torch.linspace(0, 0.95, 1001, dtype=torch.float64)
    cdf = torch.special.ndtr((torch.logit(edges) - 1.09) / 0.95)
    cdf = cdf / cdf[-1]
    ref_median = (torch.searchsorted(cdf, torch.tensor(0.5, dtype=torch.float64)).item()) / 1000
    assert abs(sigma.median().item() - ref_median) < 0.01
    assert abs(ref_median - 0.78) < 0.01


def test_logit_normal_get_sigmas_matches_timesteps():
    # loss_target source / unaugmented read sigmas through get_sigmas(), keyed by timestep
    torch.manual_seed(5)
    sched = _scheduler()
    t = sched.set_train_timesteps(1000, "cpu", timestep_type="logit_normal", logit_mean=1.09, logit_std=0.95,
                                  cutoff=0.95, uniform_mix=0.01)
    picked = t[torch.tensor([0, 1, 500, 999])]
    sigma = sched.get_sigmas(picked, n_dim=1, dtype=torch.float32, device="cpu")
    assert torch.allclose(sigma, picked / 1000)
    assert t.max() <= 1000


def test_scheduler_overrides_accept_logit_normal_options():
    from extensions_built_in.diffusion_models.ming_image.ming_image import (
        MingImageFlowMatchScheduler, scheduler_config as ming_config)
    kwargs = TrainConfig(timestep_type="logit_normal", logit_normal_mean=1.09).flow_timestep_kwargs("logit_normal")
    t = MingImageFlowMatchScheduler(**ming_config).set_train_timesteps(
        200, "cpu", timestep_type="logit_normal", latents=torch.zeros(1, 16, 64, 64), patch_size=2, **kwargs)
    assert t.shape == (200,) and abs((t / 1000).median().item() - 0.748) < 0.06


def test_every_typed_set_train_timesteps_call_forwards_the_options():
    """Calls that pick a timestep_type must pass that type's options along."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    checked = 0
    for rel in ("jobs/process/BaseSDTrainProcess.py", "jobs/process/TrainSliderProcess.py", "toolkit/guidance.py"):
        src = (root / rel).read_text(encoding="utf-8")
        start = src.find("set_train_timesteps(")
        while start != -1:
            depth, end = 0, start + len("set_train_timesteps")
            for end in range(end, len(src)):
                depth += {"(": 1, ")": -1}.get(src[end], 0)
                if depth == 0:
                    break
            call = src[start:end]
            if "timestep_type=" in call:
                assert "flow_timestep_kwargs(timestep_type)" in call, f"{rel}: {call[:80]}"
                checked += 1
            start = src.find("set_train_timesteps(", end)
    assert checked == 3


def test_logit_normal_defaults_equal_sigmoid_distribution():
    torch.manual_seed(1)
    sched = _scheduler()
    a = sched.set_train_timesteps(20000, "cpu", timestep_type="logit_normal")
    b = sched.set_train_timesteps(20000, "cpu", timestep_type="sigmoid")
    for q in (0.1, 0.5, 0.9):
        assert abs(torch.quantile(a, q) - torch.quantile(b, q)).item() < 15


def test_logit_normal_uniform_mix():
    torch.manual_seed(2)
    sched = _scheduler()
    full = sched.set_train_timesteps(20000, "cpu", timestep_type="logit_normal", logit_mean=4.0, uniform_mix=1.0)
    assert abs(full.mean().item() / 1000 - 0.5) < 0.02  # all uniform
    some = sched.set_train_timesteps(20000, "cpu", timestep_type="logit_normal", logit_mean=4.0, uniform_mix=0.2)
    low = (some < 500).float().mean().item()  # N(4,1) puts ~0 mass below sigma 0.5; uniform puts half
    assert 0.08 < low < 0.12


def test_logit_normal_timesteps_drive_add_noise():
    torch.manual_seed(3)
    sched = _scheduler()
    t = sched.set_train_timesteps(1000, "cpu", timestep_type="logit_normal", logit_mean=1.09, logit_std=0.95)
    x, noise = torch.zeros(4, 2, 2, 2), torch.ones(4, 2, 2, 2)
    picked = t[torch.tensor([0, 250, 500, 999])]
    noisy = sched.add_noise(x, noise, picked.view(-1, 1, 1, 1))
    assert torch.allclose(noisy[:, 0, 0, 0], picked / 1000)


# ---- LoRA targeting and key layout on a tiny MOSS transformer ----

from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
from extensions_built_in.diffusion_models.moss_image.src.transformer import MossImageParams, MossImageTransformer
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork

TINY = MossImageParams(dim=64, ffn_dim=128, num_heads=2, num_layers=24, text_dim=32, freq_dim=16, latent_channels=8)


def _network(network_type="lora", network_kwargs=None, **config):
    """BaseSDTrainProcess's construction on a tiny MOSS DiT (network arguments come from the NetworkConfig,
    whose transformer_only defaults to True); LoRA keys convert like MossImageModel."""
    torch.manual_seed(0)
    transformer = MossImageTransformer(TINY)
    model = MossImageModel.__new__(MossImageModel)  # only its LoRA key conversions are used
    model.lora_keys_use_comfy_prefix = True
    model.use_old_lokr_format = False
    network_kwargs = {"ignore_if_contains": ["layers.22."], **(network_kwargs or {})}
    config = NetworkConfig(type=network_type, linear=4, linear_alpha=4, network_kwargs=network_kwargs, **config)
    network = LoRASpecialNetwork(
        text_encoder=None, unet=transformer, lora_dim=config.linear, alpha=config.linear_alpha, multiplier=1.0,
        train_unet=True, train_text_encoder=False, network_config=config, network_type=config.type,
        transformer_only=config.transformer_only, is_transformer=True, base_model=model,
        target_lin_modules=["MossImageTransformer"],
        **network_kwargs,
    )
    network.model_ref = model  # the network holds the model by weakref only
    network.apply_to(None, transformer, False, True)
    network.to("cpu", dtype=torch.float32)
    with torch.no_grad():  # "trained": non-zero up factors
        for name, p in network.named_parameters():
            if "lora_up" in name or "lokr_w1" in name:
                p.normal_(0, 0.02)
    return network


def _roundtrip(network, tmp_path, factory):
    path = str(tmp_path / "lora.safetensors")
    network.save_weights(path, dtype=torch.float32)
    saved = network.get_state_dict(dtype=torch.float32)
    fresh = factory()
    fresh.load_weights(path)
    again = fresh.get_state_dict(dtype=torch.float32)
    assert sorted(again) == sorted(saved)
    assert all(torch.equal(again[k], saved[k]) for k in saved)
    return saved


def test_default_targets_blocks_skip_22_and_unfuse_qkv(tmp_path):
    saved = _roundtrip(_network(), tmp_path, _network)
    assert saved and all(k.startswith("diffusion_model.layers.") for k in saved)
    assert not any(".layers.22." in k for k in saved)
    # q k v o up_gate down per block: kohya's 252 MOSS targets less block 22
    assert len({k.rsplit(".", 2)[0] for k in saved}) == 6 * (TINY.num_layers - 1)
    # q/k/v saved as ComfyUI's separate projections, sharing the down factor
    assert not any(".proj.qkv." in k for k in saved)
    for n in range(TINY.num_layers):
        if n != 22:
            q, k = (f"diffusion_model.layers.{n}.proj.{p}.lora_A.weight" for p in "qk")
            assert torch.equal(saved[q], saved[k])


def test_all_modules_adds_modulation_embedders_head_refiner(tmp_path):
    make = lambda: _network(transformer_only=False)
    saved = _roundtrip(make(), tmp_path, make)
    for name in ("shared_block_modulation", "time_embedding", "video_embedder", "final_linear_video",
                 "text_embedder.blocks.0.ffn_down"):
        assert any(f"diffusion_model.{name}." in k for k in saved), name


def test_ffn_down_only_targets_one_linear_per_block(tmp_path):
    make = lambda: _network(network_kwargs={"only_if_contains": ["proj.ffn_down"]})
    saved = _roundtrip(make(), tmp_path, make)
    modules = {k.rsplit(".", 2)[0] for k in saved}
    assert modules == {f"diffusion_model.layers.{n}.proj.ffn_down" for n in range(TINY.num_layers) if n != 22}


def test_full_rank_lokr_keeps_qkv_fused(tmp_path):
    make = lambda: _network(network_type="lokr", lokr_factor=4)
    saved = _roundtrip(make(), tmp_path, make)
    assert any(k.endswith(".proj.qkv.lokr_w1") for k in saved)
    assert not any(k.endswith("lokr_w2_a") for k in saved), "lokr_full_rank (default) stores w2 whole"
    assert not any(".layers.22." in k for k in saved)
