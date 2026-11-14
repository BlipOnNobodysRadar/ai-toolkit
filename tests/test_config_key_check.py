"""Config keys that nothing reads are named at job start (toolkit/config_key_check.py)."""
import glob
import os
from pathlib import Path

import pytest
import yaml

from toolkit import config_modules as cm
from toolkit.config_key_check import check_sections, check_train_process, known_keys, unknown_keys

REPO = Path(__file__).resolve().parents[1]
CHECKED = (cm.TrainConfig, cm.NetworkConfig, cm.DatasetConfig, cm.ModelConfig, cm.SaveConfig, cm.SampleConfig,
           cm.LoggingConfig, cm.ValidationConfig, cm.EMAConfig)


class FakeProcess:
    def __init__(self, config):
        self.config = config

    def get_conf(self, key, default=None):
        return self.config.get(key, default)


@pytest.mark.parametrize("cls", CHECKED, ids=lambda c: c.__name__)
def test_every_checked_class_is_readable(cls):
    # None would mean the class uses its kwargs in a way the reader cannot follow, and the check silently skips it
    assert known_keys(cls)


def test_known_keys_come_from_the_class_source():
    assert {"lr", "steps", "optimizer", "optimizer_params", "lr_scheduler"} <= known_keys(cm.TrainConfig)
    assert {"type", "linear", "linear_alpha", "network_kwargs"} <= known_keys(cm.NetworkConfig)
    assert {"folder_path", "resolution", "caption_dropout_rate"} <= known_keys(cm.DatasetConfig)


def test_fork_options_are_known_where_they_belong():
    train = "allow_tf32 multires_noise_iterations multires_noise_discount ip_noise_gamma reseed_per_step " \
            "mask_resize_mode training_extras logit_normal_mean logit_normal_std logit_normal_uniform " \
            "logit_normal_cutoff noise_augmentation_seed validation_config"
    assert set(train.split()) <= known_keys(cm.TrainConfig)
    assert {"respect_alpha", "lora_plus_ratio", "lora_plus_text_encoder_ratio", "compute_dtype",
            "layer_offloading"} <= known_keys(cm.NetworkConfig)
    assert {"random_crop_percent", "bucket_tolerance", "isolate_modalities"} <= known_keys(cm.DatasetConfig)
    assert {"activation_offloading"} <= known_keys(cm.ModelConfig)
    assert {"datasets", "max_batches", "seed", "validation_sigmas", "validation_items",
            "validate_every_n_steps"} <= known_keys(cm.ValidationConfig)


def test_opaque_kwargs_use_is_not_guessed_at():
    class PassesItOn:
        def __init__(self, **kwargs):
            self.a = kwargs.get("a")
            self.rest = dict(kwargs)  # anything could be read from this later

    class LoopsOverIt:
        def __init__(self, **kwargs):
            for key in ("a", "b"):
                setattr(self, key, kwargs.get(key))

    assert known_keys(PassesItOn) is None and unknown_keys({"zzz": 1}, PassesItOn) == []
    assert known_keys(LoopsOverIt) is None and unknown_keys({"zzz": 1}, LoopsOverIt) == []


def test_base_class_keys_count():
    class Base:
        def __init__(self, **kwargs):
            self.a = kwargs.get("a")

    class Child(Base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)  # hands the dict on: not followable

    class Plain(Base):
        pass

    assert known_keys(Child) is None
    assert known_keys(Plain) == {"a"}


def test_typo_wrong_section_and_plain_unknown_are_each_named(monkeypatch):
    monkeypatch.delenv("AITK_STRICT_CONFIG", raising=False)
    lines = []
    process = FakeProcess({
        "network": {"type": "lora", "linear": 32, "lora_plus_ratio": 4},
        "train": {"steps": 10, "lora_plus_ratio": 4, "multires_noise_iteratons": 6, "zzz_not_a_thing": 1,
                  "validation_config": {"max_batchs": 5}},
        "model": {"name_or_path": "x", "model_kwargs": {"anything": "goes here"}},
        "datasets": [{"folder_path": "a"}, {"folder_path": "b", "isolate_modalitys": True}],
    })
    messages = check_train_process(process, log=lines.append)
    assert messages == [
        "train.lora_plus_ratio (a `network` option)",
        "train.multires_noise_iteratons (did you mean `multires_noise_iterations`?)",
        "train.zzz_not_a_thing",
        "train.validation_config.max_batchs (did you mean `max_batches`?)",
        "datasets[1].isolate_modalitys (did you mean `isolate_modalities`?)",
    ]
    assert len(lines) == 1 and "WARNING" in lines[0] and all(m in lines[0] for m in messages)


def test_clean_config_says_nothing(monkeypatch):
    monkeypatch.delenv("AITK_STRICT_CONFIG", raising=False)
    lines = []
    process = FakeProcess({"network": {"type": "lora", "linear": 32}, "train": {"steps": 10, "lr": 1e-4},
                           "datasets": [{"folder_path": "a", "resolution": [512, 768]}]})
    assert check_train_process(process, log=lines.append) == [] and lines == []


def test_strict_mode_raises(monkeypatch):
    monkeypatch.setenv("AITK_STRICT_CONFIG", "1")
    with pytest.raises(ValueError, match="train.stepz"):
        check_sections([("train", {"stepz": 10}, cm.TrainConfig)], log=lambda _: None)
    assert check_sections([("train", {"steps": 10}, cm.TrainConfig)], log=lambda _: None) == []


def _example_configs():
    return sorted(glob.glob(str(REPO / "config" / "examples" / "**" / "*.y*ml"), recursive=True))


@pytest.mark.parametrize("path", _example_configs(), ids=os.path.basename)
def test_repo_example_configs_have_no_unknown_keys(path, monkeypatch):
    """The false-positive guard: every example config shipped in the repo must pass clean."""
    monkeypatch.delenv("AITK_STRICT_CONFIG", raising=False)
    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError:
        pytest.skip("example file is not valid YAML")
    for proc in (doc.get("config") or {}).get("process") or []:
        if not isinstance(proc.get("datasets", []), list):
            continue
        assert check_train_process(FakeProcess(proc), log=lambda _: None) == []


def test_ui_default_job_config_has_no_unknown_keys(monkeypatch):
    """Keys of the UI's default job (ui/src/app/jobs/new/jobConfig.ts, helpers/defaultSamples.ts)."""
    monkeypatch.delenv("AITK_STRICT_CONFIG", raising=False)
    keys = lambda names: {name: None for name in names.split()}
    process = FakeProcess({
        "network": keys("type linear linear_alpha conv conv_alpha lokr_full_rank lokr_factor network_kwargs"),
        "save": keys("dtype save_every max_step_saves_to_keep save_format push_to_hub"),
        "datasets": [keys("folder_path mask_path mask_min_value default_caption caption_ext caption_dropout_rate "
                          "cache_latents_to_disk is_reg network_weight resolution controls shrink_video_to_frames "
                          "num_frames flip_x flip_y num_repeats")],
        "train": keys("batch_size bypass_guidance_embedding steps gradient_accumulation train_unet "
                      "train_text_encoder gradient_checkpointing noise_scheduler optimizer timestep_type "
                      "content_or_style optimizer_params unload_text_encoder cache_text_embeddings lr "
                      "skip_first_sample force_first_sample disable_sampling dtype diff_output_preservation "
                      "diff_output_preservation_multiplier diff_output_preservation_class switch_boundary_every "
                      "loss_type") | {"ema_config": keys("use_ema ema_decay")},
        "logging": keys("log_every use_ui_logger"),
        "model": keys("name_or_path quantize qtype quantize_te qtype_te arch low_vram model_kwargs compile"),
        "sample": keys("sampler sample_every sample_start_step width height samples neg seed walk_seed "
                       "guidance_scale sample_steps num_frames fps"),
    })
    assert check_train_process(process, log=lambda _: None) == []
