import json
from types import SimpleNamespace

import pytest
import torch

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from training_extras import toolkit_validation as tv


class _SchedulerConfig:
    num_train_timesteps = 1000


class _Scheduler:
    config = _SchedulerConfig()


class _Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))


class _FakeSD:
    def __init__(self):
        self.unet = _Unet()
        self.torch_dtype = torch.float32
        self.device_torch = torch.device("cpu")
        self.noise_scheduler = _Scheduler()
        self.is_flow_matching = True
        self.prediction_type = "flow"
        self.is_edit = False
        self.device = "cpu"

    def add_noise(self, original_samples, noise, timesteps):
        # Keep the fake prediction path deterministic and independent of the
        # scheduler details; the target hook below defines all expected errors.
        return torch.zeros_like(original_samples)

    def get_loss_target(self, *, noise, batch, timesteps):
        return torch.ones_like(noise)


class _FakeNetwork:
    def __init__(self, multiplier=7.0):
        self.is_active = False
        self.multiplier = multiplier

    def __enter__(self):
        self.is_active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.is_active = False


class _PolicyTrainer:
    def __init__(self, tmp_path):
        self.sd = _FakeSD()
        self.sd.unet.train()
        self.network = _FakeNetwork()
        self.device_torch = torch.device("cpu")
        self.step_num = 17
        self.save_root = str(tmp_path)
        self.additional_logs = {}
        self.train_config = SimpleNamespace(
            loss_type="dpo",
            dpo_beta=2.0,
            validation_config=SimpleNamespace(validation_sigmas=[0.25, 0.75]),
        )
        self._validation_cache = None

    def predict_noise(self, noisy_latents, timesteps, conditional_embeds):
        # Reference (LoRA off) has unit error. Policy is perfect on chosen and
        # one unit wrong on rejected, so the strict margin is beta / 2.
        pair_count = noisy_latents.shape[0] // 2
        if not self.network.is_active:
            return torch.zeros_like(noisy_latents)
        prediction = torch.zeros_like(noisy_latents)
        prediction[:pair_count] = 1.0
        return prediction


def _paired_cache(tmp_path, pair_count=2):
    embeds = AdvancedPromptEmbeds(
        text_embeds=[torch.ones(3, 4) * i for i in range(pair_count)]
    )
    batch = SimpleNamespace(
        file_items=[SimpleNamespace(path=f"pair-{i}.png") for i in range(pair_count)],
        latents=None,
        tensor=None,
        rejected_latents=None,
        rejected_tensor=None,
    )
    latents = torch.zeros(pair_count * 2, 1, 2, 2)
    noise = torch.zeros(pair_count, 1, 2, 2)
    return {
        "batch": batch,
        "latents": latents,
        "noise": noise,
        "embeds": embeds,
        "chosen_image_ids": ["a.png", "sub/b.png"][:pair_count],
    }


def test_paired_validation_reports_vectorized_per_pair_metrics(tmp_path):
    trainer = _PolicyTrainer(tmp_path)
    trainer._validation_cache = {
        "paired": True,
        "native_batches": [_paired_cache(tmp_path)],
    }

    tv.validate(trainer, phase="periodic")

    rows = [
        json.loads(line)
        for line in (tmp_path / "validation.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert len(rows[0]["raw_metrics"]) == 4  # two pairs x two sigmas
    by_pair = {}
    for row in rows[0]["raw_metrics"]:
        assert "chosen_image_id" in row
        assert "sigma" in row
        by_pair.setdefault(row["item"], set()).add(row["sigma"])
        assert row["policy_chosen_error"] == pytest.approx(0.0)
        assert row["policy_rejected_error"] == pytest.approx(1.0)
        assert row["reference_chosen_error"] == pytest.approx(1.0)
        assert row["reference_rejected_error"] == pytest.approx(1.0)
        assert row["margin"] == pytest.approx(1.0)
        assert row["dpo_loss"] == pytest.approx(torch.nn.functional.softplus(torch.tensor(-1.0)).item())
        assert row["strict_preference_correct"] is True
    assert by_pair == {0: {0.25, 0.75}, 1: {0.25, 0.75}}
    assert {r["chosen_image_id"] for r in rows[0]["raw_metrics"]} == {"a.png", "sub/b.png"}
    assert trainer.additional_logs["val/dpo/margin"] == pytest.approx(1.0)
    assert trainer.additional_logs["val/dpo/accuracy"] == pytest.approx(1.0)
    assert trainer.additional_logs["val/dpo/loss"] == pytest.approx(
        torch.nn.functional.softplus(torch.tensor(-1.0)).item()
    )
    assert trainer.additional_logs["val/dpo/accuracy/sigma_0.25"] == pytest.approx(1.0)
    # Per-pass state must be restored even though validation used eval mode.
    assert trainer.network.is_active is False
    assert trainer.network.multiplier == pytest.approx(7.0)
    assert trainer.sd.unet.training is True


def test_better_margin_can_hide_worse_chosen_error(tmp_path):
    trainer = _PolicyTrainer(tmp_path)
    cached = _paired_cache(tmp_path)
    cached["loss_multipliers"] = [0.0, 3.0]
    trainer._validation_cache = {"paired": True, "native_batches": [cached]}
    tv.validate(trainer)
    initial = trainer.additional_logs["val/loss"]
    assert "val/dpo/composite_loss" not in trainer.additional_logs
    trainer.train_config.dpo_anchor_weight = 2.0
    trainer.train_config.dpo_sft_weight = 0.25

    def predict(noisy, timesteps, conditioning):
        result = torch.zeros_like(noisy)
        if trainer.network.is_active:
            result[:2] = 3  # chosen error 4
            result[2:] = 4  # rejected error 9
        return result

    trainer.predict_noise = predict
    tv.validate(trainer)
    logs = trainer.additional_logs
    historical = torch.nn.functional.softplus(torch.tensor(-5.)).item()
    assert logs["val/loss"] == logs["val/dpo/loss"] == pytest.approx(historical)
    assert logs["val/loss"] < initial
    assert logs["val/dpo/chosen_delta"] == logs["val/dpo/chosen_drift_pos"] == 3
    assert logs["val/dpo/rejected_delta"] == 8
    assert logs["val/dpo/composite_loss"] == pytest.approx(1.5 * (historical + 2 * 3 + 0.25 * 4))


def _make_dpo_tree(root):
    chosen = root / "chosen"
    rejected = root / "rejected"
    chosen.mkdir(parents=True)
    rejected.mkdir(parents=True)
    (chosen / "a.png").touch()
    (rejected / "a.png").touch()
    return chosen, rejected


def _validate_cfg(chosen, rejected):
    return [{"folder_path": str(chosen), "rejected_path": str(rejected)}]


def test_paired_validation_config_rejects_invalid_combinations(tmp_path):
    train_chosen, train_rejected = _make_dpo_tree(tmp_path / "train")
    val_chosen, val_rejected = _make_dpo_tree(tmp_path / "val")
    other = tmp_path / "other"
    other.mkdir()
    trainer = SimpleNamespace(
        network=_FakeNetwork(),
        datasets=[
            SimpleNamespace(folder_path=str(train_chosen), rejected_path=str(train_rejected))
        ],
        train_config=SimpleNamespace(loss_type="dpo"),
    )

    cfg = SimpleNamespace(datasets=_validate_cfg(val_chosen, val_rejected))
    assert tv._validate_paired_validation_config(trainer, cfg) is True

    mixed = SimpleNamespace(
        datasets=_validate_cfg(val_chosen, val_rejected)
        + [{"folder_path": str(other)}]
    )
    with pytest.raises(ValueError, match="mix paired and unpaired"):
        tv._validate_paired_validation_config(trainer, mixed)

    trainer.train_config.loss_type = "mse"
    with pytest.raises(ValueError, match="requires train.loss_type: dpo"):
        tv._validate_paired_validation_config(trainer, cfg)
    trainer.train_config.loss_type = "dpo"

    controlled = SimpleNamespace(
        datasets=[{
            "folder_path": str(val_chosen),
            "rejected_path": str(val_rejected),
            "control_path": str(other),
        }]
    )
    with pytest.raises(ValueError, match="unsupported dataset option"):
        tv._validate_paired_validation_config(trainer, controlled)

    overlap = SimpleNamespace(datasets=_validate_cfg(train_chosen, train_rejected))
    with pytest.raises(ValueError, match="overlaps a training root"):
        tv._validate_paired_validation_config(trainer, overlap)


class _SetupTrainer:
    def __init__(self, tmp_path, cfg, chosen, rejected):
        self.sd = _FakeSD()
        self.sd.set_device_state_preset = lambda *a, **k: None
        self.sd.restore_device_state = lambda: None
        self.sd.text_encoder_to = lambda device: None
        self.sd.encode_prompt = lambda prompts: AdvancedPromptEmbeds(
            text_embeds=[torch.ones(2, 4) for _ in prompts]
        )
        self.network = _FakeNetwork()
        self.device_torch = torch.device("cpu")
        self.train_config = SimpleNamespace(
            validation_config=cfg,
            train_text_encoder=False,
            loss_type="dpo",
            unload_text_encoder=False,
        )
        self.datasets = [SimpleNamespace(
            folder_path=str(tmp_path / "train" / "chosen"),
            rejected_path=str(tmp_path / "train" / "rejected"),
        )]
        self.datasets_reg = None
        self.process_calls = []
        self._validation_cache = None

    def process_general_training_batch(self, batch, *, shared_pair_noise=False):
        self.process_calls.append(shared_pair_noise)
        pair_count = len(batch.file_items)
        noise = torch.arange(pair_count, dtype=torch.float32).reshape(pair_count, 1, 1, 1)
        noise = torch.cat((noise, noise), dim=0)
        timesteps = torch.cat((
            torch.full((pair_count,), 250.0),
            torch.full((pair_count,), 250.0),
        ))
        assert batch.latents.shape[0] == pair_count * 2
        prompts = [item.caption for item in batch.file_items]
        return torch.zeros_like(batch.latents), noise, timesteps, prompts, None


def test_setup_caches_paired_validation_without_touching_plain_path(tmp_path, monkeypatch):
    train_chosen, train_rejected = _make_dpo_tree(tmp_path / "train")
    val_chosen, val_rejected = _make_dpo_tree(tmp_path / "val")
    cfg = SimpleNamespace(
        datasets=_validate_cfg(val_chosen, val_rejected),
        seed=42,
        max_batches=5,
    )
    trainer = _SetupTrainer(tmp_path, cfg, val_chosen, val_rejected)

    pair_count = 2
    batch = SimpleNamespace(
        file_items=[
            SimpleNamespace(
                path=str(val_chosen / "a.png"),
                dataset_root=str(val_chosen),
                caption="same chosen caption",
                rejected_item=SimpleNamespace(),
            ),
            SimpleNamespace(
                path=str(val_chosen / "sub" / "b.png"),
                dataset_root=str(val_chosen),
                caption="another chosen caption",
                rejected_item=SimpleNamespace(),
            ),
        ],
        latents=torch.zeros(pair_count, 1, 2, 2),
        rejected_latents=torch.ones(pair_count, 1, 2, 2),
        loss_multiplier_list=[2.0, 0.0],
        tensor=None,
        rejected_tensor=None,
    )

    import toolkit.data_loader as data_loader
    monkeypatch.setattr(
        data_loader,
        "get_dataloader_from_datasets",
        lambda configs, batch_size, sd: [batch],
    )

    tv.setup(trainer)

    assert trainer.process_calls == [True]
    cache = trainer._validation_cache
    assert cache["paired"] is True
    item = cache["native_batches"][0]
    assert item["latents"].shape[0] == pair_count * 2
    assert item["loss_multipliers"] == [2.0, 0.0]
    assert item["noise"].shape[0] == pair_count
    assert item["chosen_image_ids"] == ["a.png", "sub/b.png"]
    assert len(item["embeds"].text_embeds) == pair_count


class _PlainSD(_FakeSD):
    def predict_noise(self, **kwargs):
        return torch.zeros_like(kwargs["latents"])


def _plain_optimizer():
    return SimpleNamespace(eval=lambda: None, train=lambda: None)


def test_plain_validation_keeps_its_raw_metric_schema(tmp_path):
    trainer = SimpleNamespace(
        sd=_PlainSD(),
        network=None,
        optimizer=_plain_optimizer(),
        device_torch=torch.device("cpu"),
        step_num=5,
        save_root=str(tmp_path),
        additional_logs={},
        train_config=SimpleNamespace(
            validation_config=SimpleNamespace(validation_sigmas=[0.5]),
            cfg_scale=1.0,
            bypass_guidance_embedding=False,
        ),
    )
    batch = SimpleNamespace(
        file_items=[SimpleNamespace(path="plain.png")],
        latents=torch.zeros(1, 1, 2, 2),
        tensor=None,
    )
    embeds = AdvancedPromptEmbeds(text_embeds=[torch.ones(2, 4)])
    trainer._validation_cache = {
        "native_batches": [(batch, torch.zeros(1, 1, 2, 2), embeds)]
    }

    tv.validate(trainer)

    record = json.loads((tmp_path / "validation.jsonl").read_text())
    row = record["raw_metrics"][0]
    assert set(row) == {"sigma", "item", "video_mse"}
    assert row["video_mse"] == pytest.approx(1.0)
    assert "val/dpo/loss" not in trainer.additional_logs
def test_paired_validation_restores_network_state_on_failure(tmp_path):
    trainer = _PolicyTrainer(tmp_path)
    trainer._validation_cache = {
        "paired": True,
        "native_batches": [_paired_cache(tmp_path)],
    }
    calls = {"count": 0}

    def fail_after_reference(noisy_latents, timesteps, conditional_embeds):
        calls["count"] += 1
        # First call is the no-grad reference pass; fail the policy pass.
        if calls["count"] == 1:
            return torch.zeros_like(noisy_latents)
        raise RuntimeError("injected policy validation failure")

    trainer.predict_noise = fail_after_reference
    with pytest.raises(RuntimeError, match="injected policy validation failure"):
        tv.validate(trainer)

    assert trainer.network.is_active is False
    assert trainer.network.multiplier == pytest.approx(7.0)
    assert trainer.sd.unet.training is True
