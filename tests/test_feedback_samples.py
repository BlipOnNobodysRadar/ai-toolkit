import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from safetensors.torch import load_file

import toolkit.feedback_samples as feedback_samples
import toolkit.feedback_training as feedback_training
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import SampleConfig


REFERENCE_ID = "0123456789abcdef" * 4


def _embedding(value):
    return AdvancedPromptEmbeds(text_embeds=[torch.tensor([[float(value)]])])


class FakeSD:
    def __init__(self, cache, fail_on_call=None):
        self.sample_prompts_cache = cache
        self.text_encoder = None
        self.calls = []
        self.fail_on_call = fail_on_call
        self.vae = None

    def _color(self, seed):
        return (seed % 256, (seed * 3) % 256, (seed * 7) % 256)

    def generate_images(self, configs, sampler=None):
        self.calls.append([config.prompt for config in configs])
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected second-pass generation failure")
        saved = []
        for index, config in enumerate(configs):
            if hasattr(config, "feedback_conditioning"):
                config.feedback_conditioning = self.sample_prompts_cache[index]
            config.set_gen_time()
            path = config.get_image_path(index)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.new("RGB", (config.width, config.height), self._color(config.seed)).save(path)
            time.sleep(0.002)
            saved.append(path)
        return saved

    def encode_images(self, images):
        assert len(images) == 2
        assert all(tuple(image.shape) == (3, 64, 64) for image in images)
        means = torch.stack([image.mean() for image in images])
        latents = torch.zeros((2, 4, 2, 2), dtype=torch.float32)
        for i in range(2):
            latents[i].fill_(means[i].item())
        return latents


def _trainer(tmp_path, cache, *, fail_on_call=None, feedback=None):
    save_root = tmp_path / "job"
    sample_config = SampleConfig(
        prompts=["first prompt", "second prompt"],
        width=64,
        height=64,
        format="jpg",
        sample_every=10,
        samples=[
            {"prompt": "first prompt", "width": 64, "height": 64, "seed": 101},
            {"prompt": "second prompt", "width": 64, "height": 64, "seed": 202},
        ],
    )
    sd = FakeSD(cache, fail_on_call=fail_on_call)
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(is_main_process=True),
        save_root=str(save_root),
        sample_config=sample_config,
        first_sample_config=sample_config,
        adapter_config=None,
        embedding=None,
        trigger_word=None,
        adapter=None,
        ema=None,
        logger=None,
        sd=sd,
        train_config=SimpleNamespace(
            dpo_feedback={"every": 10, "start_after": 0} if feedback is None else feedback,
            gradient_accumulation_steps=1,
        ),
        model_config=SimpleNamespace(arch="moss_image", model_kwargs={}),
        post_process_generate_image_config_list=lambda configs: configs,
        _checkpoint_completed_step=10,
        _feedback_schedule=SimpleNamespace(updates=7),
    )
    if trainer.train_config.dpo_feedback is None:
        trainer._feedback_schedule = None
    return trainer


def _pair_metadata(pair_dir):
    return json.loads((pair_dir / "pair.json").read_text(encoding="utf-8"))


def test_base_sample_publishes_two_prompts_with_cache_ordering(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_training, "feedback_reference_id", lambda trainer: REFERENCE_ID)
    monkeypatch.setattr(feedback_samples.secrets, "randbelow", lambda n: 0)
    cache = [_embedding(1), _embedding(2)]
    trainer = _trainer(tmp_path, cache)

    BaseSDTrainProcess.sample(trainer, 10)

    # The ordinary sample pass is untouched and the native second pass has the
    # same prompt order, so cache index i stays bound to ordinary sample i.
    assert len(trainer.sd.calls) == 2
    assert trainer.sd.calls[0] == trainer.sd.calls[1] == ["first prompt", "second prompt"]

    ordinary = sorted((Path(trainer.save_root) / "samples").glob("*.jpg"))
    assert ordinary
    preferences = Path(trainer.save_root) / "preferences"
    pair_dirs = sorted(p for p in preferences.iterdir() if p.is_dir() and re.fullmatch(r"[0-9a-f]{32}", p.name))
    assert len(pair_dirs) == 2

    expected = {}
    for pair_dir in pair_dirs:
        assert {p.name for p in pair_dir.iterdir()} == set(feedback_samples.PAIR_FILES)
        meta = _pair_metadata(pair_dir)
        assert meta["version"] == 1
        assert meta["id"] == pair_dir.name
        assert meta["step"] == 10
        assert meta["optimizer_updates"] == 7
        assert meta["reference_id"] == REFERENCE_ID
        assert re.fullmatch(r"[0-9a-f]{32}", meta["id"])
        assert meta["seeds"][0] != meta["seeds"][1]
        datetime.fromisoformat(meta["created_at"].replace("Z", "+00:00")).astimezone(timezone.utc)
        with Image.open(pair_dir / "a.png") as a_image, Image.open(pair_dir / "b.png") as b_image:
            assert a_image.mode == "RGB" and b_image.mode == "RGB"
            assert a_image.size == b_image.size == (meta["width"], meta["height"]) == (64, 64)
            assert a_image.getpixel((0, 0)) != b_image.getpixel((0, 0))
        latents = load_file(str(pair_dir / "latents.safetensors"))
        assert set(latents) == {"a", "b"}
        assert latents["a"].shape == latents["b"].shape == (1, 4, 2, 2)
        assert torch.isfinite(latents["a"]).all() and torch.isfinite(latents["b"]).all()
        emb = AdvancedPromptEmbeds.load(str(pair_dir / "embeddings.safetensors"))
        expected[meta["prompt"]] = int(emb["text_embeds"][0].item())

    assert expected == {"first prompt": 1, "second prompt": 2}
    assert not any(p.name == "vote.json" for p in preferences.rglob("vote.json"))
    assert not list(preferences.glob(".staging-*"))


def test_sample_settings_and_seeds_match_between_views(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_training, "feedback_reference_id", lambda trainer: REFERENCE_ID)
    cache = [_embedding(3), _embedding(4)]
    trainer = _trainer(tmp_path, cache)

    seen = []
    real_generate = trainer.sd.generate_images

    def generate(configs, sampler=None):
        seen.append(configs)
        return real_generate(configs, sampler=sampler)

    trainer.sd.generate_images = generate
    BaseSDTrainProcess.sample(trainer, 10)

    first, second = seen
    assert len(first) == len(second) == 2
    for a, b in zip(first, second):
        assert b.seed != a.seed
        assert b.output_ext == a.output_ext
        assert b.width == a.width and b.height == a.height
        assert b.guidance_scale == a.guidance_scale
        assert b.num_inference_steps == a.num_inference_steps
        assert b.network_multiplier == a.network_multiplier

    pair_dirs = [p for p in (Path(trainer.save_root) / "preferences").iterdir() if p.is_dir()]
    by_prompt = {_pair_metadata(p)["prompt"]: _pair_metadata(p) for p in pair_dirs}
    assert by_prompt["first prompt"]["seeds"] == [101, second[0].seed]
    assert by_prompt["second prompt"]["seeds"] == [202, second[1].seed]


def test_failure_is_isolated_and_retry_publishes_complete_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_training, "feedback_reference_id", lambda trainer: REFERENCE_ID)
    cache = [_embedding(1), _embedding(2)]
    trainer = _trainer(tmp_path, cache, fail_on_call=2)

    BaseSDTrainProcess.sample(trainer, 10)

    preferences = Path(trainer.save_root) / "preferences"
    assert preferences.is_dir()
    assert not [p for p in preferences.iterdir() if p.is_dir()]
    assert not list(preferences.glob(".staging-*"))
    assert list((Path(trainer.save_root) / "samples").glob("*.jpg"))

    trainer.sd.fail_on_call = None
    BaseSDTrainProcess.sample(trainer, 10)
    pair_dirs = [p for p in preferences.iterdir() if p.is_dir()]
    assert len(pair_dirs) == 2
    for pair_dir in pair_dirs:
        assert {p.name for p in pair_dir.iterdir()} == set(feedback_samples.PAIR_FILES)
    assert not list(preferences.glob(".staging-*"))


def test_embedding_failure_never_leaves_partial_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_training, "feedback_reference_id", lambda trainer: REFERENCE_ID)
    cache = [_embedding(1), _embedding(2)]
    trainer = _trainer(tmp_path, cache)
    original_save = AdvancedPromptEmbeds.save

    def fail_save(self, path):
        raise RuntimeError("injected embedding save failure")

    monkeypatch.setattr(AdvancedPromptEmbeds, "save", fail_save)
    BaseSDTrainProcess.sample(trainer, 10)

    preferences = Path(trainer.save_root) / "preferences"
    assert not [p for p in preferences.iterdir() if p.is_dir()]
    assert not list(preferences.glob(".staging-*"))
    assert list((Path(trainer.save_root) / "samples").glob("*.jpg"))
    monkeypatch.setattr(AdvancedPromptEmbeds, "save", original_save)


def test_disabled_feedback_does_not_second_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_training, "feedback_reference_id", lambda trainer: REFERENCE_ID)
    trainer = _trainer(tmp_path, [_embedding(1), _embedding(2)], feedback=None)
    trainer.train_config.dpo_feedback = None
    trainer._feedback_schedule = None

    BaseSDTrainProcess.sample(trainer, 10)

    assert len(trainer.sd.calls) == 1
    preferences = Path(trainer.save_root) / "preferences"
    assert not preferences.exists() or not [p for p in preferences.iterdir() if p.is_dir()]


def test_later_publish_failure_preserves_visible_vote(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_training, "feedback_reference_id", lambda trainer: REFERENCE_ID)
    trainer = _trainer(tmp_path, [_embedding(1), _embedding(2)])
    rename = os.rename
    seen = []

    def publish(source, destination):
        if seen:
            raise OSError("injected second rename failure")
        rename(source, destination)
        seen.append(Path(destination))
        (seen[0] / "vote.json").write_text('{"choice":"a"}')

    monkeypatch.setattr(feedback_samples.os, "rename", publish)
    BaseSDTrainProcess.sample(trainer, 10)
    assert len(seen) == 1
    assert (seen[0] / "vote.json").exists()
    assert not list(seen[0].parent.glob(".staging-*"))


def test_published_pairs_pass_the_consumer_boundary(tmp_path):
    # Use the real feedback_reference_id and feedback_training scanner/loader.
    trainer = _trainer(tmp_path, [_embedding(1), _embedding(2)])
    BaseSDTrainProcess.sample(trainer, 10)
    preferences = Path(trainer.save_root) / "preferences"
    pair_dirs = [p for p in preferences.iterdir() if p.is_dir()]
    assert len(pair_dirs) == 2
    for pair_dir in pair_dirs:
        (pair_dir / "vote.json").write_text(
            json.dumps({
                "version": 1,
                "id": pair_dir.name,
                "choice": "a",
                "voted_at": "2026-11-18T00:00:00Z",
            }),
            encoding="utf-8",
        )
    eligible = feedback_training.voted_feedback_ids(trainer)
    assert eligible == sorted(pair_dir.name for pair_dir in pair_dirs)
    batch = feedback_training.load_feedback_batch(trainer, eligible[0])
    assert batch.latents.shape == batch.rejected_latents.shape == (1, 4, 2, 2)
    assert batch.feedback_step == 10
    assert list(batch.feedback_seeds) == _pair_metadata(preferences / eligible[0])["seeds"]


def test_second_sample_uses_the_same_ema_view_and_restores_training_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_training, "feedback_reference_id", lambda trainer: REFERENCE_ID)
    trainer = _trainer(tmp_path, [_embedding(1), _embedding(2)])

    class FakeEMA:
        def __init__(self):
            self._is_train_mode = True
            self.seen = []

        def eval(self):
            self._is_train_mode = False

        def train(self):
            self._is_train_mode = True

    ema = FakeEMA()
    trainer.ema = ema
    trainer.sd.trainer = trainer
    real_generate = trainer.sd.generate_images

    def generate(configs, sampler=None):
        ema.seen.append(ema._is_train_mode)
        return real_generate(configs, sampler=sampler)

    trainer.sd.generate_images = generate
    BaseSDTrainProcess.sample(trainer, 10)

    assert ema.seen == [False, False]
    assert ema._is_train_mode is True
    assert len([p for p in (Path(trainer.save_root) / "preferences").iterdir() if p.is_dir()]) == 2
