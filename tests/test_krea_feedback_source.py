"""CPU contracts for Krea 2 native sample capture and frozen asset identity."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.krea2 import krea2 as k
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import ModelConfig


def _model(tmp_path, **model_kwargs):
    config = ModelConfig(
        name_or_path=str(tmp_path / "weights"),
        arch="krea2",
        dtype="float16",
        vae_dtype="float16",
        te_dtype="float16",
        model_kwargs=model_kwargs,
    )
    model = k.Krea2Model("cpu", config, dtype="float16")
    model.track_feedback_sources = True
    return model


def _embeds(value: float, *, requires_grad: bool = False):
    tensor = torch.tensor([[value, value + 1.0]], dtype=torch.float32)
    if requires_grad:
        tensor.requires_grad_(True)
    return AdvancedPromptEmbeds(text_embeds=[tensor])


def _gen_config(*, active: bool):
    values = dict(
        width=64,
        height=64,
        guidance_scale=1.0,
        num_inference_steps=2,
        sigmas=None,
        latents=None,
        ctrl_img=None,
        ctrl_img_1=None,
        ctrl_img_2=None,
        ctrl_img_3=None,
    )
    if active:
        values["feedback_conditioning"] = None
    return SimpleNamespace(**values)


class _Pipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [torch.zeros(1, 3, 8, 8)]


def test_generate_captures_exact_positive_conditioning_only_when_active(tmp_path):
    model = _model(tmp_path)
    model.model = SimpleNamespace(
        device=torch.device("cpu"),
        to=lambda *args, **kwargs: model.model,
    )
    positive = _embeds(3.0, requires_grad=True)
    negative = _embeds(-7.0, requires_grad=True)
    pipeline = _Pipeline()
    config = _gen_config(active=True)

    model.generate_single_image(
        pipeline, config, positive, negative, torch.Generator(), {}
    )

    captured = config.feedback_conditioning
    assert isinstance(captured, AdvancedPromptEmbeds)
    assert captured is not positive
    assert captured is not negative
    assert captured.keys() == positive.keys()
    assert captured.text_embeds[0].requires_grad is False
    assert torch.equal(captured.text_embeds[0], positive.text_embeds[0])
    assert not torch.equal(captured.text_embeds[0], negative.text_embeds[0])


def test_generate_does_not_capture_without_the_active_marker(tmp_path):
    model = _model(tmp_path)
    model.model = SimpleNamespace(
        device=torch.device("cpu"),
        to=lambda *args, **kwargs: model.model,
    )
    pipeline = _Pipeline()
    config = _gen_config(active=False)

    model.generate_single_image(
        pipeline, config, _embeds(1.0), _embeds(2.0), torch.Generator(), {}
    )

    assert not hasattr(config, "feedback_conditioning")
    assert len(pipeline.calls) == 1


def test_capture_happens_before_any_model_move(tmp_path):
    model = _model(tmp_path)

    class RefuseMove:
        device = torch.device("cpu")

        def to(self, *args, **kwargs):
            raise RuntimeError("model move should be after capture")

    model.model = RefuseMove()
    config = _gen_config(active=True)
    positive = _embeds(5.0, requires_grad=True)

    with pytest.raises(RuntimeError, match="after capture"):
        model.generate_single_image(
            _Pipeline(), config, positive, _embeds(6.0), torch.Generator(), {}
        )

    assert isinstance(config.feedback_conditioning, AdvancedPromptEmbeds)
    assert torch.equal(
        config.feedback_conditioning.text_embeds[0], positive.text_embeds[0]
    )


def test_resolve_mmdit_file_uses_directory_checkpoint_filename(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "nested" / "second.safetensors"
    third = tmp_path / "third.safetensors"
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    third.write_bytes(b"third")

    assert k._resolve_mmdit_file(str(tmp_path), "nested/second.safetensors") == os.path.realpath(
        second
    )
    with pytest.raises(FileNotFoundError, match="checkpoint_filename"):
        k._resolve_mmdit_file(str(tmp_path), None)


def test_resolve_mmdit_file_hub_uses_existing_default_filename(tmp_path, monkeypatch):
    downloaded = tmp_path / "cache" / "raw.safetensors"
    downloaded.parent.mkdir(parents=True)
    downloaded.write_bytes(b"hub fixture")
    calls = []

    def fake_download(*, repo_id, filename, token=None):
        calls.append((repo_id, filename, token))
        return str(downloaded)

    monkeypatch.setattr(k.huggingface_hub, "hf_hub_download", fake_download)
    assert k._resolve_mmdit_file("krea/Krea-2-Raw", None) == os.path.realpath(downloaded)
    assert calls[-1][:2] == ("krea/Krea-2-Raw", "raw.safetensors")


def test_unpinned_hub_components_identify_actual_loaded_contents():
    weight = torch.nn.Linear(2, 1)
    get_weight = lambda: k._loaded_source("org/repo", weight, "weights")
    first = get_weight()
    with torch.no_grad():
        weight.weight[0, 0] += 1
    assert get_weight() != first
    config = SimpleNamespace(config={"layers": 2, "_name_or_path": "cache1"})
    old = k._loaded_source("org/repo", config, "config")
    config.config["_name_or_path"] = "cache2"
    assert k._loaded_source("org/repo", config, "config") == old
    config.config["layers"] = 3
    assert k._loaded_source("org/repo", config, "config") != old
    vocab = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: "a"),
                            special_tokens_map={}, padding_side="left", truncation_side="right")
    original = k._loaded_source("org/repo", vocab, "tokenizer")
    vocab.backend_tokenizer.to_str = lambda: "b"
    assert k._loaded_source("org/repo", vocab, "tokenizer") != original


def test_reference_metadata_is_resolved_and_excludes_transient_kwargs(tmp_path):
    model = _model(
        tmp_path,
        mmdit_config={"layers": 3, "features": 128},
        sample_shift=7.0,
        checkpoint_filename="weights.safetensors",
        max_text_length=77,
        name_or_path="ignored/by/this/test",
    )
    metadata = model.feedback_reference_metadata

    assert metadata["mmdit_config"]["layers"] == 3
    assert metadata["mmdit_config"]["features"] == 128
    assert metadata["mmdit_config"]["tdim"] == k.KREA2_MMDIT_CONFIG["tdim"]
    assert metadata["selected_feature_layers"] == list(k.SELECT_LAYERS)
    assert metadata["prompt_preprocess"]["name"] == "krea2-qwen3vl-assistant"
    assert metadata["prompt_preprocess"]["start_idx"] == k.PROMPT_TEMPLATE_ENCODE_START_IDX
    assert metadata["prompt_preprocess"]["max_text_length"] == 77

    rendered = repr(metadata)
    assert "sample_shift" not in rendered
    assert "checkpoint_filename" not in rendered
    assert "name_or_path" not in rendered
    assert "num_inference_steps" not in rendered


def test_explicit_default_mmdit_config_matches_implicit_default(tmp_path):
    implicit = _model(tmp_path)
    explicit = _model(tmp_path, mmdit_config=dict(k.KREA2_MMDIT_CONFIG))
    assert implicit.feedback_reference_metadata == explicit.feedback_reference_metadata


def test_text_encoder_loader_records_actual_weight_config_and_tokenizer_sources(
    tmp_path, monkeypatch
):
    te_dir = tmp_path / "te"
    te_dir.mkdir()
    weight_file = te_dir / "model.safetensors"
    weight_file.write_bytes(b"te weights")
    model = _model(
        tmp_path,
        text_encoder_path=str(te_dir),
        text_encoder_weight_path=str(weight_file),
    )

    class FakeTokenizer:
        def __init__(self, source):
            self.name_or_path = source
            self.init_kwargs = {"_commit_hash": "te-commit"}

    class FakeTextEncoder:
        def __init__(self, source):
            self._aitk_weight_source = source
            self.config = SimpleNamespace(_commit_hash="te-commit")

        def drop_vision_tower(self):
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            assert value is False
            return self

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return FakeTokenizer(source)

    class FakeQwen2TokenizerFast:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return FakeTokenizer(source)

    class FakeQwen3VLTextEncoder:
        @classmethod
        def load_model(cls, source, **kwargs):
            return FakeTextEncoder(source)

    monkeypatch.setattr(k, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(k, "Qwen2TokenizerFast", FakeQwen2TokenizerFast)
    monkeypatch.setattr(k, "Qwen3VLTextEncoder", FakeQwen3VLTextEncoder)
    monkeypatch.setattr(k, "flush", lambda: None)

    model._load_text_encoder()

    assert model.feedback_component_sources["text_encoder"] == os.path.realpath(
        weight_file
    )
    assert model.feedback_component_sources["text_encoder_config"] == os.path.realpath(
        te_dir
    )
    assert model.feedback_component_sources["tokenizer"] == os.path.realpath(te_dir)


def test_hub_text_encoder_sources_are_pinned_without_network(tmp_path, monkeypatch):
    repo = "org/Qwen3-VL"
    snapshot = "C:/cache/hub/models--org--Qwen3-VL/snapshots/0123456789abcdef/vocab.json"
    model = _model(tmp_path, text_encoder_path=repo)

    class FakeTokenizer:
        def __init__(self):
            self.name_or_path = repo
            self.init_kwargs = {"vocab_file": snapshot}

    class FakeTextEncoder:
        def __init__(self):
            self._aitk_weight_source = repo
            self.config = SimpleNamespace(_commit_hash="abcdef0123456789")

        def drop_vision_tower(self):
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            assert value is False
            return self

    class FakeTokenizerLoader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            assert source == repo
            return FakeTokenizer()

    class FakeTextEncoderLoader:
        @classmethod
        def load_model(cls, source, **kwargs):
            assert source == repo
            return FakeTextEncoder()

    monkeypatch.setattr(k, "AutoTokenizer", FakeTokenizerLoader)
    monkeypatch.setattr(k, "Qwen2TokenizerFast", FakeTokenizerLoader)
    monkeypatch.setattr(k, "Qwen3VLTextEncoder", FakeTextEncoderLoader)
    monkeypatch.setattr(k, "flush", lambda: None)

    model._load_text_encoder()

    assert model.feedback_component_sources["text_encoder"] == (
        f"{repo}@abcdef0123456789"
    )
    assert model.feedback_component_sources["text_encoder_config"] == (
        f"{repo}@abcdef0123456789"
    )
    assert model.feedback_component_sources["tokenizer"] == (
        f"{repo}@0123456789abcdef"
    )


def test_vae_loader_records_actual_weight_and_config_sources(tmp_path, monkeypatch):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    weight_file = vae_dir / "vae.safetensors"
    weight_file.write_bytes(b"vae weights")
    config_dir = tmp_path / "vae-config"
    config_dir.mkdir()
    model = _model(
        tmp_path,
        vae_path=str(weight_file),
        vae_config_path=str(config_dir),
    )

    class FakeVAE:
        def __init__(self):
            self._aitk_weight_source = str(weight_file)
            self.config = SimpleNamespace(_commit_hash="vae-commit")

        def eval(self):
            return self

        def requires_grad_(self, value):
            assert value is False
            return self

    class FakeQwenImageVAE:
        @classmethod
        def load_model(cls, path, **kwargs):
            assert path == str(weight_file)
            assert kwargs["config_path"] == str(config_dir)
            return FakeVAE()

    monkeypatch.setattr(k, "QwenImageVAE", FakeQwenImageVAE)

    model._load_vae()

    assert model.feedback_component_sources["vae"] == os.path.realpath(weight_file)
    assert model.feedback_component_sources["vae_config"] == os.path.realpath(
        config_dir
    )


def test_transformer_loader_records_the_resolved_single_file(tmp_path, monkeypatch):
    model = _model(tmp_path, checkpoint_filename="nested/actual.safetensors")
    actual = tmp_path / "resolved.safetensors"
    actual.write_bytes(b"transformer")
    sentinel = object()
    calls = []

    def resolve(name_or_path, filename):
        calls.append((name_or_path, filename))
        return os.path.realpath(actual)

    monkeypatch.setattr(k, "_resolve_mmdit_file", resolve)
    monkeypatch.setattr(k, "load_file", lambda path: {})
    monkeypatch.setattr(
        k.SingleStreamDiT,
        "load_from_state_dict",
        staticmethod(lambda state_dict, dtype, config=None: sentinel),
    )
    monkeypatch.setattr(k, "flush", lambda: None)

    assert model._load_transformer() is sentinel
    assert calls == [
        (str(tmp_path / "weights"), "nested/actual.safetensors")
    ]
    assert model.feedback_component_sources["transformer"] == os.path.realpath(actual)


def test_construction_does_no_network_lookup(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("model construction must not resolve extra Hub files")

    monkeypatch.setattr(k.huggingface_hub, "hf_hub_download", fail)
    model = _model(tmp_path, sample_shift=2.0)
    assert model.feedback_component_sources == {}
    assert model.feedback_reference_metadata["mmdit_config"]["layers"] == k.KREA2_MMDIT_CONFIG["layers"]
