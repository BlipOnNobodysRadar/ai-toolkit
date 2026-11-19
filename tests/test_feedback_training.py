"""CPU contracts for GUI preference-pair consumption."""

from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import (
    DatasetConfig,
    ModelConfig,
    NetworkConfig,
    TrainConfig,
)
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit import feedback_training as feedback


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _make_base_tree(tmp_path, name="base"):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.json"
    if not config.exists():
        config.write_text('{"arch": "moss_image"}', encoding="utf-8")
    vae = root / "vae"
    vae.mkdir(exist_ok=True)
    if not (vae / "vae.safetensors").exists():
        (vae / "vae.safetensors").write_bytes(b"vae-fixture")
    text_encoder = root / "text_encoder"
    text_encoder.mkdir(exist_ok=True)
    if not (text_encoder / "te.safetensors").exists():
        (text_encoder / "te.safetensors").write_bytes(b"te-fixture")
    return root


def _make_component_sources(tmp_path, suffix="one", content=b"transformer-fixture"):
    components = tmp_path / "components" / suffix
    components.mkdir(parents=True, exist_ok=True)
    transformer = components / "transformer.safetensors"
    if not transformer.exists():
        transformer.write_bytes(content)
    tokenizer = components / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    tokenizer_file = tokenizer / "tokenizer.json"
    if not tokenizer_file.exists():
        tokenizer_file.write_text('{"fixture": true}', encoding="utf-8")
    return {
        "transformer": str(transformer),
        "tokenizer": str(tokenizer),
    }


def _fake_trainer(
    tmp_path,
    *,
    base_name="base",
    component_suffix="one",
    component_content=b"transformer-fixture",
    beta=100.0,
    every=10,
    lr=1e-6,
    steps=100,
    dataset_weight=1.25,
    dataset_loss=0.75,
    network_rank=4,
):
    base = _make_base_tree(tmp_path, base_name)
    model_config = ModelConfig(
        name_or_path=str(base),
        arch="moss_image",
        dtype="float16",
        quantize=False,
        qtype="qfloat8",
        quantize_te=False,
        qtype_te="qfloat8",
        te_name_or_path=None,
        model_kwargs={"edit": False},
    )
    train_config = TrainConfig(
        loss_type="mse",
        dpo_feedback={"every": every, "start_after": 0},
        dpo_beta=beta,
        lr=lr,
        steps=steps,
        gradient_accumulation_steps=2,
    )
    network_config = NetworkConfig(type="lora", linear=network_rank)
    dataset_config = DatasetConfig(
        folder_path=str(tmp_path / "source"),
        cache_latents=True,
        cache_text_embeddings=True,
        resolution=512,
        buckets=True,
        network_weight=dataset_weight,
        loss_multiplier=dataset_loss,
        caption_dropout_rate=0.4,
        token_dropout_rate=0.3,
        random_crop=True,
    )
    component_sources = _make_component_sources(
        tmp_path, suffix=component_suffix, content=component_content
    )
    sd = SimpleNamespace(
        model_config=model_config,
        vae=torch.nn.Linear(2, 2),
        text_encoder=torch.nn.Linear(2, 2),
        vae_torch_dtype="float16",
        te_torch_dtype="float16",
        te_padding_side="right",
        cache_latents_as_uint8=False,
        load_rgba=False,
        is_audio_model=False,
        sample_rate=None,
        feedback_component_sources=component_sources,
    )
    trainer = SimpleNamespace(
        save_root=str(tmp_path / "job"),
        model_config=model_config,
        train_config=train_config,
        network_config=network_config,
        dataset_configs=[dataset_config],
        sd=sd,
        network=torch.nn.Linear(2, 2),
        adapter_config=None,
        decorator_config=None,
        adapter=None,
        decorator=None,
    )
    (tmp_path / "job").mkdir(parents=True, exist_ok=True)
    return trainer


def _write_pair(
    tmp_path,
    trainer,
    *,
    pair_id=None,
    prompt="a fixed feedback prompt",
    choice="a",
    write_vote=True,
    pair_version=1,
    vote_id=None,
    reference_id=None,
    latent_a=None,
    latent_b=None,
    created_at=None,
    png_a=None,
    png_b=None,
    embeddings_writer=None,
):
    pair_id = pair_id or uuid.uuid4().hex
    pair_dir = tmp_path / "job" / "preferences" / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    if png_a is None:
        Image.new("RGB", (32, 24), (210, 20, 10)).save(pair_dir / "a.png")
    else:
        (pair_dir / "a.png").write_bytes(png_a)
    if png_b is None:
        Image.new("RGB", (32, 24), (10, 20, 210)).save(pair_dir / "b.png")
    else:
        (pair_dir / "b.png").write_bytes(png_b)
    prompt_embeds = AdvancedPromptEmbeds(
        text_embeds=[torch.arange(12, dtype=torch.float32).reshape(3, 4)]
    )
    if embeddings_writer is None:
        prompt_embeds.save(str(pair_dir / "embeddings.safetensors"))
    else:
        embeddings_writer(pair_dir / "embeddings.safetensors", prompt_embeds)
    latents_a = torch.ones(1, 4, 2, 2) if latent_a is None else latent_a
    latents_b = torch.full((1, 4, 2, 2), 2.0) if latent_b is None else latent_b
    save_file({"a": latents_a, "b": latents_b}, str(pair_dir / "latents.safetensors"))
    record = {
        "version": pair_version,
        "id": pair_id,
        "step": 20,
        "prompt": prompt,
        "seeds": [101, 202],
        "width": 32,
        "height": 24,
        "reference_id": reference_id or feedback.feedback_reference_id(trainer),
        "created_at": created_at or _utc_now(),
    }
    (pair_dir / "pair.json").write_text(json.dumps(record), encoding="utf-8")
    if write_vote:
        vote = {
            "version": 1,
            "id": pair_id if vote_id is None else vote_id,
            "choice": choice,
            "voted_at": _utc_now(),
        }
        (pair_dir / "vote.json").write_text(json.dumps(vote), encoding="utf-8")
    return pair_id, pair_dir, prompt_embeds, latents_a, latents_b


def _replace_vote(pair_dir, pair_id, choice):
    temporary = pair_dir / "vote.tmp"
    temporary.write_text(
        json.dumps(
            {
                "version": 1,
                "id": pair_id,
                "choice": choice,
                "voted_at": _utc_now(),
            }
        ),
        encoding="utf-8",
    )
    os.replace(temporary, pair_dir / "vote.json")


def test_reference_identity_uses_only_frozen_model_and_encoding_identity(tmp_path):
    first = _fake_trainer(tmp_path)
    ref = feedback.feedback_reference_id(first)
    assert feedback.REFERENCE_ID_RE.fullmatch(ref)

    # Dataset, train knobs, optimizer-ish live weights, and LoRA settings are
    # not reference identity. A separate trainer with different values still
    # addresses the same votes.
    second = _fake_trainer(
        tmp_path,
        beta=7.5,
        every=3,
        lr=3e-5,
        steps=999,
        dataset_weight=9.0,
        dataset_loss=8.0,
        network_rank=16,
    )
    second.model_config.model_kwargs.update(batched_cfg=False, sdpa_cudnn=False, fused_pointwise=True,
                                           uncond_skip_layers=[], sample_shift=7.0)
    assert feedback.feedback_reference_id(second) == ref
    with torch.no_grad():
        first.network.weight.add_(123.0)
        second.network.weight.add_(456.0)
    assert feedback.feedback_reference_id(first) == ref

    # Frozen model construction changes do change the identity.
    changed_quantization = _fake_trainer(tmp_path)
    changed_quantization.model_config.quantize = True
    changed_quantization.model_config.qtype = "convrot8"
    assert feedback.feedback_reference_id(changed_quantization) != ref

    changed_components = _fake_trainer(
        tmp_path,
        component_suffix="two",
        component_content=b"different-transformer",
    )
    assert feedback.feedback_reference_id(changed_components) != ref


def test_feedback_reference_follows_the_loaded_recovery_adapter(tmp_path):
    def reference(source):
        trainer = _fake_trainer(tmp_path)
        trainer.model_config.accuracy_recovery_adapter = "org/repo/adapter.safetensors"
        trainer.sd.accuracy_recovery_adapter = SimpleNamespace(_aitk_weight_source=str(source))
        return feedback.feedback_reference_id(trainer)

    first = tmp_path / "adapter.safetensors"
    first.write_bytes(b"first revision")
    before = reference(first)
    assert reference(first) == before
    first.write_bytes(b"second revision")
    assert reference(first) != before


def test_reference_identity_ignores_transient_loaded_encoder_state(tmp_path):
    trainer = _fake_trainer(tmp_path)
    loaded = feedback.feedback_reference_id(trainer)

    # A resumed process may not have the VAE/TE resident yet. The stable
    # path/config identity must not depend on loaded object classes.
    del trainer._feedback_reference_id
    del trainer._feedback_reference_identity
    trainer.sd.vae = None
    trainer.sd.text_encoder = None
    assert feedback.feedback_reference_id(trainer) == loaded


def test_reference_identity_is_cached_once_on_trainer(tmp_path, monkeypatch):
    trainer = _fake_trainer(tmp_path)
    calls = {"count": 0}
    original = feedback._feedback_reference_identity

    def counted(value):
        calls["count"] += 1
        return original(value)

    monkeypatch.setattr(feedback, "_feedback_reference_identity", counted)
    first = feedback.feedback_reference_id(trainer)
    trainer.train_config.steps = 12345
    trainer.train_config.dpo_beta = 1.0
    trainer.dataset_configs[0].network_weight = 42.0
    second = feedback.feedback_reference_id(trainer)

    assert first == second
    assert calls["count"] == 1


def test_voted_ids_sorted_and_skips_pending_skipped_foreign_old_and_stale(tmp_path):
    trainer = _fake_trainer(tmp_path)
    trainer.additional_logs = {}
    first_id, _, _, _, _ = _write_pair(tmp_path, trainer, choice="a")
    second_id, _, _, _, _ = _write_pair(tmp_path, trainer, choice="b")
    pending_id, _, _, _, _ = _write_pair(tmp_path, trainer, write_vote=False)
    skipped_id, _, _, _, _ = _write_pair(tmp_path, trainer, choice="skip")
    foreign_id, _, _, _, _ = _write_pair(
        tmp_path, trainer, reference_id="f" * 64
    )
    old_id, _, _, _, _ = _write_pair(tmp_path, trainer, pair_version=0)
    stale_id, _, _, _, _ = _write_pair(
        tmp_path, trainer, vote_id=uuid.uuid4().hex
    )

    eligible = feedback.voted_feedback_ids(trainer)
    assert eligible == sorted([first_id, second_id])
    assert pending_id not in eligible
    assert skipped_id not in eligible
    assert foreign_id not in eligible
    assert old_id not in eligible
    assert stale_id not in eligible

    assert trainer.additional_logs["feedback/eligible_pairs"] == 2
    assert trainer.additional_logs["feedback/foreign_votes"] == 1
    assert trainer.additional_logs["feedback/skipped_pairs"] == 1


def test_poll_cache_observes_delayed_vote_but_rejects_changed_immutable_vote(tmp_path):
    trainer = _fake_trainer(tmp_path)
    pair_id, pair_dir, _, _, _ = _write_pair(
        tmp_path, trainer, write_vote=False
    )
    assert feedback.voted_feedback_ids(trainer) == []

    _replace_vote(pair_dir, pair_id, "a")
    assert feedback.voted_feedback_ids(trainer) == [pair_id]

    _replace_vote(pair_dir, pair_id, "b")
    assert feedback.voted_feedback_ids(trainer) == []
    with pytest.raises(feedback.FeedbackCorruptError, match="changed"):
        feedback.load_feedback_batch(trainer, pair_id)


def test_polling_does_not_read_tensor_payloads(tmp_path, monkeypatch):
    trainer = _fake_trainer(tmp_path)
    pair_id, _, _, _, _ = _write_pair(tmp_path, trainer)

    def fail(*args, **kwargs):
        raise AssertionError("polling must not load tensor payloads")

    monkeypatch.setattr(feedback, "_load_latents", fail)
    monkeypatch.setattr(feedback, "_load_embeddings", fail)
    assert feedback.voted_feedback_ids(trainer) == [pair_id]


def test_load_batch_uses_real_dtos_neutral_dataset_and_native_collation(tmp_path):
    trainer = _fake_trainer(tmp_path)
    pair_id, pair_dir, prompt_embeds, latents_a, latents_b = _write_pair(
        tmp_path, trainer, choice="b", prompt="normal pipeline prompt"
    )

    batch = feedback.load_feedback_batch(trainer, pair_id)

    assert type(batch) is DataLoaderBatchDTO
    assert len(batch.file_items) == 1
    chosen = batch.file_items[0]
    assert type(chosen) is FileItemDTO
    assert type(chosen.rejected_item) is FileItemDTO

    # The file contract is one-image [1,C,H,W]; the per-item DTO axis is
    # [C,H,W], and the native batch constructor adds exactly one batch axis.
    assert tuple(chosen.get_latent().shape) == (4, 2, 2)
    assert tuple(batch.latents.shape) == (1, 4, 2, 2)
    assert tuple(batch.rejected_latents.shape) == (1, 4, 2, 2)
    assert tuple(chosen.rejected_item.get_latent().shape) == (4, 2, 2)
    torch.testing.assert_close(batch.latents, latents_b)
    torch.testing.assert_close(batch.rejected_latents, latents_a)
    assert batch.tensor is None
    assert batch.rejected_tensor is None

    assert chosen.path == str(pair_dir / "b.png")
    assert chosen.rejected_path == str(pair_dir / "a.png")
    assert chosen.rejected_item.path == str(pair_dir / "a.png")
    assert chosen.caption == "normal pipeline prompt"
    assert chosen.raw_caption == "normal pipeline prompt"
    assert chosen.crop_width == 32
    assert chosen.crop_height == 24
    assert chosen.dpo_pair_signature == pair_id

    # Synthetic pairs have a private neutral config, not the training dataset.
    assert type(chosen.dataset_config) is DatasetConfig
    assert chosen.dataset_config is batch.dataset_config
    assert chosen.dataset_config is not trainer.dataset_configs[0]
    assert chosen.network_weight == 1.0
    assert chosen.loss_multiplier == 1.0
    assert chosen.dataset_config.network_weight == 1.0
    assert chosen.dataset_config.loss_multiplier == 1.0
    assert chosen.dataset_config.caption_dropout_rate == 0.0
    assert chosen.dataset_config.token_dropout_rate == 0.0
    assert chosen.dataset_config.random_crop is False
    assert chosen.dataset_config.augments == []

    assert batch.get_caption_list() == ["normal pipeline prompt"]
    assert batch.get_caption_short_list() == ["normal pipeline prompt"]
    assert batch.get_network_weight_list() == [1.0]
    assert batch.get_is_reg_list() == [False]
    assert batch.loss_multiplier_list == [1.0]
    assert batch.feedback_pair_id == pair_id
    assert batch.feedback_choice == "b"
    assert batch.feedback_seeds == (101, 202)
    assert batch.feedback_step == 20

    # Prompt embeddings are collated by DataLoaderBatchDTO from the native
    # cached AdvancedPromptEmbeds payload.
    assert isinstance(batch.prompt_embeds, AdvancedPromptEmbeds)
    assert batch.prompt_embeds is not prompt_embeds
    assert batch.prompt_embeds.keys() == ["text_embeds"]
    assert len(batch.prompt_embeds.text_embeds) == 1
    torch.testing.assert_close(
        batch.prompt_embeds.text_embeds[0], prompt_embeds.text_embeds[0]
    )


def test_file_item_constructor_failure_fails_closed_without_fake_fallback(
    tmp_path, monkeypatch
):
    trainer = _fake_trainer(tmp_path)
    pair_id, _, _, _, _ = _write_pair(tmp_path, trainer)

    class BrokenFileItemDTO:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("native constructor rejected the source")

    monkeypatch.setattr(feedback, "FileItemDTO", BrokenFileItemDTO)
    with pytest.raises(feedback.FeedbackCorruptError, match="FileItemDTO"):
        feedback.load_feedback_batch(trainer, pair_id)


def test_load_batch_rejects_malformed_png(tmp_path):
    trainer = _fake_trainer(tmp_path)
    pair_id, _, _, _, _ = _write_pair(
        tmp_path, trainer, png_a=b"not a png at all"
    )
    with pytest.raises(feedback.FeedbackCorruptError, match="PNG"):
        feedback.load_feedback_batch(trainer, pair_id)


def test_load_batch_rejects_wrong_png_geometry(tmp_path):
    trainer = _fake_trainer(tmp_path)
    pair_id, pair_dir, _, _, _ = _write_pair(tmp_path, trainer)
    Image.new("RGB", (16, 16), (1, 2, 3)).save(pair_dir / "a.png")
    with pytest.raises(feedback.FeedbackCorruptError, match="pair.json declares"):
        feedback.load_feedback_batch(trainer, pair_id)


def test_load_batch_rejects_malformed_latent_shapes_and_keys(tmp_path):
    trainer = _fake_trainer(tmp_path)
    bad_shape_id, _, _, _, _ = _write_pair(
        tmp_path, trainer, latent_b=torch.ones(2, 4, 2, 2)
    )
    with pytest.raises(feedback.FeedbackCorruptError, match=r"shape \[1,C,H,W\]"):
        feedback.load_feedback_batch(trainer, bad_shape_id)

    nonfinite_id, _, _, _, _ = _write_pair(
        tmp_path, trainer, latent_a=torch.full((1, 4, 2, 2), float("nan"))
    )
    with pytest.raises(feedback.FeedbackCorruptError, match="non-finite"):
        feedback.load_feedback_batch(trainer, nonfinite_id)

    wrong_keys_id, pair_dir, _, _, _ = _write_pair(tmp_path, trainer)
    save_file({"a": torch.ones(1, 4, 2, 2)}, str(pair_dir / "latents.safetensors"))
    with pytest.raises(feedback.FeedbackCorruptError, match="exactly keys"):
        feedback.load_feedback_batch(trainer, wrong_keys_id)


def test_load_batch_rejects_malformed_embeddings(tmp_path):
    def write_raw_embeddings(path, _prompt_embeds):
        save_file({"text_embeds": torch.ones(3, 4)}, str(path))

    trainer = _fake_trainer(tmp_path)
    pair_id, _, _, _, _ = _write_pair(
        tmp_path, trainer, embeddings_writer=write_raw_embeddings
    )
    with pytest.raises(
        feedback.FeedbackCorruptError, match="AdvancedPromptEmbeds"
    ):
        feedback.load_feedback_batch(trainer, pair_id)


def test_load_batch_rejects_path_traversal_and_symlink_material(tmp_path):
    trainer = _fake_trainer(tmp_path)
    pair_id, pair_dir, _, _, _ = _write_pair(tmp_path, trainer)

    with pytest.raises(feedback.FeedbackPathError):
        feedback.load_feedback_batch(trainer, f"../{pair_id}")
    with pytest.raises(feedback.FeedbackPathError):
        feedback.load_feedback_batch(trainer, "A" * 32)

    target = pair_dir / "real_latents.safetensors"
    (pair_dir / "latents.safetensors").replace(target)
    try:
        os.symlink(target, pair_dir / "latents.safetensors")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")
    with pytest.raises(feedback.FeedbackPathError, match="symlink"):
        feedback.load_feedback_batch(trainer, pair_id)


def test_load_batch_preserves_domain_errors_for_ineligible_pairs(tmp_path):
    trainer = _fake_trainer(tmp_path)
    pending_id, _, _, _, _ = _write_pair(tmp_path, trainer, write_vote=False)
    skipped_id, _, _, _, _ = _write_pair(tmp_path, trainer, choice="skip")
    foreign_id, _, _, _, _ = _write_pair(
        tmp_path, trainer, reference_id="e" * 64
    )
    old_id, _, _, _, _ = _write_pair(tmp_path, trainer, pair_version=0)

    with pytest.raises(feedback.FeedbackPendingError):
        feedback.load_feedback_batch(trainer, pending_id)
    with pytest.raises(feedback.FeedbackSkippedError):
        feedback.load_feedback_batch(trainer, skipped_id)
    with pytest.raises(feedback.FeedbackForeignReferenceError):
        feedback.load_feedback_batch(trainer, foreign_id)
    with pytest.raises(feedback.FeedbackCorruptError):
        feedback.load_feedback_batch(trainer, old_id)
    with pytest.raises(feedback.FeedbackCorruptError, match="does not exist"):
        feedback.load_feedback_batch(trainer, uuid.uuid4().hex)


def test_committed_vote_survives_interrupted_receipt_publish(tmp_path):
    trainer = _fake_trainer(tmp_path)
    pair_id, pair_dir, *_ = _write_pair(tmp_path, trainer, choice="b")
    committed = pair_dir / ".decision"
    committed.mkdir()
    (pair_dir / "vote.json").rename(committed / "vote.json")
    assert feedback.voted_feedback_ids(trainer) == [pair_id]
    assert feedback.load_feedback_batch(trainer, pair_id).feedback_choice == "b"
