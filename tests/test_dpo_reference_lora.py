"""The original transformer LoRA stays active when DPO switches the policy off."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import test_krea_dpo as krea
import test_moss_dpo_training as moss
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.config_modules import TrainConfig
from toolkit.inference_lora import InferenceLoRA


@pytest.mark.parametrize("architecture", ["moss", "krea"])
def test_frozen_lora_stacks_with_a_real_policy(tmp_path, architecture):
    fixture = moss if architecture == "moss" else krea
    model, transformer = fixture._make_model()
    name, linear = next((k, m) for k, m in transformer.named_modules()
                        if isinstance(m, torch.nn.Linear) and k.endswith(
                            "proj.qkv" if architecture == "moss" else "wq"))
    x = torch.randn(3, linear.in_features)
    base = linear(x).detach()
    a = torch.randn(2, linear.in_features) * 0.1
    b = torch.randn(linear.out_features, 2) * 0.1
    weights = model.convert_lora_weights_before_save({
        f"transformer.{name}.lora_A.weight": a,
        f"transformer.{name}.lora_B.weight": b,
    })
    path = str(tmp_path / "original.safetensors")
    save_file(weights, path)
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.sd = model
    trainer.train_config = TrainConfig(loss_type="dpo", dpo_reference_lora_path=path,
                                       dpo_reference_lora_strength=0.75)
    trainer.network_config = SimpleNamespace(pretrained_lora_path=None)
    trainer.hook_after_model_load()
    with torch.inference_mode():
        expected = base + 0.75 * (x @ a.T) @ b.T
        torch.testing.assert_close(linear(x), expected)
    # Sampling may populate the hook cache before training builds an autograd graph.
    linear(x.detach().requires_grad_()).sum().backward()
    transformer.zero_grad()
    network, config = fixture._make_network(model, transformer)
    if architecture == "moss":
        job = moss._make_trainer(model, transformer, network, config)
        batch = moss._make_latent_batch()
    else:
        job = krea._make_trainer(model, network, config)
        batch = krea._make_latent_batch(model)
    batch.file_items[0].preference_probability = 0.7
    job.network.is_active = False
    torch.testing.assert_close(linear(x), expected)
    params = list(network.parameters())
    optimizer = torch.optim.SGD(params, lr=0.01)
    before = [p.detach().clone() for p in params]
    loss = job.train_single_accumulation(batch, accum_scale=1.0)
    assert loss.item() == pytest.approx(0.693147, abs=2e-5)
    optimizer.step()
    assert any(not torch.equal(p, v) for p, v in zip(params, before))
    job.network.is_active = False
    torch.testing.assert_close(linear(x), expected)
    assert all(p.grad is None for p in transformer.parameters())


@pytest.mark.parametrize("parts", [
    {"lora_B.weight": torch.ones(2, 1)},
    {"lora_A.weight": torch.ones(1, 3), "lora_B.weight": torch.ones(2, 4)},
    {"lora_A.weight": torch.ones(1, 3), "lora_B.weight": torch.ones(2, 1), "dora_scale": torch.ones(2)},
])
def test_partial_frozen_reference_fails_before_attachment(tmp_path, parts):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    path = str(tmp_path / "bad.safetensors")
    save_file({"transformer.0." + key: value for key, value in parts.items()}, path)
    with pytest.raises(ValueError, match="LoRA"):
        InferenceLoRA(path).load(SimpleNamespace(model=model), strict_transformer_lora=True)
    assert not model[0]._forward_hooks


def test_resume_requires_the_same_frozen_adapter_and_strength(tmp_path):
    import json
    identity = {"sha256": "a" * 64, "strength": 0.75}
    job = SimpleNamespace(step_num=5, epoch_num=1, _dpo_reference_identity=identity,
                          accelerator=SimpleNamespace(is_main_process=True),
                          network_config=SimpleNamespace(pretrained_lora_path=None),
                          train_config=SimpleNamespace(start_step=None))
    info = BaseSDTrainProcess.get_training_info(job)
    path = str(tmp_path / "policy.safetensors")
    save_file({}, path, metadata={"training_info": json.dumps(info)})
    BaseSDTrainProcess.load_training_state_from_metadata(job, path)
    for changed in (None, {**identity, "strength": 0.5}, {**identity, "sha256": "b" * 64}):
        job._dpo_reference_identity = changed
        with pytest.raises(ValueError, match="differs from this checkpoint"):
            BaseSDTrainProcess.load_training_state_from_metadata(job, path)
