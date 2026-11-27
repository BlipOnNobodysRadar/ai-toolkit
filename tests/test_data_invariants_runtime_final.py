"""Focused validation, inference-LoRA, and optimizer regressions (CPU, tiny files)."""
import os
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from training_extras import toolkit_validation as tv


# ---------------------------------------------------------------- VALIDATION-01 root/file disjointness

def _trainer(train_root, validation_root, training_paths):
    dataset = SimpleNamespace(
        file_list=[SimpleNamespace(path=str(p)) for p in training_paths],
        dataset_config=SimpleNamespace(folder_path=str(train_root)),
    )
    cfg = SimpleNamespace(datasets=[{"folder_path": str(validation_root)}])
    return SimpleNamespace(
        train_config=SimpleNamespace(validation_config=cfg),
        datasets=[dataset],
        data_loader=None,
    )


def test_file_identity_is_dev_inode_only_and_matches_hardlinks(tmp_path):
    source = tmp_path / "train.png"
    source.write_bytes(b"pixels")
    hardlink = tmp_path / "hard.png"
    os.link(source, hardlink)
    assert tv._canonical_path(source) != tv._canonical_path(hardlink)
    identity = tv._file_identity(source)
    assert identity == tv._file_identity(hardlink)
    assert len(identity) == 2 and identity[1] != 0


def test_setup_refuses_equal_and_warns_on_disjoint_nested_roots(tmp_path, monkeypatch):
    root = tmp_path / "data"
    nested = root / "validation"
    nested.mkdir(parents=True)
    train_file = root / "train.png"
    train_file.write_bytes(b"train")
    val_file = nested / "val.png"
    val_file.write_bytes(b"val")

    equal = _trainer(root, root, [train_file])
    with pytest.raises(ValueError, match="equal"):
        tv.setup(equal)

    # Disjoint files inside a nested validation folder warn by default instead of hard-blocking
    nested_trainer = _trainer(root, nested, [train_file])
    with pytest.warns(UserWarning, match="nested"):
        tv._check_validation_disjoint(nested_trainer)

    # Actual overlap inside a nested folder is still hard-refused
    overlapping_nested = _trainer(root, nested, [train_file, val_file])
    with pytest.raises(ValueError, match="also an admitted training file"):
        tv._check_validation_disjoint(overlapping_nested)

    # Strict mode still refuses any nested root
    monkeypatch.setenv("AITK_STRICT_INVARIANTS", "1")
    with pytest.raises(ValueError, match="nested"):
        tv.setup(nested_trainer)


def test_sibling_with_shared_string_prefix_is_admitted(tmp_path):
    train_root = tmp_path / "data"
    val_root = tmp_path / "data2"
    train_root.mkdir()
    val_root.mkdir()
    train_file = train_root / "train.png"
    val_file = val_root / "val.png"
    train_file.write_bytes(b"train")
    val_file.write_bytes(b"val")
    tv._check_validation_disjoint(_trainer(train_root, val_root, [train_file]))


def test_symlink_alias_and_hardlink_under_disjoint_roots_are_detected(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "validation"
    train_root.mkdir()
    val_root.mkdir()
    train_file = train_root / "train.png"
    train_file.write_bytes(b"same-bytes")

    hardlink = val_root / "hard.png"
    os.link(train_file, hardlink)
    with pytest.raises(ValueError, match="also an admitted training file"):
        tv._check_validation_disjoint(_trainer(train_root, val_root, [train_file]))
    hardlink.unlink()

    symlink = val_root / "alias.png"
    try:
        os.symlink(train_file, symlink)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlink creation unavailable in this environment: {e}")
    with pytest.raises(ValueError, match="also an admitted training file"):
        tv._check_validation_disjoint(_trainer(train_root, val_root, [train_file]))


@pytest.mark.skipif(os.name != "nt", reason="NTFS directory junction fixture is Windows-only")
def test_directory_junction_alias_is_detected_as_canonical_overlap(tmp_path):
    train_root = tmp_path / "train"
    train_root.mkdir()
    train_file = train_root / "train.png"
    train_file.write_bytes(b"train")
    val_alias = tmp_path / "validation_alias"
    env = dict(os.environ, JUNCTION_TARGET=str(train_root), JUNCTION_LINK=str(val_alias))
    import subprocess
    result = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", "%JUNCTION_LINK%", "%JUNCTION_TARGET%"],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        pytest.skip(f"directory junction creation unavailable: {result.stdout} {result.stderr}")
    with pytest.raises(ValueError, match="equal"):
        tv._check_validation_disjoint(_trainer(train_root, val_alias, [train_file]))


def test_nested_controls_warns_when_disjoint_and_refuses_when_strict(tmp_path, monkeypatch):
    train_root = tmp_path / "data"
    val_root = train_root / "_controls" / "validation"
    val_root.mkdir(parents=True)
    train_file = train_root / "train.png"
    train_file.write_bytes(b"train")
    (val_root / "val.png").write_bytes(b"val")
    with pytest.warns(UserWarning, match="nested"):
        tv._check_validation_disjoint(_trainer(train_root, val_root, [train_file]))
    monkeypatch.setenv("AITK_STRICT_INVARIANTS", "1")
    with pytest.raises(ValueError, match="nested"):
        tv._check_validation_disjoint(_trainer(train_root, val_root, [train_file]))


def test_actual_admitted_file_list_on_loader_path_is_checked(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "validation"
    train_root.mkdir()
    val_root.mkdir()
    train_file = train_root / "train.png"
    train_file.write_bytes(b"same-bytes")
    os.link(train_file, val_root / "hard.png")
    dataset = SimpleNamespace(file_list=[SimpleNamespace(path=str(train_file))])
    trainer = SimpleNamespace(
        train_config=SimpleNamespace(validation_config=SimpleNamespace(
            datasets=[{"folder_path": str(val_root)}])),
        datasets=[],
        data_loader=SimpleNamespace(datasets=[dataset]),
    )
    with pytest.raises(ValueError, match="also an admitted training file"):
        tv._check_validation_disjoint(trainer)


# ---------------------------------------------------------------- INFERENCE-01 loaded file generation

def _tiny_lora(path, scale=1.0):
    save_file({
        "transformer.0.lora_A.weight": torch.eye(2),
        "transformer.0.lora_B.weight": torch.eye(2) * scale,
    }, str(path))
    return str(path)


def _holder():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.zero_()
    return SimpleNamespace(model=model, text_encoder=None)


def _replace_lora(path, scale):
    tmp = path.with_suffix(".new.safetensors")
    _tiny_lora(tmp, scale)
    time.sleep(0.01)
    os.replace(tmp, path)


def test_hook_stack_keeps_loaded_generation_and_engine_reloads_replaced_file(tmp_path):
    from extensions_built_in.inference_engine.engine import Engine
    from toolkit.inference_lora import InferenceLoRA, LoRAStack

    path = tmp_path / "tiny.safetensors"
    _tiny_lora(path, scale=1.0)
    holder = _holder()
    stack = LoRAStack(holder, "hook").load([{"path": str(path), "strength": 1.0}])
    old_lora = stack.loras[0]
    loaded_key = stack.key()
    stack.apply()
    x = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(holder.model[0](x), x)

    _replace_lora(path, scale=2.0)
    assert stack.key() == loaded_key  # cleanup/eviction still refers to the weights actually loaded
    desired = LoRAStack(holder, "hook")
    desired.loras = [InferenceLoRA(str(path), 1.0)]
    assert desired.key() != loaded_key
    assert stack.set_strengths([{"path": str(path), "strength": 0.5}]) is False

    engine = Engine("cpu", str(tmp_path / "outputs"))
    engine.holder = holder
    engine.status_cb = lambda message: None
    engine.lora_stack = stack
    job = SimpleNamespace(emit=lambda *args, **kwargs: None)
    engine._apply_loras(job, [{"path": str(path), "strength": 1.0}], "hook")
    assert engine.lora_stack is not stack
    assert old_lora._hooks == [] and stack.loras == []  # old hook generation was detached
    assert torch.allclose(holder.model[0](x), x * 2.0)
    engine.lora_stack.remove()
    assert torch.allclose(holder.model[0](x), torch.zeros_like(x))


def test_merge_mode_loaded_key_stays_stable_for_cleanup_and_remove_does_not_unmerge(tmp_path):
    from toolkit.inference_lora import InferenceLoRA, LoRAStack

    path = tmp_path / "tiny.safetensors"
    _tiny_lora(path, scale=1.0)
    holder = _holder()
    stack = LoRAStack(holder, "merge").load([{"path": str(path), "strength": 1.0}])
    loaded_key = stack.key()
    stack.apply()
    merged_weight = holder.model[0].weight.detach().clone()
    assert torch.allclose(merged_weight, torch.eye(2))

    _replace_lora(path, scale=2.0)
    assert stack.key() == loaded_key
    desired = LoRAStack(holder, "merge")
    desired.loras = [InferenceLoRA(str(path), 1.0)]
    assert desired.key() != loaded_key
    stack.remove()
    assert stack.loras == []
    assert torch.allclose(holder.model[0].weight, merged_weight)  # merge cleanup is reload-based


# ---------------------------------------------------------------- OPTIMIZER-01 exact dispatch

def _optimizer_stubs(monkeypatch):
    calls = []

    class FakeDAdaptAdam:
        def __init__(self, params, eps, lr, **kwargs):
            self.kind, self.eps, self.lr = "dadapt_adam", eps, lr

    class FakeDAdaptLion:
        def __init__(self, params, eps, lr, **kwargs):
            self.kind, self.eps, self.lr = "dadapt_lion", eps, lr

    class FakeLion:
        def __init__(self, params, lr, **kwargs):
            self.kind, self.lr = "lion", lr

    monkeypatch.setitem(sys.modules, "dadaptation", SimpleNamespace(
        DAdaptAdam=FakeDAdaptAdam, DAdaptLion=FakeDAdaptLion))
    monkeypatch.setitem(sys.modules, "lion_pytorch", SimpleNamespace(Lion=FakeLion))
    return calls


def test_dadaptationadam_and_lion_dispatch_to_their_real_optimizers(monkeypatch):
    from toolkit.optimizer import get_optimizer
    _optimizer_stubs(monkeypatch)
    params = [torch.nn.Parameter(torch.zeros(1))]
    assert get_optimizer(params, "dadaptationadam", learning_rate=0.01).kind == "dadapt_adam"
    assert get_optimizer(params, "dadaptationlion", learning_rate=0.01).kind == "dadapt_lion"
    assert get_optimizer(params, "dadaptation", learning_rate=0.01).kind == "dadapt_adam"
    assert get_optimizer(params, "lion", learning_rate=0.01).kind == "lion"


def test_bare_legacy_adam_stays_torch_adam(monkeypatch):
    from toolkit.optimizer import get_optimizer
    _optimizer_stubs(monkeypatch)
    params = [torch.nn.Parameter(torch.zeros(1))]
    optimizer = get_optimizer(params, "adam", learning_rate=0.01)
    assert isinstance(optimizer, torch.optim.Adam)
