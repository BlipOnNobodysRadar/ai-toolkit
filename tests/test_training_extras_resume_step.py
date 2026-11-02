"""training_extras.pt is restored only into the step it was saved for (sdtrainer-2). CPU only."""
import os
from types import SimpleNamespace

import pytest
import torch

from training_extras import runtime as rt

CFG = {"edm2": {"enabled": True, "channels": 16, "optimizer": "adamw"}}


def _sched():
    return SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))


def _extras():
    return rt.TrainingExtras(CFG, _sched(), True, "epsilon", torch.device("cpu"), 10)


def _trained(path, step):
    ex = _extras()
    loss = torch.tensor([1., 2.], requires_grad=True)
    value, _ = ex.augment(loss, torch.empty(0), torch.empty(0), torch.empty(0), torch.tensor([1., 500.]))
    value.mean().backward(); ex.step()
    ex.checkpoint(path, step=step)
    return ex


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_matching_step_restores(tmp_path):
    path = str(tmp_path / "training_extras.pt")
    ex = _trained(path, 150)
    fresh = _extras(); logs = []
    assert fresh.restore_for_step(path, 150, log=logs.append) == "restored"
    assert _same(ex, fresh) and not logs


def test_mismatched_step_raises(tmp_path):
    path = str(tmp_path / "training_extras.pt")
    _trained(path, 150)
    fresh = _extras(); before = [p.detach().clone() for p in fresh.parameters()]
    with pytest.raises(ValueError, match="saved at step 150 but training resumes at step 300"):
        fresh.restore_for_step(path, 300, log=lambda _: None)
    assert all(torch.equal(a, b) for a, b in zip(before, fresh.parameters()))
    # start_step override: warn and restore instead of raising
    logs = []
    assert fresh.restore_for_step(path, 300, strict=False, log=logs.append) == "restored" and logs


def test_fresh_start_moves_stale_file_aside(tmp_path):
    path = str(tmp_path / "training_extras.pt")
    ex = _trained(path, 150)
    fresh = _extras(); logs = []
    assert fresh.restore_for_step(path, 0, log=logs.append) == "stale"
    assert not os.path.exists(path) and any(n.startswith("training_extras.pt.stale-") for n in os.listdir(tmp_path))
    assert not _same(ex, fresh) and logs


def test_legacy_file_without_step_restores_with_warning(tmp_path):
    path = str(tmp_path / "training_extras.pt")
    ex = _trained(path, 150)
    data = torch.load(path, weights_only=True); del data["step"]; torch.save(data, path)
    fresh = _extras(); logs = []
    assert fresh.restore_for_step(path, 450, log=logs.append) == "legacy"
    assert _same(ex, fresh) and logs
