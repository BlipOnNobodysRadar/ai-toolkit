"""A resumed run must draw exactly what an uninterrupted run draws (images, timesteps, noise) and end bit-identical.

Regression (found 26 Oct 2026): with a training_seed, every process start reseeded the global RNG and rebuilt the
shuffling sampler from it, and neither RNG state nor dataset position was restored. A resume at step k therefore
replayed steps 0.. exactly (same image, timestep and noise) on newer weights. train.reseed_per_step (default on)
reseeds each step from (seed, step), gives the sampler a per-epoch generator, and continues the epoch at the right
batch; checkpoints also record completed_steps so non-RAWR resumes don't repeat the last update.

Runs the real SDTrainer on CPU through testing/resume_rng_probe.py with the tiny SD fixture from the 19 Oct RAWR
resume review: straight vs stop-after-save + resume, comparing every micro-batch draw of every step. Cases: mid-epoch,
an exact epoch-boundary resume with batch 2 x GA 2, two aspect buckets with batch 2, and a non-RAWR scheduler.
Skipped when the fixture is absent (set AITK_TINY_SD to its folder to override). About a minute per case.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TINY = Path(os.environ.get("AITK_TINY_SD", r"C:\Users\someb\Documents\Codex\2026-09-09\create-an-image-of\outputs"
                                          r"\windows-ai\training\rawr-resume-review-20261019"))
pytestmark = pytest.mark.skipif(not (TINY / "tiny-sd" / "model_index.json").exists(), reason="tiny SD fixture not found")
COLORS = [(220, 40, 40), (40, 200, 60), (40, 80, 220), (230, 200, 40), (160, 60, 200), (40, 200, 200), (250, 120, 30),
          (90, 90, 90)]
# name: (image sizes, train overrides, steps, save_every); the first run stops after its first save
CASES = {
    "mid_epoch": ([(64, 64)] * 6, {}, 12, 4),
    "epoch_boundary_b2_ga2": ([(64, 64)] * 8, {"batch_size": 2, "gradient_accumulation": 2}, 8, 3),
    "two_buckets_b2": ([(64, 64)] * 4 + [(64, 96)] * 4, {"batch_size": 2}, 10, 4),
    "non_rawr": ([(64, 64)] * 6, {"lr_scheduler": "constant"}, 12, 4),
}


def _config(tmp, name, train, steps, save_every):
    c = yaml.safe_load((TINY / "tiny-control.yaml").read_text(encoding="utf-8"))
    p = c["config"]["process"][0]
    c["config"]["name"] = name
    p["model"]["name_or_path"] = str(TINY / "tiny-sd")
    p["datasets"][0]["folder_path"] = str(tmp / "data")
    p["training_folder"] = str(tmp / f"out-{name}")
    p["train"].update(steps=steps, **train)
    if train.get("lr_scheduler", "easy:RAWR") != "easy:RAWR":
        p["train"].pop("lr_scheduler_params", None)
    p["save"].update(save_every=save_every)
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    return path


def _run(cfg, evidence, *extra):
    env = dict(os.environ, ACCELERATE_USE_CPU="true", CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, str(REPO / "testing" / "resume_rng_probe.py"), "--repo", str(REPO), "--config", str(cfg),
                    "--evidence", str(evidence), *extra], check=True, env=env, cwd=REPO, timeout=1500,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [json.loads(line) for line in (evidence / "draws.jsonl").read_text(encoding="utf-8").splitlines()]


def _per_step(draws):
    out = {}
    for d in draws:  # every micro-batch, in order
        out.setdefault(d["step"], []).append((tuple(d["files"]), tuple(d["timesteps"]), d["noise_sum"]))
    return out


@pytest.mark.parametrize("case", list(CASES))
def test_resume_draws_and_weights_match_uninterrupted(tmp_path, case):
    from PIL import Image
    from safetensors.torch import load_file
    sizes, train, steps, save_every = CASES[case]
    (tmp_path / "data").mkdir()
    for i, (size, rgb) in enumerate(zip(sizes, COLORS)):
        Image.new("RGB", size, rgb).save(tmp_path / "data" / f"{i}.png")
        (tmp_path / "data" / f"{i}.txt").write_text(f"a plain square number {i}", encoding="utf-8")
    straight = _run(_config(tmp_path, "straight", train, steps, save_every), tmp_path / "ev-straight")
    resumed_cfg = _config(tmp_path, "resumed", train, steps, save_every)
    first = _run(resumed_cfg, tmp_path / "ev-first", "--stop-after-save", str(save_every))
    second = _run(resumed_cfg, tmp_path / "ev-second")

    a, b = _per_step(straight), _per_step(first + second)
    assert len(a) == steps and second[0]["step"] == max(_per_step(first)) + 1, "resume must continue after the save"
    assert [k for k in a if a[k] != b.get(k)] == []

    wa = load_file(str(tmp_path / "out-straight" / "straight" / "straight.safetensors"))
    wb = load_file(str(tmp_path / "out-resumed" / "resumed" / "resumed.safetensors"))
    assert wa.keys() == wb.keys() and all((wa[k] == wb[k]).all() for k in wa)
