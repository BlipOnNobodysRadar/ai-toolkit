"""A resumed run must draw exactly what an uninterrupted run draws (images, timesteps, noise) and end bit-identical.

Regression (found 26 Oct 2026): with a training_seed, every process start reseeded the global RNG and rebuilt the
shuffling sampler from it, and neither RNG state nor dataset position was restored. A resume at step k therefore
replayed steps 0.. exactly (same image, timestep and noise) on newer weights. train.reseed_per_step (default on)
reseeds each step from (seed, step), gives the sampler a per-epoch generator, and skips the batches the current
epoch already consumed.

Runs the real SDTrainer on CPU through testing/resume_rng_probe.py with the tiny SD fixture from the 19 Oct RAWR
resume review: 12 straight updates against 5 + a resume to 12, six 64 px images (two epochs, resume mid-epoch).
Skipped when the fixture is absent (set AITK_TINY_SD to its folder to override). Takes about a minute.
"""
import copy
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


def _config(tmp, name, folder):
    c = yaml.safe_load((TINY / "tiny-control.yaml").read_text(encoding="utf-8"))
    p = c["config"]["process"][0]
    c["config"]["name"] = name
    p["model"]["name_or_path"] = str(TINY / "tiny-sd")
    p["datasets"][0]["folder_path"] = str(tmp / "data")
    p["training_folder"] = str(tmp / folder)
    p["train"].update(steps=12)
    p["save"].update(save_every=4)
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    return path


def _run(cfg, evidence, *extra):
    env = dict(os.environ, ACCELERATE_USE_CPU="true", CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, str(REPO / "testing" / "resume_rng_probe.py"), "--repo", str(REPO), "--config", str(cfg),
                    "--evidence", str(evidence), *extra], check=True, env=env, cwd=REPO, timeout=900,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [json.loads(line) for line in (evidence / "draws.jsonl").read_text(encoding="utf-8").splitlines()]


def test_resume_draws_and_weights_match_uninterrupted(tmp_path):
    from PIL import Image
    (tmp_path / "data").mkdir()
    for i, (rgb, word) in enumerate([((220, 40, 40), "red"), ((40, 200, 60), "green"), ((40, 80, 220), "blue"),
                                     ((230, 200, 40), "yellow"), ((160, 60, 200), "purple"), ((40, 200, 200), "cyan")]):
        Image.new("RGB", (64, 64), rgb).save(tmp_path / "data" / f"{i}.png")
        (tmp_path / "data" / f"{i}.txt").write_text(f"a plain {word} square", encoding="utf-8")
    control = _run(_config(tmp_path, "straight", "out-straight"), tmp_path / "ev-straight")
    resumed_cfg = _config(tmp_path, "resumed", "out-resumed")
    first = _run(resumed_cfg, tmp_path / "ev-first", "--stop-after-save", "4")
    second = _run(resumed_cfg, tmp_path / "ev-second")

    key = lambda d: (tuple(d["files"]), tuple(d["timesteps"]), d["noise_sum"])
    straight = {d["step"]: key(d) for d in control}
    assert len(straight) == 12 and second[0]["step"] == len(first)
    assert {d["step"]: key(d) for d in first + second} == straight

    from safetensors.torch import load_file
    a = load_file(str(tmp_path / "out-straight" / "straight" / "straight.safetensors"))
    b = load_file(str(tmp_path / "out-resumed" / "resumed" / "resumed.safetensors"))
    assert a.keys() == b.keys() and all((a[k] == b[k]).all() for k in a)
