"""Train with explicitly uploaded Modal inputs and store checkpoints on a Modal Volume."""

import os
from pathlib import Path
import subprocess
import sys

import modal


REMOTE_ROOT = "/root/ai-toolkit"
OUTPUT_DIR = f"{REMOTE_ROOT}/modal_output"
SOURCE_DIRS = ("jobs", "toolkit", "extensions_built_in", "extensions/example", "training_extras")
SOURCE_FILES = ("run.py", "version.py", "info.py")
RUNTIME_DATA = {
    "toolkit/audio/ostris_logo.jpg",
    "training_extras/optimizer-catalog.json",
    "training_extras/optimizer-index.json",
    "training_extras/source-manifest.json",
}
REQUIREMENTS = ("requirements_base.txt", "requirements-training-extras.txt")


def training_image():
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
        .env({
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DISABLE_TELEMETRY": "YES",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        })
        .pip_install("torch", "torchvision", "torchaudio")
    )
    if not modal.is_local():
        return image

    root = Path(__file__).resolve().parent
    paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--", *SOURCE_DIRS, *SOURCE_FILES, *REQUIREMENTS],
    )
    tracked = {Path(os.fsdecode(name)) for name in paths.split(b"\0") if name}

    def regular_file(relative):
        target = root / relative
        return relative in tracked and not target.is_symlink() and target.is_file() and target.resolve() == target

    for name in REQUIREMENTS:
        if not regular_file(Path(name)):
            raise ValueError(f"Expected a tracked requirements file at {name}")
        image = image.pip_install_from_requirements(str(root / name))

    for name in SOURCE_FILES:
        if not regular_file(Path(name)):
            raise ValueError(f"Expected a tracked runtime file at {name}")
        image = image.add_local_file(root / name, f"{REMOTE_ROOT}/{name}")

    def ignore(relative, folder):
        path = folder / relative
        full = root / path
        if full.is_dir():
            return full.is_symlink()
        if not regular_file(path):
            return True
        if path.suffix == ".py" or path.as_posix() in RUNTIME_DATA:
            return False
        if path.parent == Path("toolkit/keymaps"):
            return path.suffix not in {".json", ".safetensors"}
        if path.parent == Path("toolkit/orig_configs"):
            return path.suffix not in {".yaml", ".yml"}
        return True

    for folder in map(Path, SOURCE_DIRS):
        if (root / folder).resolve() != root / folder:
            raise ValueError(f"Unexpected symlink at runtime source directory {folder}")
        image = image.add_local_dir(
            root / folder, f"{REMOTE_ROOT}/{folder.as_posix()}",
            ignore=lambda path, folder=folder: ignore(path, folder),
        )
    return image


app = modal.App("flux-lora-training")
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
hf_secret = os.environ.get("AITK_MODAL_HF_SECRET")


@app.function(
    image=training_image(),
    gpu="A100",
    timeout=7200,
    volumes={
        f"{REMOTE_ROOT}/config": modal.Volume.from_name("ai-toolkit-configs"),
        f"{REMOTE_ROOT}/datasets": modal.Volume.from_name("ai-toolkit-datasets"),
        OUTPUT_DIR: model_volume,
    },
    secrets=[modal.Secret.from_name(hf_secret, required_keys=["HF_TOKEN"])] if hf_secret else [],
)
def main(config_file_list_str: str, recover: bool = False, name: str = ""):
    configs = [path.strip() for path in config_file_list_str.split(",") if path.strip()]
    if not configs:
        raise ValueError("Supply at least one config path on the Modal config volume.")
    command = [sys.executable, f"{REMOTE_ROOT}/run.py"]
    if recover:
        command.append("--recover")
    if name:
        command.extend(["--name", name])
    try:
        subprocess.run([*command, "--", *configs], cwd=REMOTE_ROOT, check=True)
    finally:
        model_volume.commit()
