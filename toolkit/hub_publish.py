"""Files intended for model publication, kept separate from local trainer and resume state."""
import json
import re

import yaml
from huggingface_hub.utils import HFValidationError, validate_repo_id
from pathlib import Path


_COMPONENT_FILES = {
    "model_index.json", "config.json", "config.yaml", "scheduler_config.json", "generation_config.json",
    "preprocessor_config.json", "processor_config.json", "image_processor_config.json",
    "feature_extractor_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model",
    "sentencepiece.bpe.model", "chat_template.jinja",
}
_COMPONENT_WEIGHTS = re.compile(
    r"(?:model|diffusion_pytorch_model|pytorch_model)(?:[.-][\w-]+)*\.(?:safetensors|bin)(?:\.index\.json)?"
)
_COMPONENT_DIR = re.compile(
    r"(?:transformer|unet|vae|text_encoder|tokenizer|scheduler|feature_extractor|safety_checker|"
    r"image_encoder|image_processor|text_conditioner|dit_model)(?:_\d+)?"
)


def _model_directory(path):
    # Torch and Diffusers write manifests and tokenizer files alongside weights in the
    # model root or its named components. Other directories can hold arbitrary user files.
    for item in path.iterdir():
        if item.is_symlink() or item.resolve().parent != path.resolve():
            continue
        if item.is_dir() and _COMPONENT_DIR.fullmatch(item.name):
            candidates = item.iterdir()
        else:
            candidates = [item]
        for file in candidates:
            if file.is_symlink() or not file.is_file():
                continue
            if file.name in _COMPONENT_FILES or _COMPONENT_WEIGHTS.fullmatch(file.name):
                yield file


def model_files(root, kinds, arch):
    """Yield only this run's final inference artifacts, preserving their repo-relative names."""
    root = Path(root)
    files = {}
    for kind, name, suffixes in kinds:
        if kind == "critic" or Path(name).name != name:
            continue
        if kind == "model" and arch == "wan22_14b":
            suffixes = ("_high_noise", "_low_noise")
        for suffix in suffixes:
            stem = name + suffix
            if Path(stem).name != stem:
                continue
            candidates = [root / (stem + ".safetensors")]
            if kind == "embedding" and stem.lower() not in {"optimizer", "training_extras"}:
                candidates.append(root / (stem + ".pt"))
            candidates.append(root / stem)
            for asset in candidates:
                if asset.is_symlink() or asset.resolve().parent != root.resolve():
                    continue
                if asset.is_file() and asset.suffix in {".safetensors", ".pt"}:
                    files[asset.name] = asset
                elif asset.is_dir() and kind in {"model", "adapter", "refiner"}:
                    # Native writers can create a directory at the exact requested .safetensors path.
                    for file in _model_directory(asset):
                        files[file.relative_to(root).as_posix()] = file
    return files


def file_state(root, kinds, arch):
    state = {}
    for name, path in model_files(root, kinds, arch).items():
        stat = path.stat()
        state[name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return state


def repo_reference(value):
    try:
        validate_repo_id(value)
    except HFValidationError:
        return None
    return None if Path(value).exists() else value


def publish_payload(path, arch):
    if arch == "iris3b" and path.name == "config.yaml":
        # The local writer preserves the source checkpoint's entire config. Inference only
        # consumes the architecture, text encoder and flow settings.
        with path.open(encoding="utf-8") as handle:
            source = yaml.safe_load(handle)
        encoder = source.get("text_encoder", {})
        text_fields = {key: encoder[key] for key in ("name", "dim", "max_length", "hidden_layers") if key in encoder}
        if pretrained := repo_reference(encoder.get("pretrained")):
            text_fields["pretrained"] = pretrained
        flow = source.get("flow", {})
        flow_fields = {key: flow[key] for key in ("num_train_timesteps", "shift", "prediction", "x_pred_sigma_min") if key in flow}
        return yaml.safe_dump({"model": source["model"], "text_encoder": text_fields, "flow": flow_fields}).encode()
    # Diffusers and Transformers can save the original private checkpoint location in their standard configs.
    if path.name not in {"config.json", "model_index.json", "tokenizer_config.json"}:
        return path
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    def remove_source(item):
        if isinstance(item, list):
            for value in item:
                remove_source(value)
        if isinstance(item, dict):
            item.pop("_name_or_path", None)
            item.pop("name_or_path", None)
            for value in item.values():
                remove_source(value)
    remove_source(config)
    return json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")
