"""Lazy compatibility hooks for fork-specific AI Toolkit features."""

import builtins
import os
import sys


def _patch_huggingface_downloader():
    """Route Hugging Face byte transfers through aria2 unless explicitly disabled."""
    if os.environ.get("AITK_HF_DOWNLOADER", "aria2").strip().lower() != "aria2":
        return False

    # huggingface_hub reads this during import. Force the regular HTTP path so
    # our aria2 transfer hook cannot be bypassed by hf-xet.
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    try:
        from toolkit.aria2_hf_download import patch_huggingface_downloads

        return patch_huggingface_downloads()
    except ModuleNotFoundError as exc:
        # During initial dependency installation huggingface_hub may not exist
        # yet. The next normal ai-toolkit process will patch it at startup.
        if exc.name == "huggingface_hub":
            return False
        raise


_patch_huggingface_downloader()


def _patch_dataset_config():
    module = sys.modules.get("toolkit.config_modules")
    dataset_config = getattr(module, "DatasetConfig", None) if module is not None else None
    if dataset_config is None:
        return False

    if getattr(dataset_config.__init__, "_blip_isolate_modalities_patch", False):
        return True

    original_init = dataset_config.__init__

    def dataset_config_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.isolate_modalities = kwargs.get("isolate_modalities", False)

    dataset_config_init._blip_isolate_modalities_patch = True
    dataset_config.__init__ = dataset_config_init
    return True


if not _patch_dataset_config():
    _original_import = builtins.__import__

    def _import_with_dataset_patch(name, globals=None, locals=None, fromlist=(), level=0):
        module = _original_import(name, globals, locals, fromlist, level)
        if name == "toolkit.config_modules" and _patch_dataset_config():
            builtins.__import__ = _original_import
        return module

    builtins.__import__ = _import_with_dataset_patch
