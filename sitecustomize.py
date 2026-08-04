"""Compatibility hooks for fork-specific AI Toolkit features."""

from toolkit.config_modules import DatasetConfig


if not getattr(DatasetConfig.__init__, "_blip_isolate_modalities_patch", False):
    _original_dataset_config_init = DatasetConfig.__init__

    def _dataset_config_init(self, *args, **kwargs):
        _original_dataset_config_init(self, *args, **kwargs)
        self.isolate_modalities = kwargs.get("isolate_modalities", False)

    _dataset_config_init._blip_isolate_modalities_patch = True
    DatasetConfig.__init__ = _dataset_config_init
