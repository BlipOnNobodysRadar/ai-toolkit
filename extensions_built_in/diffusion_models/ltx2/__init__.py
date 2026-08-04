from .ltx2 import LTX2Model, LTX23Model


# Preserve the fork's dataset-level LTX modality isolation without replacing
# upstream's rapidly evolving LTX implementation.
if not getattr(LTX2Model.get_noise_prediction, "_blip_isolate_modalities_patch", False):
    _original_get_noise_prediction = LTX2Model.get_noise_prediction

    def _get_noise_prediction_with_dataset_isolation(self, *args, **kwargs):
        batch = kwargs.get("batch")
        if batch is None and len(args) >= 4:
            batch = args[3]

        dataset_config = getattr(batch, "dataset_config", None)
        isolate_modalities = bool(getattr(dataset_config, "isolate_modalities", False))
        if not isolate_modalities:
            return _original_get_noise_prediction(self, *args, **kwargs)

        transformer = self.transformer
        original_forward = transformer.forward

        def forward_with_isolation(*forward_args, **forward_kwargs):
            forward_kwargs["isolate_modalities"] = True
            return original_forward(*forward_args, **forward_kwargs)

        transformer.forward = forward_with_isolation
        try:
            return _original_get_noise_prediction(self, *args, **kwargs)
        finally:
            transformer.forward = original_forward

    _get_noise_prediction_with_dataset_isolation._blip_isolate_modalities_patch = True
    LTX2Model.get_noise_prediction = _get_noise_prediction_with_dataset_isolation
