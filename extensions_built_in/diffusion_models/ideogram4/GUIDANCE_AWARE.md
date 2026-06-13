# Guidance-aware LoRA training for Ideogram 4

## Why this exists

Ideogram 4 does not do classic single-model CFG. It ships **two** 9.3B
transformers — the conditional model and a separately refined
"unconditional" model — and samples with

```
v = v_uncond + g * (v_cond - v_uncond)
```

where the unconditional branch is *asymmetric*: it runs over image tokens
only with zeroed text features, on the dedicated negative checkpoint
(`unconditional_transformer/` in the official repo). The official presets use
`g = 7` for almost the whole trajectory (with a short `g = 3` polish tail).

That creates a dilemma for LoRAs trained on the conditional branch alone:

- **LoRA on cond only at inference:** the LoRA residual `δ` appears in the
  guided output as `g·δ` — amplified ~7x relative to how it was trained.
  Outputs fry.
- **LoRA on both models at inference:** `δ` cancels out of the differential,
  so the style lands at 1x while the *base* quality/aesthetic differential
  still pulls at full `(g-1)` strength. Coherent but muted — and the LoRA is
  an off-target patch on the unconditional weights anyway (different base
  weights, no text tokens in that pass).
- **Uncond disabled:** full style, but the quality vector that lives in the
  differential is gone.

`guidance_aware_training` removes the dilemma by baking the inference
operator into the loss. Each step runs the frozen unconditional model
forward (no grad), forms `v_uncond + g * (v_cond_lora - v_uncond)`, and
regresses **that** against the flow-matching target. The LoRA learns a
residual that is correct *under guidance*.

## How to use the resulting LoRA

- Apply it to the **conditional model only**.
- Leave the **unconditional model untouched** (no LoRA, full strength).
- Sample at the guidance scale you trained with (`train_guidance_scale`,
  default 7 — also the official default).

## Config

Everything lives under `model.model_kwargs` (see
`config/examples/train_lora_ideogram4_24gb_guidance_aware.yaml`):

| key | default | meaning |
| --- | --- | --- |
| `guidance_aware_training` | `false` | master switch for the guided loss |
| `train_guidance_scale` | `7.0` | number, or `[low, high]` for per-sample uniform sampling (e.g. `[3.0, 7.0]` to also cover the polish tail) |
| `guidance_grad_rescale` | `true` | divides the gradient by `g` (forward value unchanged) so gradient magnitudes — and therefore your usual LR recipes — match ordinary training. Without it, the chain rule multiplies gradients by `g`, which behaves like a 7x LR bump. |
| `guidance_aware_probability` | `1.0` | apply the guided loss on this fraction of steps; the rest use the plain conditional loss. A speed knob. |
| `load_unconditional_model` | = `guidance_aware_training` | set `true` on its own to load the negative model just for previews |
| `unconditional_path` | `null` | where the negative checkpoint lives; defaults to `model.name_or_path` |
| `unconditional_subfolder` | `"unconditional_transformer"` | matches the official HF repo layout |
| `unconditional_layer_offload_percent` | `1.0` | fraction of the negative model's linear layers kept on CPU and streamed. It is forward-only / no-grad, so full offload costs speed, not correctness. Set `0.0` to keep it resident if you have the VRAM. |
| `sample_with_unconditional` | `true` | training previews use the real dual-model asymmetric CFG instead of single-model CFG with an empty negative prompt |

## VRAM / 24 GB notes

- The unconditional model is quantized with the same settings as the main
  transformer (`model.quantize: true`) and then fully layer-offloaded by
  default, so its resident footprint is approximately zero; its weights
  stream from CPU during the extra forward pass.
- Its load path always stages weight dequantization through CPU (independent
  of `model.low_vram`), since the main transformer already occupies the GPU
  by the time it loads. Setting `model.low_vram: true` is still recommended
  so the *main* transformer's load is staged the same way.
- The negative pass is also cheaper than the positive one: it is image
  tokens only (no text region) and runs without gradients or activation
  checkpointing.
- Expect each training step to take roughly 1.5–2x as long as plain
  conditional training, dominated by streaming the offloaded weights.
- The unconditional model is never trained, never saved, and the LoRA
  network never attaches to it (networks attach via `get_model_to_train()`,
  which returns the conditional model).

## Caveats

- Loss values are not directly comparable to non-guided runs: the residual
  being regressed is the *guided* prediction error, which starts at the
  guidance overshoot of the base model on your data rather than near zero.
- Sampled previews ignore the negative prompt when
  `sample_with_unconditional` is on — the real negative branch takes no text
  at all.
- Not intended to be combined with trainer-side CFG-style options that
  concatenate unconditional embeddings into the training batch.
- The exact training recipe of the official unconditional checkpoint is not
  published; what is verifiable is that it is a separate set of weights used
  as the negative branch with the formula above (see
  `pipeline_ideogram4.py` in the official repo).
