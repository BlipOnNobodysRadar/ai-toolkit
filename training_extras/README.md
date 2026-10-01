# training_extras — opt-in package user guide

`training_extras` is an **opt-in** package for our existing AI Toolkit fork, ported from
`67372a/LoRA_Easy_Training_Scripts`. It is not enabled by default; existing recipes, presets
and checkpoints keep working untouched unless you explicitly turn an extras section on.

## 1. Installation addon dependency

The addon dependency is pinned:

```
pytorch-optimizer==3.10.1
```

**Do not reinstall Torch** to satisfy it. Torch is already provided by the Toolkit environment;
reinstalling it can break the CUDA/Torch pairing the host was built against.

Presets and options live under:

* `C:/Users/someb/AI/configs/training-extras/h3-comparisons` (16 presets)
* `C:/Users/someb/AI/configs/training-extras/krea2-comparisons` (16 presets)
* MiniCPM options: `C:/Users/someb/AI/configs/training-extras/minicpm-comparisons`.

## 2. Main Toolkit config fields

These fields belong under `config.process[0].train` in Toolkit YAML. Use the UI Advanced editor; importing a preset automatically selects it. Optimizer and scheduler fields remain native Toolkit fields. Loss options are nested under `training_extras`:

```yaml
optimizer: easy:WarpAINO
optimizer_params: {compile_step: false, meta_lr: 0}
lr_scheduler: easy:RAWR
lr_scheduler_params: {d: 0.45}
training_extras:
  loss: log_cosh
  anchor: {scale: 0.01}
  edm2: {enabled: true, channels: 448, float64: true, optimizer: 'easy:SCORN', lr: 0.015}
```

`optimizer: easy:WarpAINO` with `optimizer_params: {compile_step: false}` and
`lr_scheduler: easy:RAWR` with `d: 0.45` is the documented main-Toolkit wiring shape. Keep
`compile_step` false unless you have specifically validated compilation on your host.

The `TrainingExtras` runtime rejects unknown top-level keys. The accepted set is:

`loss`, `delta`, `delta_schedule`, `scale`, `loss_float64`, `x0_direct`, `anchor`, `hf`,
`frequency`, `topology`, `dynamic`, `edm2`, `wavelet`, `token_mining`.

Any other key is an error, so misspellings fail loudly instead of silently doing nothing.

### Loss (`loss`)

The bridge accepts these loss names:

`l2`, `l1`, `standard_pseudo_huber`, `standard_huber`, `standard_smooth_l1`, `huber`,
`smooth_l1`, `x_sigmoid`, `log_cosh`, `squared_logarithmic`, `soft_welsch`,
`scaled_quadratic`, `standard_deviation_loss`, `psnr_loss`, `geman_mcclure_loss`,
`smooth_l2_log`, `mse_pyramid_2d`.

`mse` and `mae` are accepted as aliases for `l2` and `l1`. That is **17 distinct losses** in
the source. `standard_deviation_loss`, `psnr_loss` and `geman_mcclure_loss` are scalar per-sample
objectives; the rest are element-wise, and `mse_pyramid_2d` has a video path when the prediction
is a 5-D BCTHW tensor.

### Signed-domain and negative-PSNR restrictions

Diffusion velocity / epsilon predictions are **signed** quantities. Two restrictions follow
from that and are enforced in source:

* `squared_logarithmic` requires predictions and targets greater than `-1`. It is unsuitable for
  signed diffusion velocities and raises when that domain is violated.
* `psnr_loss` is a negative-PSNR objective. It **cannot** be combined with EDM2, token mining, or
  a non-`none` dynamic auxiliary weight. Any of those combinations is rejected.

`delta` must be positive. `delta_schedule` is `constant`, `exponential` or `snr`; the `snr`
Huber schedule requires a DDPM scheduler and is rejected for flow-matching models.

### Auxiliary sources

The bridge composes five vendored auxiliary sources:

* `anchor_loss` (`anchor.levels`, `anchor.snr_weight`)
* `hf_token_loss` (high-frequency / token, `hf.patch`, `hf.exponent`)
* `focal_frequency_loss` (`frequency.*`)
* `patch_topology_loss` (`topology.*`)
* `element_weighting` (token mining + wavelet)

Each auxiliary has a `scale`; scales must be non-negative, and a zero scale means the auxiliary is
not constructed at all. Auxiliaries compute an x0 estimate using the source-convention mode
(`x0_direct`, `flow`, `vpred_ddpm`, `eps_ddpm`) via the vendored `hf_x0_hat` helper.

### Dynamic weighting (DWA / GradNorm)

`dynamic` wraps the vendored `DynamicLossWeighter`. When `dynamic.mode != none` **and** at least
one auxiliary is active, the combined auxiliary delta is reweighted by
`compute_weight(per_sample.mean(), aux.mean(), parameters)` and logged as `loss/dynamic_weight`.
There is no dynamic weighting without an active auxiliary — with no aux, `dynamic` is a no-op.

## 3. EDM2 — independent SCORN optimizer

EDM2 weighting trains its own small MLP with an **independent** optimizer and scheduler. Its LR
is separate from the model LR:

```json
{
  "edm2": {
    "enabled": true,
    "optimizer": "easy:SCORN",
    "lr": 0.015,
    "channels": 128,
    "importance": false
  }
}
```

The EDM2 default LR is `0.015` with default optimizer `easy:SCORN`. **This is not the model LR**;
do not copy the model LR into it. The aux optimizer is stepped via `TrainingExtras.step()`, and
zeroed independently after actual optimizer updates.

Scope limits you must know:

* EDM2 is **single GPU only**. `WORLD_SIZE>1` raises; DDP is unsupported.
* FP16 EDM2 and distributed EDM2 are **not** supported.
* EDM2 checkpoint restore checks a config hash and refuses a mismatched config.

### EDM CSV

On every `TrainingExtras.checkpoint(path)` with EDM2 enabled, the weights curve is written
alongside the checkpoint as `edm2-weighting.csv`, with columns
`timestep, sigma, log_variance, prediction_weight`. Render it with `python -m training_extras.plot_edm --csv <file> --out <png>`.

The **raw validation objective is tracked separately from the learned weights**. Logs keep
`loss/raw` distinct from `loss/edm_objective` and `loss/edm_data`.

## 4. Native held-out validation

Toolkit validation uses native dataset configurations under `train.validation_config.datasets`, each with a distinct `folder_path`. Include `validation_items: []`, `max_batches: 5`, `seed: 42`, `validation_sigmas: [0.25, 0.75]`, and `validate_every_n_steps: 25`. Plain image/video/audio datasets are encoded using the model's real loader. Initial, periodic, and final raw metrics are appended to `validation.jsonl`; video and audio MSE are separate. Validation runs at optimizer boundaries and preserves RNG and model modes.

Toolkit rejects identical train/validation folders, but does not detect duplicated or re-edited footage across different directories. Split by source video yourself. MiniCPM additionally rejects duplicate paths, file hashes, and matching `source_group` labels across its JSONL files. Crop variants from the same original are not independent held-out examples.

## 5. Flow matching: H3 and Krea2

H3 and Krea2 are flow-matching models and already have the canonical velocity signs handled in
source. Do not “fix” sign conventions in your config; the handled conventions are the ones the
vendored code expects. **Audio keeps native MSE**, and EDM2 is **video only**.

## 6. MiniCPM

MiniCPM is a different objective entirely. It uses:

* native audiovisual input
* caption cross-entropy (CE)
* frozen perception tower
* q/v adapters only

Adapters: `lora`, `dora`, `loha`, `lokr` (`--adapter-type`).

MiniCPM does **not** use diffusion losses, EDM2, topology, or anchor x0. Its anchor is a
**KL to the frozen base** (`caption_loss.kl_anchor`), not an x0 anchor.

MiniCPM config options are JSON, and it runs standalone:

```
python -m training_extras.minicpm_trainer --data <train> --val-data <val> --out <dir> --steps 2000
```

The CLI is:

`--data`, `--val-data`, `--out`, `--steps`, `--rank`, `--alpha`, `--lr`,
`--verify-reload`, `--options`, `--validate-every`, `--grad-accum`, `--checkpoint-every`,
`--resume`, `--adapter-type`, `--seed`.

`CaptionLoss` rejects spatial/diffusion keys (`edm2`, `anchor`, `frequency`, `hf`) — those are
invalid for a categorical token loss. Valid caption keys are `type`, `label_smoothing`,
`focal_gamma`, `z_loss`, `kl_anchor`, `temperature`; types are `ce`, `label_smoothed_ce`,
`focal_ce`.

## 7. Rank, LoRA+, DoRA/LoHa/LoKr, alpha

For Toolkit LoRA/DoRA/LoCon, set `network.lora_plus_ratio: 4` and optionally `network.lora_plus_text_encoder_ratio: 6`; block base learning rates are retained. To test alpha at rank/4, set `network.linear_alpha` and `network.respect_alpha: true`. Legacy Toolkit PEFT behavior forced alpha to rank; the new opt-in path bakes alpha/rank into exported B factors and reverses that scaling when loading training weights. Exported adapters retain the trained strength in existing inference loaders.

MiniCPM uses `--rank` and `--alpha`; `lora_plus_ratio` belongs in its JSON options and supports LoRA/DoRA only. Its PEFT adapter configuration carries scaling directly. LoHa/LoKr use their own factorization, with LoRA+ disabled. Resume checks type, scaling, options, prompt, and datasets; CPU/CUDA RNG and both optimizer/scheduler states are saved. MiniCPM preserves the original scheduler horizon when extending a run. Native GPU resume was finite and reloadable but not bitwise identical to uninterrupted training; use tolerance-based checks and record this limitation when interpreting comparisons.

## 8. SDXL original recipe — do not overwrite

The SDXL original recipe is preserved as-is. **Do not copy** `rank512 alpha1024` or `LR 0.0004`
to largeDiT or LLM training. Those values belong to the original SDXL recipe only.

## 9. Presets and source data

* `C:/Users/someb/AI/configs/training-extras/h3-comparisons` — 16 presets
* `C:/Users/someb/AI/configs/training-extras/krea2-comparisons` — 16 presets
* MiniCPM: `C:/Users/someb/AI/configs/training-extras/minicpm-comparisons` (9 options)
* Original Melinoe TOML: `C:/Users/someb/AI/configs/training-extras/melinoe-usual-original.toml`

**Krea2 data is still synthetic integration fixtures.** It must be replaced with real data before
any quality training. Treat those results as integration checks only.

## 10. Current scope and explicit non-goals

Supported now:

* single GPU
* frozen text encoder
* plain image/video conditioning for cached validation

Not supported / not implemented:

* guided, meanflow, t0, or turbo paths
* spatial masked auxiliaries
* FP16 EDM2
* distributed EDM2
* arbitrary model support (models are not claimed “fully supported”)

No quality gains are claimed. The catalog contains 102 imported optimizer subclasses and 2 schedulers. Selected optimizers were update-tested; importing an algorithm does not certify its complete runtime behavior. Closure, multistep, and wrapper paths require **host-specific integration** — they are
not drop-in here.

## 11. Licensing

The source code and its licenses are pinned. The backend is **GPLv3**; `sd-scripts` is
**Apache-2.0**, and both are retained. No licensing claim is made for combined redistributions —
if you redistribute a combined build, review the combined obligations yourself.


## Installed entry points

Toolkit: `C:/Users/someb/AI/apps/ai-toolkit`. Addon dependencies are in `requirements-training-extras.txt`; the existing Torch build is preserved. For a new host install addons into its existing environment with `python -m pip install -r requirements-training-extras.txt`.

MiniCPM wrapper: `C:/Users/someb/AI/video-captioning/training/train_lora_extras.py`, run with `C:/Users/someb/AI/envs/minicpm-o45/Scripts/python.exe`. This environment remains separate from Toolkit's newer Torch. Supply `--options` with a comparison JSON, `--data` and `--val-data` as JSONL rows containing `video`, `caption`, and preferably `source_group`. `--steps` counts optimizer updates; `--grad-accum` counts microbatches per update. The original `train_lora.py` is preserved. No ASR/TTS head training is implemented. Set the desired run horizon before training; resuming does not stretch the saved scheduler horizon.

The 16 diffusion presets isolate baseline, WarpAINO, compiled WarpAINO, REX d=.45, alpha at rank/4, LoRA+, EDM2, log-cosh, scheduled Huber, wavelet, token mining, mapped usual recipe, anchor, high frequency, focal frequency, and topology. The original TOML is an SDXL reference; `usual_mapped` preserves the destination model LR/rank and maps only the applicable optimizer/scheduler/weighting controls. IP noise, multiresolution noise, and SDXL parameter targeting are not blindly transplanted into the DiTs or language model.

Current Toolkit adapter implementation is used; the screenshot's experimental LyCORIS branch has not replaced the installed library. New branch-specific bypass/DoRA kernels require separate compatibility work. This port does not claim to reproduce every SDXL-only frontend option on every architecture. Existing Toolkit model-native controls remain available.

## Provenance and acceptance

Pinned commits, transformations, and vendor hashes are in `source-manifest.json`; optimizer constructor signatures and update coverage are in `optimizer-catalog.json`. See `C:/Users/someb/AI/training-extras/ACCEPTANCE.md` for fresh runtime evidence and usage accounting. Presets are hypotheses, not quality recommendations. Nothing in the comparison directories automatically starts a long training grid.
