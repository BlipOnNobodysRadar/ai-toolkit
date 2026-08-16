# MiniMax-H3 Caption Lab (experimental)

This is an experimental attempt to turn MiniMax-H3 back around on unlabeled video. It is a lab tool, not a production captioner.

The pipeline is:

1. Qwen3-VL proposes several dense video captions.
2. Qwen3-VL is unloaded from the GPU.
3. MiniMax-H3 encodes the real clip to its own video/audio latent spaces.
4. For each caption, H3 receives the same noisy version of the real clip at several flow timesteps.
5. Captions are ranked by how much they improve H3's flow-prediction error relative to a blank-caption baseline.

The interesting part is step 5: H3 itself provides the ranking signal in the representation used for H3 training. The score is not a calibrated likelihood and still needs broad human validation.

## Current field-test status

The first real 4090 test succeeded end-to-end in separate stages:

- the official `Qwen/Qwen3-VL-8B-Instruct`, dynamically quantized to NF4, generated dense captions for a 14-second clip;
- H3 loaded with full text-encoder/transformer offload and scored the clip at a 256-pixel longest edge;
- at timestep 500, two plausible captions both ranked above an intentionally unrelated rainy-night sports-car caption;
- the more complete of the two plausible captions ranked first.

That is an encouraging sanity check, not proof that the score is reliable enough for automatic labeling.

The original default `unsloth/Qwen3-VL-32B-Instruct-bnb-4bit` checkpoint loaded on the same machine but failed during vision inference in BitsAndBytes with an uninitialized FP4 quantization state. The proven-good default is therefore now the official 8B model. `--qwen-model` remains available for experiments with other proposers.

## 24 GB design

The proposer and H3 scorer are never resident together. Qwen is unloaded before H3 loads. H3 uses the same fully-offloaded low-VRAM path as training, H3 scoring defaults to a 256-pixel longest edge, the VAEs are returned to CPU after latent extraction, and the DiT is parked on CPU while H3's 32B Qwen conditioner produces text embeddings.

Use the normal ai-toolkit environment. TorchCodec 0.13 is required for the optional audio path with the current PyTorch build.

## Full caption + rank

Run from the ai-toolkit repository root:

```bash
venv/bin/python tools/h3_caption_lab.py caption /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --write-sidecar
```

The script adds the repository root to `sys.path` itself, so `PYTHONPATH=.` is no longer required.

Default result names are command-specific:

- `generate` -> `clip.mp4.qwen.json`
- `score` -> `clip.mp4.h3score.json`
- `caption` -> `clip.mp4.h3caption.json`
- `--write-sidecar` -> `clip.txt`

Existing result JSON and sidecars are not overwritten unless `--overwrite` is explicitly supplied. `--output` can always be used to choose a different result path.

## Generate candidates without H3

```bash
venv/bin/python tools/h3_caption_lab.py generate /path/to/clip.mp4 \
  --candidates 4 \
  --qwen-video-tokens 1536
```

The default proposer is now:

```text
Qwen/Qwen3-VL-8B-Instruct
```

For a smaller first test:

```bash
venv/bin/python tools/h3_caption_lab.py generate /path/to/clip.mp4 \
  --candidates 2 \
  --qwen-video-tokens 1024
```

## Feed it the official H3 prompt manual

The built-in proposer instruction covers temporal order, subjects, environment, camera behavior, composition, lighting, and visible text, but it is not intended to replace MiniMax's prompt guide.

```bash
venv/bin/python tools/h3_caption_lab.py caption /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --guide /path/to/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
```

The guide is injected into the Qwen system instruction and capped at 40k characters.

## Score supplied captions

```bash
venv/bin/python tools/h3_caption_lab.py score /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --candidate "first caption" \
  --candidate "second caption"
```

A JSON produced by `generate`, or a text file with captions separated by blank lines, can be supplied directly:

```bash
venv/bin/python tools/h3_caption_lab.py score /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --candidate-file /path/to/clip.mp4.qwen.json
```

## Contrast / corruption sanity test

Before trusting the scorer on a dataset, compare a correct caption against deliberately controlled errors. Useful negatives include:

- the same scene with temporal order swapped;
- correct subjects but wrong camera behavior;
- correct action but wrong lighting/weather;
- correct scene but wrong clothing/colors;
- a deliberately vague caption;
- a completely unrelated caption.

For a stronger test than a single pass, average several noise seeds at all three default timesteps:

```bash
venv/bin/python tools/h3_caption_lab.py score /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --candidate-file /path/to/contrast_candidates.txt \
  --timesteps 250 500 750 \
  --score-repeats 3 \
  --no-audio-score
```

`--score-repeats N` repeats every timestep with independent matched noise while keeping exactly the same noise for all candidate captions in each pass.

## Reproducibility

H3's video VAE samples from a posterior during normal `encode_images`. Earlier prototype runs therefore used different clean latent samples even when the diffusion-noise seed was unchanged, which made repeated scores drift noticeably.

The lab now seeds the VAE posterior immediately before latent extraction. The default is:

```text
--latent-seed 1701
```

Diffusion noise remains controlled independently by:

```text
--score-seed 1776
```

The result JSON records the latent seed, score seed, timesteps, repeat count, scoring resolution, and audio settings. Exact bitwise equality is not promised across different CUDA/library builds, but repeated runs in the same environment should now be comparing the same latent sample and matched noise.

## 4090 defaults

```text
Qwen proposer:         Qwen/Qwen3-VL-8B-Instruct, dynamic NF4
Qwen video sampling:   2 fps
Qwen visual budget:    1536 tokens
H3 scoring edge:       256 px
H3 flow timesteps:     250, 500, 750
H3 score repeats:      1
H3 latent seed:        1701
H3 offload:            100% text encoder + transformer
candidate count:       4
```

For a fast proof of concept:

```bash
venv/bin/python tools/h3_caption_lab.py caption /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --candidates 2 \
  --qwen-video-tokens 1024 \
  --timesteps 500 \
  --no-audio-score
```

For more discriminating ranking, increase `--score-repeats`, include more timesteps, and/or increase `--score-max-edge`. All increase runtime; larger scoring resolution can also increase peak VRAM.

## What the H3 score means

For clean clip latent `x0`, sampled noise `n`, and flow noise level `sigma`, the tool constructs:

```text
x_sigma = (1 - sigma) * x0 + sigma * n
```

H3 predicts ai-toolkit's `noise - clean` velocity target. The tool measures MSE against the known target while using identical noise for every caption in a matched pass.

The reported video relative gain is:

```text
(blank_loss - caption_loss) / blank_loss
```

When audio scoring is enabled, video and audio improvements are normalized independently before being combined. `--audio-weight` controls the audio contribution.

The ranking therefore asks:

> Which candidate makes this specific H3 checkpoint better explain this specific observed clip?

That is potentially useful for H3 training captions, but it is not equivalent to recovering the original unknown prompt or evaluating general prose quality.

## Important limitations

- Qwen3-VL sees video, not soundtrack audio, so the proposer is instructed not to invent audio. H3 can still use the real soundtrack in the scoring signal. A later proposer path should add transcription and non-speech audio analysis.
- The scorer may prefer H3-specific wording quirks or excessive specificity. Human evaluation is required before bulk auto-labeling.
- The default scoring path uses the H3 training assistant LoRA. Use `--no-assistant-lora` to score the naked base checkpoint.
- The scorer resizes while preserving aspect ratio; it does not reproduce every ai-toolkit dataset bucketing/cropping choice.
- A single easy negative is not enough validation. The next useful experiment is a controlled corruption suite across many clips.

## Next experiments

1. Test controlled caption corruptions (temporal order, camera, lighting, attributes, vagueness, unrelated content) across 20-50 clips and multiple noise seeds.
2. Add iterative refinement: Qwen proposes variants of the current winner and H3 re-ranks them.
3. Add speech transcription and non-speech audio-event analysis; a Qwen3-Omni/other audio-capable proposer is a possible route.
4. Probe whether MiniMax's first 50 Qwen3-VL layers differ materially from stock Qwen3-VL-32B. If they do, experiment with restoring layers 50-63 + final norm + LM head and using the H3-adapted lower stack itself as an inverse captioner.
5. If the score continues to correlate with human preference, train a small inverse adapter from H3 video/audio features into a language decoder using synthetic `(H3 prompt, H3 generation)` pairs.
