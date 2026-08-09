# MiniMax-H3 Caption Lab (experimental)

This is a first-pass attempt to turn MiniMax-H3 back around on unlabeled video. It is intentionally a lab tool rather than a production captioner.

The pipeline is:

1. Qwen3-VL proposes several dense video captions.
2. Qwen3-VL is unloaded from the GPU.
3. MiniMax-H3 encodes the real clip to its own video/audio latent spaces.
4. For each caption, H3 receives the same noisy version of the real clip at several flow timesteps.
5. The caption is ranked by how much it improves H3's flow-prediction error relative to a blank-caption baseline.

That last step is the interesting part: H3 itself decides which text better explains the clip in the representation used for H3 training. The score is still experimental; it is not a calibrated likelihood and should be validated against human preference before trusting it at scale.

## Why it should fit a 24 GB card

The proposer and scorer are never resident together. The default proposer is a pre-quantized 4-bit Qwen3-VL-32B model. H3 uses the same fully-offloaded path as low-VRAM training, and H3 scoring defaults to a 256-pixel longest edge. The H3 video/audio VAEs are moved back to CPU after latent extraction, and the DiT is parked on CPU whenever the H3 Qwen conditioner is needed.

A 4090 is therefore a plausible target, but the 32B proposer is the tightest stage. If it OOMs on a particular clip, reduce `--qwen-video-tokens`, reduce `--qwen-fps`, or temporarily use an 8B Qwen3-VL proposer. H3 scoring is independent of which model proposed the captions.

## Requirements

Use the normal ai-toolkit environment. For the current PyTorch 2.13 builds, TorchCodec 0.13 is required for the optional audio decode path; this branch updates the stale 0.9.1 pin.

The default Qwen proposer is:

```text
unsloth/Qwen3-VL-32B-Instruct-bnb-4bit
```

It is a separate download from H3's truncated/quantized Qwen conditioner.

## Full caption + rank

Run from the ai-toolkit repository root:

```bash
venv/bin/python tools/h3_caption_lab.py caption /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --write-sidecar
```

This writes:

- `clip.mp4.h3caption.json` — all candidates, per-pass losses, and ranking.
- `clip.txt` — the winning caption, only when `--write-sidecar` is given.

Existing `.txt` captions are never overwritten unless `--overwrite` is also supplied.

## Feed it the official H3 prompt manual

The built-in proposer instruction covers the important semantic categories but is not intended to replace MiniMax's prompt guide. If you have the official base prompt guide locally:

```bash
venv/bin/python tools/h3_caption_lab.py caption /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --guide /path/to/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
```

The guide is injected into the Qwen system instruction (capped at 40k characters).

## Generate candidates without H3

Useful to inspect Qwen before spending time on H3 scoring:

```bash
venv/bin/python tools/h3_caption_lab.py generate /path/to/clip.mp4 \
  --candidates 4
```

## Score captions you already wrote

```bash
venv/bin/python tools/h3_caption_lab.py score /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --candidate "first caption" \
  --candidate "second caption"
```

Or put captions in a text file separated by blank lines:

```bash
venv/bin/python tools/h3_caption_lab.py score /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --candidate-file candidates.txt
```

## 4090 knobs

The conservative defaults are:

```text
Qwen video sampling: 2 fps
Qwen visual budget:   1536 tokens
H3 scoring edge:      256 px
H3 flow timesteps:    250, 500, 750
H3 offload:           100% text encoder + transformer
candidate count:      4
```

For a faster proof-of-concept:

```bash
venv/bin/python tools/h3_caption_lab.py caption /path/to/clip.mp4 \
  --models-path /path/to/ComfyUI/models \
  --candidates 2 \
  --qwen-video-tokens 1024 \
  --timesteps 500 \
  --no-audio-score
```

For more discriminating H3 ranking, add timesteps and/or use a larger `--score-max-edge`. That increases runtime and peak memory.

## What the H3 score means

For clean clip latent `x0`, sampled noise `n`, and flow noise level `sigma`, the tool constructs:

```text
x_sigma = (1 - sigma) * x0 + sigma * n
```

H3 predicts ai-toolkit's `noise - clean` velocity target. The tool measures MSE against the known target using identical noise for every caption. It repeats this at several timesteps.

The reported `relative_gain` is the matched relative improvement over an empty caption:

```text
(blank_loss - caption_loss) / blank_loss
```

Video and audio improvements are normalized independently before being combined, so their raw MSE scales do not have to match. `--audio-weight` controls the audio contribution.

The ranking therefore asks a narrower question than normal caption quality:

> Which candidate makes this specific H3 checkpoint better explain this specific observed clip?

That is exactly why it is potentially useful for H3 training captions, but it remains an empirical hypothesis until tested.

## Important limitations

- Qwen3-VL sees video, not soundtrack audio. The candidate generator is therefore instructed not to invent audio. H3 can still use the real soundtrack in the ranking signal. A later stage should add transcription and non-speech audio analysis.
- H3's score can prefer wording quirks or overly specific captions. Human evaluation is needed before bulk auto-labeling.
- The default scoring path uses the H3 training assistant LoRA, matching the normal H3 training configuration. Use `--no-assistant-lora` to score the naked base checkpoint instead.
- The first version resizes the clip for scoring while preserving aspect ratio. It does not reproduce every ai-toolkit dataset bucketing/cropping choice.
- No claim is made that this recovers the original unknown prompt that produced a clip. It ranks captions by conditional flow prediction error.

## Next experiments if this signal works

1. Compare H3 ranking against human ranking on 20-50 clips with deliberately good/bad/camera-wrong/temporally-wrong captions.
2. Add iterative refinement: let Qwen see the current winner and produce targeted variants, then re-rank.
3. Add Whisper transcription plus an audio-event model.
4. Probe whether MiniMax's first 50 Qwen3-VL layers differ materially from stock Qwen3-VL-32B. If they do, experiment with restoring layers 50-63 + norm + LM head and using the H3-adapted lower stack as the caption proposer itself.
5. If the scorer is useful, train a small inverse adapter from H3 video/audio features into a language decoder using synthetic `(H3 prompt, H3 generation)` pairs.
