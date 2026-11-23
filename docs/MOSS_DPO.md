# Preference tuning MOSS-Image and Krea 2 Raw

This trainer supports two related preference workflows for ordinary text-to-image transformer LoRAs. They share a
frozen reference checkpoint and the same positive `dpo_beta` objective, but they obtain pairs differently.

- **Interactive ranked sample feedback** (the normal GUI workflow): a standard text-to-image transformer LoRA
  is trained with its ordinary loss while the job generates regular sample images. You rank those images later in
  the job gallery. This is also called interleaved or periodic online DPO.
- **Offline dataset DPO**: chosen and rejected image folders are paired by matching relative paths and trained as
  `loss_type: dpo`. This remains available for curated datasets and held-out scoring.

Both workflows train ordinary text-to-image transformer LoRAs. Preference updates use paired velocity errors.
Ordinary updates in an interleaved run can still use their dataset masks and loss weighting.

## Krea 2 Raw

Choose **Krea 2 (raw)** (`krea2`) for text-to-image preference training. This preset supports both gallery voting
and matched offline datasets. Use the base Raw weights with the assistant adapter field empty. Turbo and edit
checkpoints use different training inputs and are outside the supported presets.

## Interactive ranked sample feedback (GUI)

Use the normal **New Job** editor, choose a MOSS-Image or Krea 2 Raw text-to-image checkpoint, and keep the normal
target dataset folder. In the Training card, enable **Periodic ranked sample feedback (online DPO)**. This is a
separate switch from the Loss Type dropdown: MSE/MAE/Wavelet/Stepped Recovery remain selectable. Online feedback and
offline DPO cannot run together, so choosing offline DPO clears the online switch; choose an ordinary loss before
enabling periodic feedback.

When enabled, the GUI writes this object under `train`:

```yaml
train:
  loss_type: mse                 # any ordinary loss; loss_type: dpo is rejected here
  dpo_beta: 100                  # positive; default 100
  dpo_feedback:
    every: 10                    # successful optimizer updates
    start_after: 0               # successful optimizer updates
```

`train.dpo_feedback` absent or `null` means off. `every` counts successful optimizer updates, not raw steps, and
`start_after` also counts optimizer updates. The regular **Sample Every** setting in the Sample card is independent:
it controls how often ordinary gallery samples are generated. Feedback Every controls how often eligible voted pairs
can be reused for online preference training.

The switch does not replace the ordinary target folder. Because feedback needs real samples to rank, enabling it
turns sampling on. If the normal sample list has no usable prompt, the editor appends a fallback prompt; otherwise it
preserves the user's prompts, sample cadence, dimensions and other normal sample settings. You can edit the prompt
list before launching, and the saved job config is ordinary YAML/JSON that can be reopened in the GUI.

You can give `first_sample` its own prompts when caching text embeddings. Initial pairs use the cached
conditioning for that list, and later pairs use the regular sample list. For quantized jobs that use
an accuracy recovery adapter, the frozen reference records the loaded adapter. A different adapter
revision can make older votes ineligible when you resume.

### Voting is delayed

For example, with **Feedback Every** set to 10, a pair generated at update 100 can be ranked at update 137.
At update 140 the trainer can use your vote. The job keeps the two images, their training latents and prompt
conditioning. At training time it draws fresh shared noise and a timestep, scores the pair with the **current**
LoRA and with the frozen base, and applies the preference gradient to the current LoRA. No old gradient or
step-100 LoRA is needed.

Each scheduled slot takes one complete optimizer window. A slot with no eligible votes does ordinary training.
Eligible pairs rotate through later slots. Skip excludes a pair. A vote after the final update stays saved
and can train the model if the job can resume for more updates. RAWR checkpoints require the original
planned training length and scheduler configuration. With RAWR, rate pairs before the run ends, or pause
and resume within the existing plan. To use later ratings in a new offline DPO job, use the selected PNGs
and their original prompts as chosen/rejected pairs.

The reference identifier tracks the frozen base and encoding setup. Changing the frozen reference makes
its earlier votes ineligible. The learning rate, step budget and feedback cadence do not enter the vote
identity. The optimizer and scheduler can impose their own checkpoint compatibility requirements.

### Training progress in the feedback panel

When a job has enabled feedback (or still has published pairs), the Preference feedback panel may show a **Latest
training log** summary read from the job's existing `loss_log.db`. It includes the following metrics.

- `feedback/updates`: completed preference optimizer updates, cumulative at the latest recorded row.
- `feedback/optimizer_updates`: successful optimizer updates, cumulative at its latest recorded row.
- `feedback/foreign_votes`: aggregate incompatible votes found by the most recent recorded scan. The count reflects the latest recorded scan.

Each value is shown with the loss-log step that owns its row. The two update counts are cumulative. Logging can lag training, and a rewind can leave
older rows in the log. These job-wide counts report activity across all eligible pairs. If the log
is absent, disabled, unreadable, corrupt, or locked, the pair gallery and voting continue to work and the summary is
omitted. If votes exist before any feedback metric row, the panel only notes that the votes are recorded.

### On-disk pair contract

Each new independent sample pair belongs to the current job and is stored under:

```text
<save_root>/preferences/<id>/
```

`<id>` is a lowercase UUID4 hex string. The directory contains fixed filenames:

```text
pair.json
a.png
b.png
embeddings.safetensors
latents.safetensors
vote.json          # after voting
.decision/vote.json # durable decision used to recover an interrupted receipt write
```

`pair.json` contains exactly the metadata needed to interpret the displayed pair:

```json
{
  "version": 1,
  "id": "<lowercase uuid4 hex>",
  "step": 123,
  "optimizer_updates": 41,
  "ema": false,
  "network_multiplier": 1,
  "sampler": "flowmatch",
  "sampling_steps": 20,
  "guidance_scale": 3.5,
  "negative_prompt": "",
  "prompt": "the sample prompt",
  "seeds": [42, 43],
  "width": 1024,
  "height": 1024,
  "reference_id": "<frozen checkpoint/config identity>",
  "created_at": "2026-11-18T00:00:00Z"
}
```

`a.png` and `b.png` are the exact rated images. `embeddings.safetensors` is a single
`AdvancedPromptEmbeds.save` payload for the pair's prompt. `latents.safetensors` has keys `a` and `b`; each value is
a `[1, C, H, W]` tensor containing the normalized MOSS training `x0` for the corresponding displayed image. Keeping
the displayed pixels and the training tensors together makes the vote unambiguous.

`vote.json` is an immutable, atomic decision:

```json
{
  "version": 1,
  "id": "<same pair id>",
  "choice": "a",
  "voted_at": "2026-11-18T00:05:00Z"
}
```

`choice` is `a`, `b`, or `skip`. Re-voting the same choice is idempotent. The UI writes votes; the trainer sorts and
uses only eligible voted pairs. Pair IDs and metadata are validated. UI and data code treat every path and metadata
field as untrusted: files are accessed by fixed names resolved against the job's bound root, never by a path supplied
by the user or a pair manifest.

### Interpreting the objective

Each pair shares its prompt, timestep and sampled noise. The trainer predicts the flow velocity for both images with
the LoRA disabled (the frozen reference), then repeats that prediction with the trainable LoRA enabled. It computes
mean squared error per image in FP32. The objective follows the
[Diffusion-DPO implementation](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py) with the MOSS
flow target.

```text
gain(image) = reference_error(image) - policy_error(image)
margin      = (dpo_beta / 2) * (gain(chosen) - gain(rejected))
loss        = mean(softplus(-margin))
```

A fresh zero-initialized adapter should start near 0.693 with unit example weights. `dpo/margin` reports the mean
scaled margin and `dpo/accuracy` the fraction of pairs with a strictly positive margin on the logged microbatch. The
starting beta and learning rate are illustrative. Judge saved checkpoints on prompts held out from the preference
pairs to measure whether images improve.

## Offline dataset DPO

### From a batch of generations to a second LoRA

1. Generate PNGs for one positive prompt with your current LoRA. In
   [Blind LoRA Ranker](https://github.com/BlipOnNobodysRadar/blind-lora-ranker), open **Rate for DPO**,
   paste the prompt, and drop the PNGs or choose their folder. Give each image a score from 1 to 10.
2. Choose **Export DPO dataset**. In the Toolkit job editor, select offline DPO. Point the target dataset
   at the export's `chosen` folder and the rejected dataset at its `rejected` folder. For multiple
   prompts, export each batch and add one dataset entry per export.
3. Select the MOSS-Image or Krea 2 Raw text-to-image checkpoint used for generation. In **Frozen reference LoRA
   (optional)**, enter the path to the existing transformer LoRA on the training machine. Set **Reference LoRA
   strength** to the strength you generated with. Give the job a fresh name and leave
   `network.pretrained_lora_path` unset for a fresh preference adapter.
4. Load the original LoRA at that strength together with the newly saved LoRA in ComfyUI. The original sits
   on both sides of the DPO comparison during training. Only the new LoRA receives gradients and gets saved.
   The reference option accepts local safetensors with ordinary linear transformer LoRA weights.

The YAML equivalents are `train.dpo_reference_lora_path` and `train.dpo_reference_lora_strength` (default
`1.0`). The [offline example](../config/examples/train_dpo_moss_image.yaml) includes both settings. A resumed
job checks the original adapter's file digest and strength against the saved training state.

### How numerical ratings reach the loss

The ranker writes an optional `chosen/<stem>.dpo.json` for each pair. For example, a 9 versus a 3 produces
`{"version":1,"preference_probability":0.8333333333333333}`. The 1–10 scale maps to a soft training target
`p = 0.5 + (chosen_score - rejected_score) / 18`. This mapping expresses how strongly to favor the winner.

For a margin `m`, the per-pair loss is `p * softplus(-m) + (1-p) * softplus(m)`. At a fresh adapter's
zero margin, a 9 versus an 8 produces one ninth of the preference gradient of a 10 versus a 1. A soft
target also limits how far the objective pushes a near-tie once the model already favors the winner.
The loader accepts finite targets from 0.5 through 1. Missing sidecars use 1, preserving the binary
objective for existing folders and exports from the Toolkit gallery.

To reuse your gallery ratings, open **Job → Samples → Preference feedback** and select **Export DPO ZIP**. Extract the ZIP and point `folder_path` at `chosen/` and `rejected_path` at `rejected/`. Each A or B rating contributes the two original PNGs and their shared prompt. The download excludes skipped and pending pairs. You can use the dataset with another compatible text-to-image LoRA.

Put the image you want MOSS to favor in the chosen folder and its alternative in the rejected folder. Save the shared
prompt next to the chosen image.

```text
preferences/
  chosen/
    landscapes/001.png
    landscapes/001.txt
  rejected/
    landscapes/001.png
```

Use the same relative path, extension and displayed dimensions for both images. The loader keeps the pair together
through shuffling, bucketing and flipping. One unit of `batch_size` is one pair. The example caches both image latents
and the shared caption. Choose a fresh output name when starting a new experiment, as the trainer resumes existing
jobs by name.

Copy [the example](../config/examples/train_dpo_moss_image.yaml), set both dataset paths and select your MOSS
checkpoint. Launch it from the toolkit checkout with its usual Python environment.

```shell
python run.py config/examples/train_dpo_moss_image.yaml
```

`model.name_or_path` supplies the frozen reference. Both MOSS-Image Base and SFT use the same training path. To align
a previously fine-tuned model against itself, supply that merged checkpoint as the model and start a fresh LoRA.
Loading a trainable adapter with `network.pretrained_lora_path` initializes the policy only. During resume, keep the
same reference checkpoint and quantization settings.

## Scoring held-out preferences

Use separate chosen and rejected folders with the same layout. Keep all pairs from a given prompt in the same split.
Add this block under `train` to score a reproducible sample at fixed noise levels.

```yaml
validation_config:
  datasets:
    - folder_path: /path/to/held/chosen
      rejected_path: /path/to/held/rejected
      resolution: 512
  seed: 42
  max_batches: 16
  validation_sigmas: [0.25, 0.75]
  validate_every_n_steps: 25
```

`val/dpo/accuracy` measures how often the policy improves the chosen image's denoising error more than its mate's,
relative to the frozen reference. Ties count as zero. The job's `validation.jsonl` records the image ID, noise level,
margin and four underlying errors so you can inspect individual pairs.
