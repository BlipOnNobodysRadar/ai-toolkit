# Preference tuning MOSS-Image

Train a transformer LoRA from two images made for the same prompt. Put the image you want MOSS to favor in the chosen folder and its alternative in the rejected folder. Save the shared prompt next to the chosen image.

```text
preferences/
  chosen/
    landscapes/001.png
    landscapes/001.txt
  rejected/
    landscapes/001.png
```

Use the same relative path, extension and displayed dimensions for both images. The loader keeps the pair together through shuffling, bucketing and flipping. One unit of `batch_size` is one pair. The example caches both image latents and the shared caption. Choose a fresh output name when starting a new experiment, as the trainer resumes existing jobs by name.

Copy [the example](../config/examples/train_dpo_moss_image.yaml), set both dataset paths and select your MOSS checkpoint. Launch it from the toolkit checkout with its usual Python environment.

```shell
python run.py config/examples/train_dpo_moss_image.yaml
```

`model.name_or_path` supplies the frozen reference. Both MOSS-Image Base and SFT use the same training path. To align a previously fine-tuned model against itself, supply that merged checkpoint as the model and start a fresh LoRA. Loading a trainable adapter with `network.pretrained_lora_path` initializes the policy only. During resume, keep the same reference checkpoint and quantization settings.

## Interpreting the loss

Each pair shares its prompt, timestep and sampled noise. The trainer predicts the flow velocity for both images with the LoRA disabled, then repeats that prediction with the trainable LoRA enabled. It computes mean squared error per image in FP32. The objective follows the [Diffusion-DPO implementation](https://github.com/SalesforceAIResearch/DiffusionDPO/blob/main/train.py) with the MOSS flow target.

```text
gain(image) = reference_error(image) - policy_error(image)
margin      = (dpo_beta / 2) * (gain(chosen) - gain(rejected))
loss        = mean(softplus(-margin))
```

A fresh zero-initialized adapter should start near 0.693 with unit example weights. `dpo/margin` reports the mean scaled margin and `dpo/accuracy` the fraction of pairs with a strictly positive margin on the logged microbatch. The starting beta and learning rate in the example are illustrative. Judge saved checkpoints on prompts held out from the preference pairs to measure whether the images improve.

## Scoring held-out preferences

Use separate chosen and rejected folders with the same layout. Keep all pairs from a given prompt in the same split. Add this block under `train` to score a reproducible sample at fixed noise levels.

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

`val/dpo/accuracy` measures how often the policy improves the chosen image's denoising error more than its mate's, relative to the frozen reference. Ties count as zero. The job's `validation.jsonl` records the image ID, noise level, margin and four underlying errors so you can inspect individual pairs.

The first implementation supports text-to-image MOSS transformer LoRAs. Use ordinary image pairs with a frozen text encoder. MOSS edit conditioning, masks, auxiliary training losses and full-model tuning require separate support. The trainer uses one transformer for both forwards and keeps gradients only for the policy pass.
