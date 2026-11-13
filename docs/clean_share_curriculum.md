# Opt-in clean-half timestep curriculum

Original implementation of the published Fizgig 8.8.1 epoch curriculum for
MOSS flow-matching training. This implements the schedule, not a claim of
training-quality parity with Fizgig or a recommendation to change all runs.
Algorithm reference: [Fizgig 8.8.1 announcement](https://www.reddit.com/r/StableDiffusion/comments/1xazk0f/fizgig_881_a_new_mossimage_training_technique/).
This is an independent implementation; no Fizgig source code was copied.

Existing configurations remain unchanged unless this field is set:

```yaml
train:
  noise_scheduler: flowmatch
  timestep_type: logit_normal
  content_or_style: balanced
  logit_normal_clean_schedule: "60:7:30:15:2"
  logit_normal_std: 0.95
  logit_normal_cutoff: 0.95
  logit_normal_uniform: 0.01
```

The five fields mean start-percent, hold-through-epoch, end-percent,
ramp-end-epoch, refresh-period. Epochs are one-based primary dataloader passes:
60% through epoch 7; linear ramp to 30% at epoch 15; 60% refresh at epochs
17, 19, ... and 30% otherwise. Refresh period 0 disables refreshes.

“Clean” means sigma < 0.5, where noisy latents are
`(1-sigma)*clean + sigma*noise`. Only the logit-normal mean changes. The solver
accounts for the configured standard deviation, truncation/rescaling cutoff
and uniform mixture. Targets describe ideal continuous probabilities, not exact
counts in a minibatch, finite timestep pool, or reduced-precision sampling.
The configured static mean is overridden during training, not mutated.

## Continuation and limitations

Each fetched microbatch carries its own epoch, including accumulation groups
that cross an epoch boundary. Checkpoints persist the consumed primary-batch
cursor and check schedule, sampler settings, dataset identity and loader length
on resume. Old checkpoints without this cursor cannot resume into a curriculum;
use a new output folder. Mid-epoch resumes reconstruct the loader offset. This
preserves curriculum phase, **not** exact shuffled sample order, augmentation,
RNG or bitwise results. Dataset identity uses the fork's existing file/config
fingerprints, not cryptographic content hashes of every training file.

Use `gradient_accumulation` for microbatch accumulation; legacy
`gradient_accumulation_steps` must remain 1 (no cross-step pending gradients).

Currently supported: one nonempty fixed-length primary loader, single process,
full-range balanced logit-normal flowmatch selection, no regularization loader,
no multistage/refiner, no first-timestep override or linear timestep overrides.
Unsupported combinations fail rather than silently misrepresent percentages.
Recorded stochastic-iterator exact resume is not supported together with this
feature; ordinary RAWR optimizer/scheduler checkpointing is not prohibited.
OOM aborts rather than skipping data. Incomplete batch groups cannot checkpoint
a curriculum cursor. Native held-out validation retains its explicit fixed
sigmas and never advances the training curriculum.

No GPU training-quality validation has been performed for this addition.
CPU tests cover phase boundaries, probability inversion, accumulation provenance
and resume guards. Run a short new-output training/resume smoke test before a
long experiment. Gradient clipping remains a separate `max_grad_norm` setting.

## Review evidence

An independent implementation review checked epoch provenance across accumulation,
resume offsets including exact epoch boundaries and JSON metadata roundtrips,
validation bypass, probability inversion and unsupported-mode guards. Eight CPU
unit/AST integration tests passed. Full PyTorch/GPU validation remains pending.
Project-required shared-board review and owner merge approval remain pending;
this review record is not a substitute for those gates.
