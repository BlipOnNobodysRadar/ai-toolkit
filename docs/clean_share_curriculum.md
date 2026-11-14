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
CPU tests cover phase boundaries, probability inversion against the real Torch
sampler, accumulation provenance and resume guards. Real tiny-model CPU trainer
runs also preserve curriculum phase through seeded mid-epoch, unseeded
epoch-boundary and seeded RAWR save/resume paths. These fixtures do not establish
MOSS quality, GPU behavior or bitwise continuation. Run a short new-output MOSS
training/resume smoke test before a long experiment. Gradient clipping remains a
separate `max_grad_norm` setting; this feature does not change clipping defaults.

## Review evidence

The 14 November CPU validation includes 68 targeted tests and three uninterrupted
versus save/resume comparisons through the real trainer (nine executions total).
All runtime probes reported CUDA uninitialized. The sampler tests cover 200,000
draws per case for hold, ramp, end and refresh targets with and without cutoff.
Shared-board independent review is recorded separately from these test results.
There is no PR-triggered CI workflow in this checkout; absent checks are not a
passing CI result.
