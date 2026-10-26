# RAWR checkpoint continuation

RAWR stores its cycle position in optimizer parameter groups. Loading those
groups, resetting their LR, constructing RAWR, and taking the trainer startup
step used to advance the saved cycle twice. Loading only scheduler.state_dict
is insufficient because the optimizer groups also carry scheduling state.

`training_extras/rawr_checkpoint.py` captures loaded groups before mutation,
restores scheduler and group state together after construction, and tells the
trainer to skip the startup scheduler step on a restored run. It recognizes
Accelerate's actual scheduler wrapper. Vendor RAWR arithmetic is unchanged.

New optimizer.pt files contain versioned scheduler state, group metadata,
declared configuration, and the model metadata step. Version 2 includes pending
gradients and the next accumulation counter. Version 3 also records stochastic
continuation state (or an explicit unsupported-mode reason). Older code rejects
new versions instead of silently dropping the pending work. Changed LR/configuration,
step mismatches, unsupported versions, and inconsistent groups are rejected.
RAWR optimizer load/save errors propagate instead of silently continuing.
Resuming model metadata beyond step zero also requires its optimizer.pt;
weights-only adaptation should use a new output folder/pretrained LoRA input.
Other schedulers retain their existing behavior and checkpoint payloads.

Periodic saves occur before the training loop increments its zero-based label.
New RAWR metadata records the completed loop count so resume does not repeat
an update. An explicitly tracked completed count also handles UI save requests
from the end-step hook; the filename is not used to infer the resume position.
These are completed **loop iterations**, not optimizer updates when accumulation
is enabled. RAWR continues to advance once per loop iteration, matching the
existing scheduler horizon.

Fixed-count gradient accumulation is restored after startup clears gradients
and prepares Accelerate. Periodic saves preserve gradients in the running
process too. Changing either accumulation setting is rejected. Whole-epoch
accumulation (`gradient_accumulation_steps: -1`) cannot safely resume without
the dataset iterator and is rejected explicitly.

RAWR optimizer writes use a same-directory temporary file, flush/fsync, and
atomic replacement. A serialization or replacement failure leaves the previous
optimizer file readable. Plain LoRA training also uses a committed pair:

- `.rawr-checkpoints/a` and `b` alternate as inactive/active slots. The active
  slot is never written during a save. Internal storage is bounded to two pairs.
- `optimizer.pt` in the output folder is a hard link to the committed slot's optimizer where the volume allows it
  (NTFS, ext4), and a copy otherwise. That saves a full optimizer write per save: about 3 minutes for a 7 GB
  optimizer on a 40 MB/s drive. Two consequences:
  - **It is a live alias.** Anything that writes into `optimizer.pt` in place (`torch.save` to that path, opening
    it `r+b`, copying *over* it) also changes the committed slot. Native resume then refuses the pair on its hash
    check. Copy it elsewhere before modifying it. Every optimizer writer in the trainer swaps in a new file instead.
  - **Copies don't keep links.** Copying, zipping or moving the folder to another volume stores the optimizer twice,
    and folder-size tools count it twice. For a resumable archive, `rawr-resume.json` plus `.rawr-checkpoints` is
    enough.
  - A reader holding `optimizer.pt` open only delays the alias update (a warning; the pair is committed). On NTFS it
    doesn't block rewriting the slot.
- `rawr-resume.json` is replaced only after model and optimizer files are closed,
  flushed, and hashed. Its replacement commits both files as one resume point.
- Native resume selects this manifest, verifies both SHA-256 hashes and the
  generation in model metadata, then loads the optimizer from the same slot.
- Normal model filenames and `optimizer.pt` remain export aliases, published
  after commit. An alias-update failure warns; it cannot invalidate the committed
  resume point. Copy the manifest and the `.rawr-checkpoints` directory when
  moving a resumable run. A standalone new model export can still initialize a
  new run through `pretrained_lora_path`; it cannot masquerade as a paired resume.

This paired protocol covers one plain LoRA file and its optimizer. EMA, merged
full models, auxiliary optimizers, embeddings, decorators, adapters, and learnable
SNR files are not part of that transaction. Unsupported save modes cannot replace
an existing committed pair in place; use a new output folder. Legacy directories
without a manifest keep their existing selection behavior. An interrupted first
paired save does not overwrite legacy exports. Concurrent writers/readers of a
slot and power-loss durability of directory metadata remain unverified; process
interruption before commit is covered by an actual trainer subprocess test.

SimplifiedAdEMAMixExM supports non-compiled CPU updates when CUDA is unavailable,
and restores its moment buffers to the configured storage dtype/device instead
of retaining PyTorch's parameter-dtype cast. Set `CUDA_VISIBLE_DEVICES=''` before
importing Torch for CPU-only runs on a machine with a GPU.

## Legacy checkpoints

Optimizer-only RAWR files retain their saved LR and cycle fields with a warning.
Their original d/gamma/cycle multiplier and exact model pairing cannot be
certified. Subsequent scheduling uses the current configuration. Missing cycle
fields and a known changed initial LR are rejected. Old zero-based labels are
not silently reinterpreted: their original save phase cannot be inferred.
With fixed accumulation greater than one, optimizer-only legacy files are
rejected because they cannot identify a safe boundary. Version-1 RAWR files
can resume at a known completed-loop boundary, but are rejected mid-group
because they contain no unfinished gradients. Accumulation-one legacy loading
retains its warning and continuation behavior.

## Stochastic continuation

Two mechanisms cover this, and only one is active in a given run.

**Per-step reseeding (the default with a `training_seed`).** `train.reseed_per_step` (on unless set false)
reseeds Python, NumPy and Torch (all devices) from (seed, step) at the top of every loop step. It seeds the
training sampler per epoch, and on resume skips the batches an uninterrupted run would already have consumed.
Any scheduler, GPU, cached latents and EDM2 work; see `train.reseed_per_step` in `toolkit/config_modules.py`
for the limits (loader workers, parallel batch loading, reg datasets).
- **The recorded path stands aside.** Version-3 files store `supported: false` with the reseeding reason, and no
  warning is printed.
- **Pairs still carry the rest:** the learnable SNR state and a dataset identity travel in `training_state`. Resume
  restores the SNR exactly and warns if images, captions or dataset configuration changed.
- **Mode changes are refused:** a checkpoint recorded with reseeding off refuses to resume with it on. Set
  `reseed_per_step: false` to continue such a run.

**Recorded state (reseeding off: no `training_seed`, or `reseed_per_step: false`).**
The serial CPU `SDTrainer`/`StableDiffusion` image path records Torch CPU, Python,
and NumPy RNG states, each loader's actual shuffled index order and cursor, and
dataset batch/crop layout and epoch. Resume restores the iterator state before
iteration and restores RNG only after startup and iterator creation have finished.
Already-consumed images are not fetched again. File path/size/mtime, caption,
configuration, and order identities are checked before accepting dataset state.
These checks assume stable inputs; they are not content hashes of every image.

The supported path requires one CPU process, standard non-replacement random
samplers, zero loader workers, serial per-batch loading, sampling and validation
disabled, and no EMA, auxiliary state, parameter swapping, cached stochastic
data, controls, or stateful augmentation pipelines. Custom models/hooks and
GPU execution record an explicit unsupported reason and warn. Resuming a
supported checkpoint into an unsupported mode fails before training updates.
Version-1/2 files or version-3 unsupported-mode files warn that stochastic
continuation is approximate; they retain the applicable legacy boundary policy.

Exact results also require unchanged base weights, configuration, software and
deterministic CPU kernels. Restoring global RNG cannot restore independent
generators hidden in custom models or transforms. Worker prefetch queues and
GPU RNG are deliberately not reconstructed.

## Verification boundaries

Unit regressions compare independent uninterrupted/resumed parameter,
optimizer, group, and scheduler trajectories, including Accelerate wrappers.
The acceptance script runs the real SDTrainer with a tiny diffusion model,
normal LoRA serialization, on-disk optimizer state, real metadata loading,
Accelerate preparation, and an interruption immediately after a normal save.

The supported tiny CPU diffusion path is tested with normal stochastic noise
and shuffled two-image data, once with the recorded state (reseeding off) and
once with per-step reseeding. In both modes, uninterrupted and resumed
parameters, optimizer state, scheduler state, pending gradients and RNG must
match exactly. That claim does not extend to unsupported paths described above.
`tests/test_resume_rng.py` covers per-step reseeding separately: every
micro-batch draw (images, timesteps, noise) across mid-epoch, epoch-boundary,
bucketed and non-RAWR resumes.

`tests/test_rawr_trainer_continuity.py` exercises the actual SDTrainer update
hook and BaseSDTrainProcess save with scalar CPU fixtures, periodic/end-step
save phases, cycle transitions, future updates, legacy boundaries, malformed
accumulation state, and injected write/replace failures. The tiny diffusion
acceptance path additionally covers model serialization and the host loop.

`tests/test_rawr_stochastic_pair.py` checks RNG streams, sampler epoch boundaries,
save failures at each transaction stage, pair corruption, legacy selection and
recovery updates. `tests/test_rawr_cpu_integration.py` runs the actual tiny
trainer in serial subprocesses, including an abrupt exit after model/optimizer
serialization but before manifest commit. Supply `AITK_RAWR_CPU_FIXTURES` pointing
to a local directory with `template.yaml`, `tiny-sd/` and `tiny-data/`; no download
is attempted. Without those local fixtures, these integration tests explicitly
skip. The worker entry point is `testing/rawr_cpu_probe.py`.

## Reproduce a real trainer check

Use a local tiny SD diffusion checkpoint, two 64px images with text captions,
rank 2/alpha 4, sampling disabled, and ten updates with save_every 3. SAEM's
beta1_warmup and the declared training budget must both remain ten across
interruption and resume. No full-size model is needed.

```powershell
$env:ACCELERATE_USE_CPU = 'true'
$env:CUDA_VISIBLE_DEVICES = ''
python testing/rawr_checkpoint_acceptance.py --repo . --config C:/path/tiny.yaml `
  --evidence C:/path/save-evidence --stop-after-save 3
python testing/rawr_checkpoint_acceptance.py --repo . --config C:/path/tiny.yaml `
  --evidence C:/path/resume-evidence
```

The first process stops after the production save, containing four completed
updates. The second must start at four and perform six updates. Compare its LR
and scheduler trace with steps 4-9 of an uninterrupted control. Add
`--manual-save-at 4 --stop-after-save 4` with periodic saving disabled to
exercise the same save phase as a UI end-step request.
