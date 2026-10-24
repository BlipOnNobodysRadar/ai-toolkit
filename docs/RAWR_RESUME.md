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
declared configuration, and the model metadata step. Trainer saves use version 2
to include pending gradients and the next accumulation counter; older code
rejects these files instead of silently dropping the pending work. Changed LR/configuration,
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
optimizer file readable. This is **not** an atomic model/optimizer transaction:
the model saves first, and a failure may leave a newer model with the previous
optimizer. The existing step-pairing check rejects that combination; retain the
older matching model. Power-loss durability of directory metadata is unverified.

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

## Verification boundaries

Unit regressions compare independent uninterrupted/resumed parameter,
optimizer, group, and scheduler trajectories, including Accelerate wrappers.
The acceptance script runs the real SDTrainer with a tiny diffusion model,
normal LoRA serialization, on-disk optimizer state, real metadata loading,
Accelerate preparation, and an interruption immediately after a normal save.

This does not promise bit-identical full diffusion continuation: Toolkit does
not restore dataset position or RNG state here. CPU tests with controlled
per-step randomness isolate optimizer/scheduler/accumulation continuity; normal
stochastic continuation still differs. Distributed execution, EMA, auxiliary
optimizer gradients, GPU/offload behavior, and atomic paired model/optimizer
writes remain separate concerns. Keep optimizer.pt paired with its model checkpoint;
matching numeric labels alone cannot authenticate the pair.

`tests/test_rawr_trainer_continuity.py` exercises the actual SDTrainer update
hook and BaseSDTrainProcess save with scalar CPU fixtures, periodic/end-step
save phases, cycle transitions, future updates, legacy boundaries, malformed
accumulation state, and injected write/replace failures. The tiny diffusion
acceptance path additionally covers model serialization and the host loop.

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
