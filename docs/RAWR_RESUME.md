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
declared configuration, and the model metadata step. Changed LR/configuration,
step mismatches, unsupported versions, and inconsistent groups are rejected.
RAWR optimizer load/save errors propagate instead of silently continuing.
Resuming model metadata beyond step zero also requires its optimizer.pt;
weights-only adaptation should use a new output folder/pretrained LoRA input.
Other schedulers retain their existing behavior and checkpoint payloads.

Periodic saves occur before the training loop increments its zero-based label.
New RAWR metadata records the completed loop count so resume does not repeat
an update. An explicitly tracked completed count also handles UI save requests
from the end-step hook; the filename is not used to infer the resume position.

## Legacy checkpoints

Optimizer-only RAWR files retain their saved LR and cycle fields with a warning.
Their original d/gamma/cycle multiplier and exact model pairing cannot be
certified. Subsequent scheduling uses the current configuration. Missing cycle
fields and a known changed initial LR are rejected. Old zero-based labels are
not silently reinterpreted: their original save phase cannot be inferred.

## Verification boundaries

Unit regressions compare independent uninterrupted/resumed parameter,
optimizer, group, and scheduler trajectories, including Accelerate wrappers.
The acceptance script runs the real SDTrainer with a tiny diffusion model,
normal LoRA serialization, on-disk optimizer state, real metadata loading,
Accelerate preparation, and an interruption immediately after a normal save.

This does not promise bit-identical full diffusion continuation: Toolkit does
not restore dataset position or RNG state here. Distributed execution, EMA,
partial gradient accumulation, and atomic paired model/optimizer writes remain
separate concerns. Keep the optimizer.pt paired with its model checkpoint;
matching numeric labels alone cannot authenticate the pair.

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
