# State machines: catalog, invariants, transitions, defects

**CURRENT STATUS — 2026-11-07.** Section 20 supersedes the relevant older behaviour/status. Commits: UI `bb21cae1`; core and shared CLI/UI leases `964d79a1`; data/cache/inference `cb692933`. Final combined CPU suite: **863 passed, 53 skipped**, with unchanged Python source hashes. Full UI: **101 passed**; both TypeScript projects: **zero diagnostics**. All fixes are committed on `codex/opus-state-machines-20261106`, unmerged and unpushed; main remains `f6c137b4`. The separate detached `db8dafb5` GPU comparison is unchanged and does not validate these new commits. No GPU throughput/full-model validation was run here.

**Historical status (sections 1-19; preserved verbatim; superseded where section 20 applies):**

Status: **living document**, branch `codex/opus-state-machines-20261106`, base `f6c137b4`; phase03 Python work committed
by Codex as `b83505f9`; the reviewed UI/manager/MiniCPM groups are also committed locally (section 19). Upstream (`ostris`) HEAD and main are still `beb94824`,
already merged into fork main `f6c137b4` (read-only check, run folder `upstream-final-remote.txt`).
Worker phases for the UI part: **phase04** (20:12-21:14) was stopped by the provider's session quota in the middle of
the boot-identity/boundary integration and left compile errors (Codex kept its exact source as
`phase04-interrupted-source.zip`, plus `typecheck-phase04.json`); **phase05** (21:50-22:31) finished that integration
and the remaining agreed work; **phase06** (22:32-22:36) applied two late review findings (Retry Delete in the UI, UTF-8
prefix validity) and this status. Test results below are from phase05/06 on the settled tree, never from the
interrupted phase04 source. This is the one authoritative record of the audit: machine catalog, states, invariants, transition
tables (original and current), reachable invalid transitions with evidence, impact-ranked issue groups, reviewer
agreement, implementation and test status, design decisions and coverage gaps. Logs, probes, proposals and JSON in
the run folder (`../opus-collaboration/` next to this worktree) are supporting evidence only. All accepted code groups are committed on this local branch; none of the audit fixes is merged into main. Codex completed independent source/probe acceptance, and Opus completed the focused final follow-up and regression checks. Broader GPU, platform and browser examination remains before a main merge.

---

## 1. Method, provenance and evidence rules

**Provenance.** The Qwen-derived inventory was not used as a starting point in any phase. Phase 01 (Opus) was a fresh
source audit written without the Qwen inventory or the coordinator's earlier reports. Phase 02 (Opus) *did* receive
Codex's independently reproduced findings (C01-C11, including some direct probes Codex had run before this audit) as
hypotheses; each was rechecked against source and, where possible, an original-code test or probe before agreement.
Phase 03 repairs the issues Codex's independent review found in the phase02 implementation (R01-R04, C14-C16) and
implements the proposals Codex adjudicated (P-01, P-03, P-04, P-07, C12 initialization). Earlier direct findings
therefore did inform the joint fixes; what was independent is the source recheck and the evidence.

**Evidence levels** (column *Ev.* below):
* **E1** confirmed on the original `f6c137b4`: a regression fails on the extracted original snapshot
  (`../opus-collaboration/orig-f6c137b4/`, `git archive`, test files added as listed in its `HARNESS_ADDITIONS.txt`)
  for a behavioural reason, or an original-code probe reproduces it. A test that fails on the original only because a
  new helper is absent is **not** evidence and is labelled so.
* **E2** confirmed on the phase02 draft (a review finding in new code), by Codex's probe or a test.
* **E3** source-supported (read, not reproduced).
* **E4** design/proposal.

**Status labels.** *implemented* (on the branch, tests pass) -> *accepted* (Codex independent acceptance, recorded in
the run folder's `codex-acceptance-ledger.txt`) -> *committed*. The phase01-03 Python groups are committed as
`b83505f9` (after Codex review and the 722-test suite). The remaining code groups were accepted and committed after phase06: 741 Python tests passed, 52 skipped, 85 UI tests passed, both TypeScript configurations clean. Commit mapping is in section 19. Acceptance is limited to the recorded evidence and does not imply production/GPU/platform validation.

**Coverage words.** *read* = the source was read in this audit; *partial* = the named lines were read; *searched* =
only pattern hits were seen. Nothing searched is claimed audited.

**Environment.** Opus Python commands/results are in `../opus-collaboration/opus-test-log.txt` (full outputs in `opus-test-output/`); UI commands/results are in `opus-ui-test-log.txt` and `ui-test-output/`. Independent Codex probes have their own scripts, JSON and logs in that run folder. Python suites ran through `opus_run_logged.sh` with `CUDA_VISIBLE_DEVICES=-1` (an *empty* value did not
hide the GPU on this Windows build; checked in `env-check`). No GPU validation, model download, package installation, live-service launch or production-DB mutation was performed. Process signals were confined to owned test fixtures. **Exception, found and stopped in phase03:** the pre-existing
`tests/test_resume_rng.py` launched its tiny-SD CPU training subprocesses with `CUDA_VISIBLE_DEVICES=""`; the probe's
own record from the first phase03 full suite says `cuda_initialized: true` with 0 bytes allocated/reserved, i.e. a
CUDA context was created on the GPU (earlier full suites in phase02/03 ran the same test; whether they did is not
recorded). The test now passes `-1` and asserts `cuda_initialized is False` for every probe run (15/15 false since).
No claim of GPU/runtime validation is made; GPU-only paths are labelled unverified. Phase04-06 UI tests started only
owned, hidden, CPU-only fixture processes (stdlib Python and Node), cleaned up through their own run boundaries or
identity-checked handles; each is logged in the run folder (`opus-ui-progress.txt`, `ui-test-output/`).

**Reproduce (from the repository root, Windows, CPU):**
`set CUDA_VISIBLE_DEVICES=-1` (PowerShell `$env:CUDA_VISIBLE_DEVICES='-1'`), then
`python -m pytest -q -p no:cacheprovider tests` (or a single file below); UI tests: section 16 ("UI environment").
Original-code evidence: the same test file run from an extracted `git archive f6c137b4` tree (test files copied in).

Transition tables use **From / Event / Guard / Actions / To / Failure** (what happens if the actions fail part-way).
Tables marked *current* describe the branch; *original* rows are the `f6c137b4` behaviour.

---

## 2. Machine catalog

| ID | Machine | Owner (main files) | Kind | Result | Ev. | Coverage |
|----|---------|--------------------|------|--------|-----|----------|
| M1 | Accumulation window, OOM, commit stage, whole-epoch boundary | `BaseSDTrainProcess` loop methods incl. `_fetch_batch_list`, `SDTrainer.hook_train_loop` | explicit counters + implicit pending grads | SM-01, SM-03, SM-19 (C13) | E1 | read |
| M2 | Hidden pending gradients (`p._accum_grad`) | `toolkit/optimizers/*`, `pending_grads.py` | implicit | SM-02 | E1 | read (8 optimizers) |
| M3 | Fused-backward updates | Automagic2/3/4, AdamConvRot, AutomagicEXPERIMENT | implicit | SM-01 (C11) | E1 | read |
| M3a | Automagic4 calibration -> updating | `automagic4.py` | implicit | no confirmed defect in inspected paths | test | partial |
| M4 | RAWR committed pair | `rawr_pair.py`, `save()` | explicit manifest | no defect found (read in full); stale manifest SM-04 | E1 | read |
| M5 | RAWR resume | `_load_optimizer_state`, `rawr_checkpoint` | explicit | SM-04 | E1 | read |
| M6 | Sampling borrowed state | `toolkit/sampling_state.py`, both `generate_images`, `sample()` | explicit record | SM-05, SM-06, R02 | E1/E2 | read |
| M7 | Prior-prediction borrowed flags | `SDTrainer.get_prior_prediction` | explicit | SM-05 | E1 | read |
| M8 | Latent disk cache + device preset | `dataloader_mixins.py` | implicit (file state) | SM-07, R04 | E1 | read |
| M9 | UI trainer stop/save/sample | `DiffusionTrainer`, `UITrainer`, `BaseCaptioner` | explicit flags | SM-03, SM-08 | E1 | read |
| M10 | UI job/run ownership: admission, launch phases, run boundary, reconciler, stop/force stop, legacy rows | `ui/cron/ownership/*`, `ui/cron/actions/*`, `ui/cron/worker.ts`, job routes, `toolkit/ui_procctl.py`, `toolkit/ui_run_fence.py` | explicit DB + OS boundary | SM-15 implemented (section 13) | E1 (C01) / E2 (draft probes) | read |
| M11 | Inference request | `engine.py` `GenerationJob` | explicit | SM-09, SM-12 | E1 | read |
| M12 | Inference engine / holder / LoRA stack | `engine.py`, `server.py`, `inference_lora.py` | explicit | SM-10, SM-11 | E1 | read |
| M13 | Parameter swapping | `adafactor.py`, `automagic.py` | implicit | SM-13 | E1 | partial |
| M14 | Caption publication | `BaseCaptioner.save_caption_for_file` | implicit (file) | SM-14 | E1 | partial |
| M15 | Schedule-free optimizer modes | vendored schedule-free optimizers, `training_extras/optim.py` | explicit flag | SM-16 (C12, C16, validation drift) | E1 | read (eval/train of all classes) |
| M16 | Text-embedding / CLIP-vision caches | `dataloader_mixins.py` | implicit (file) | SM-07 (P-01) | E1 | read |
| M17 | Checkpoint selection and pruning | `get_latest_save_path`, `clean_up_saves` | implicit (file times) | SM-17 (OPUS-11, P-07) | E1 | read |
| M18 | Browser request ownership, log/loss cursors | `ui/src/hooks/*`, `ui/src/utils/requestOwner.ts`, log/loss routes, generate and job pages | explicit tickets | SM-15 (C07) implemented | E1 (simulated host) | read (all poll callers) |
| M19 | Upload sessions and operation lock | `ui/src/server/uploadSession.ts`, loras upload route, `ui/cron/ownership/opLock.ts` | explicit state file | SM-15 (C08) implemented | E1 (Codex, real files) | read |
| M20 | EMA train/eval | `toolkit/ema.py`, `save()`, `sample()` | explicit flag | SM-05 (R03/C14, P-04) | E1 | read |
| M21 | Trainer run lifecycle, observer ownership, job entry | `BaseSDTrainProcess.run`, `run.py`, `BaseJob` | explicit | R01 | E2 | read |
| M22 | Inference process startup/shutdown, stop watcher | `InferenceEngine.py`, `server.serve_in_thread`, `ui_job_status.py` | explicit | SM-09 (C15) | E1/E2 | read |
| M23 | Dataset scan/epoch/item retry, sampler RNG | `toolkit/data_loader.py`, `_seed_epoch_sampler` | explicit | no confirmed defect in inspected paths (C13 is M1) | E3 | partial (phase02) |
| M24 | Network activation/multiplier/merge ownership | `network_mixins`, validation, sampling, prior | explicit flags | SM-05/06, validation activation | E3 | read |
| M25 | Component pool / memory manager | `toolkit/models/v2/pool.py`, `MemoryManager` | explicit | no confirmed defect in inspected paths; OPUS-20 hazard | E3 | read pool; partial MM |
| M26 | Async DB/log writers | `UIJobStatus`, UI loss logger | explicit | no confirmed defect in inspected paths (swallows write errors) | E3 | partial |
| M27 | Alternative trainers' update policy | slider, rescale, concept-replacer, reference-slider | explicit | SM-20 (setting ignored; now refused) | E3 + test | read `hook_train_loop` |
| M28 | SAEM step clock / host-offload write-back | `ademamix.SimplifiedAdEMAMixExM` | explicit | no confirmed defect on CPU paths; GPU unverified | E3 + existing tests | partial |
| M29 | CPUOffloadOptimizer gradient queue | vendored `low_bit_optim/cpu_offload.py` | implicit | unverified (GPU-only) | — | partial |
| M30 | Multistage boundary cycling | `SDTrainer.hook_train_loop`, Wan22 14B model, `BaseSDTrainProcess` timestep range | explicit | no confirmed defect in inspected paths | E3 | partial |
| M31 | Server request cache, model-arch loader | `ui/src/server/apiCache.ts`, `ui/src/extensions/modelArchs.ts` | explicit | no confirmed defect in inspected paths | E3 | apiCache read; modelArchs Codex-read only |
| M32 | Manager migrations | `manager/migrations.py` | explicit | no current reachable defect (registry empty) | E3 | read |
| M33 | Manager UI launcher and browser watcher | `manager/launch.py` | explicit | SM-22 (watcher ownership) | E1 | read (`launch.py`; `env.py`/`nodejs.py` Codex-read only) |
| M34 | Log terminal emulator | `ui/src/utils/terminalEmulator.ts` | explicit parser/cursor | SM-21 (scrollback cap, retention) | E1 | read |
| M35 | Job deletion (reservation, resumable, serialized) | `deleteJob`/`beginDelete` in `ui/cron/ownership/store.ts`, delete route | explicit DB + lock | SM-15 (draft regression fixed) | E2 | read |
| M36 | Queue tick: pause, admission, idle stop | `ui/cron/actions/processQueue.ts`, store | explicit DB | SM-15 (snapshot races fixed) | E2 | read |

No confirmed defect in the inspected paths (tables in sections 4, 5 and 12): RAWR two-slot commit/select/alias (M4,
read in full), RAWR scheduler capture/restore order, recorded stochastic path, `training_extras` step stamping,
dataset bounded item retry (M23), `.aitk_size.json` (rewritten when unreadable), assistant-LoRA load-time merges in
krea2/z_image/minimax_h3 (intentional, flagged unmerged), pool request generations (M25), SAEM step clock (M28),
Automagic4 calibration (M3a), multistage boundary cycling (M30), request cache and model-arch loader (M31),
migrations (M32). Partial reads stay partial: "no confirmed defect" is not "audited".

---

## 3. M1 Accumulation window, OOM, commit stage and whole-epoch boundary

### 3.1 State and invariants
`c` = `grad_accumulation_step` (starts 1); `A` = `is_grad_accumulation_step` (True: no optimizer step this
iteration); `P` = pending gradients (`p.grad`, `p._accum_grad`); `K` = `_checkpoint_completed_step`; `S` = `step_num`;
`G` = `gradient_accumulation_steps` (`-1` whole epoch); `B` = `gradient_accumulation`; `U` = `_update_commit_stage`;
`E` = `_epoch_cursor` (main batches taken from the current epoch); `L` = `_main_epoch_len` (`len(data_loader)`,
re-read after every `setup_epoch`).
* **W1** with `G >= 1`, at an iteration's top `1 <= c <= G` and `P` holds exactly the `c - 1` preceding iterations.
* **W2** with `G = -1`, `P` holds the batches of the current main-loader epoch fetched so far (plus the regularization
  batches fetched in the same iterations), never a batch of the next epoch; `0 <= E <= L`.
* **W3** one optimizer update consumes exactly one complete window (`G = -1`: one epoch, or the run's final partial
  epoch, or the remainder after a skipped OOM).
* **W4** a checkpoint records (`K`, next `c`, `P`) that resume into the uninterrupted next state (RAWR refuses `-1`).
* **W5** once `U` is set the iteration cannot be skipped or retried.
* **W6** a whole-epoch update sums its epoch's microbatches (one batch per iteration: `TrainConfig` refuses
  `gradient_accumulation > 1` with `-1`).

### 3.2 Transitions (current)
| From | Event | Guard | Actions | To | Failure |
|------|-------|-------|---------|----|---------|
| startup | before the loop | `G = -1` and no sized, nonempty main loader | `ValueError` | not started | — |
| startup | optimizer built | fused and (`G != 1` or `B > 1`); swapping and `G != 1` | `ValueError` | not started | — |
| loop top | `_setup_accumulation_iteration` | `G >= 1`, `c >= G` | `A=False`, `c=0` | update iteration | — |
| loop top | same | `G >= 1`, `c < G` | `A=True` | accumulation iteration | — |
| loop top | same | `G = -1`, last step of the run | `A=False`, `c=0` (final partial window applied) | update iteration | — |
| fetch | main batch taken (`_fetch_batch_list`) | | `E += 1` | | — |
| fetch | same | `G = -1`, `E >= L` | `A=False`, `c=0`; the batch list ends here | update iteration | — |
| fetch | main iterator exhausted | | epoch reset (seed sampler, new iterator, `setup_epoch`), `E=0`, `L` re-read | next epoch | `G = -1` and `E < L`: `RuntimeError` (loader shorter than its length) |
| any | hook returns | | `K=step+1`, `num_consecutive_oom=0` | after-hook | — |
| update iter | inside hook | | materialize hidden buffers if clipping; clip; `U='optimizer.step()'`; step; zero_grad; EMA | | OOM: abort |
| any | CUDA OOM | `U` unset and (update observable or `A`) | `num_consecutive_oom += 1` (>3 aborts); drop the window incl. hidden buffers; `c=0`; `K=step+1` | after-hook (skipped) | — |
| any | CUDA OOM | `U` set, or update iteration with an unobservable optimizer | `RuntimeError` "cannot be skipped or retried" | aborted before any save | — |
| after-hook | periodic save / sample / UI save/sample | | window kept (all schedulers) | | RAWR refuses an invalid position |
| after-hook | end of iteration | | every fetched batch DTO cleaned; `S=step+1`; `c += 1`; `end_step_hook()` | loop top | — |
| resume (non-RAWR) | loader positioned | `G != 1` | cursor from `step * B`; limit printed (no unfinished window in the checkpoint) | | mid-window save: the first update after resume is smaller |

### 3.3 Original invalid transitions (E1)
| # | Trace on `f6c137b4` | Evidence |
|---|---------------------|----------|
| 1 | `G=4`, OOM at step 1: next update averages 2 microbatches at 1/4 scale | AST-extracted original loop probe P1 |
| 2 | OOM on a save step commits `next_accumulation_step=0`; restore rejects it; manifest points at it (OPUS-01) | P2 |
| 3 | OOM inside `optimizer.step()` after parameters changed is skipped like a forward OOM | P3 |
| 4 | Fused Automagic3 with `G=2` accepted; parameters move on accumulation iterations (C11) | P4 |
| 5 | Periodic non-RAWR save mid-window clears the pending microbatch (OPUS-04) | P6 |
| 6 | UI "Sample next step" clears the window (OPUS-03) | test fails on original |
| 7 | `G=-1`, epoch of 4: first update holds 5 batches (`e0b0..e0b3, e1b0`), later windows shifted by one batch, a 4-step run never updates (C13) | batch-fetch statements AST-extracted from the original loop over a real DataLoader of 4 (`opus_probe_epoch_window.py` in the run folder) |

### 3.4 C13 decision and implementation (Codex decision rev. 2; committed `b83505f9`)
Agreed contract: each `-1` update contains one main-loader epoch plus the regularization batches fetched in its
iterations, never the next epoch's first batch; the run's final nonempty partial window is applied; a pre-update OOM
discards the unfinished window and the rest of the epoch becomes a shorter window; fixed-size mode unchanged; RAWR
`-1` resume stays refused. Implemented with a sized-loader cursor (no lookahead, so augmentation RNG and
`setup_epoch` timing are unchanged); the epoch length is re-read after each `setup_epoch` (bucket rebuilds can change
it). `TrainConfig` refuses `gradient_accumulation > 1` with any `gradient_accumulation_steps != 1` (`-1` included;
`toolkit/config_modules.py`), so whole-epoch iterations always carry one batch; the `B > 1` normalization and resume
branches I first added were removed as unreachable (only a harmless list break remains), and Codex's N4/B3 helper
probe is a synthetic draft-hardening counterexample, not a product issue. **Resume limit (unchanged format):** a
non-RAWR checkpoint does not store an unfinished window; a resume is exact only from a save made right after an update
(the tiny-SD `whole_epoch` case), a mid-window/mid-epoch save loses the pending window, and a resume with
accumulation now prints that. Tests: `tests/test_epoch_accumulation.py` (real `_fetch_batch_list`/hook/OOM path over
real DataLoaders: N=1 and N=4, exact epoch end, final partial window, three epochs, interleaved regularization, OOM
remainder, unsized/empty/short loaders, fixed-size unchanged, summed epoch and final-window gradients, every DTO
cleaned, `TrainConfig` refusal of `B > 1`) and `tests/test_resume_rng.py` case `whole_epoch` (real tiny-SD CPU
trainings: straight vs resumed draws and final weights identical). My own errors caught while doing this: the loop's
later `batch` cleanup lost its local when the fetch moved into a method (the tiny-SD probe's NameError), and fixture
hooks; both fixed.

---

## 4. M2 hidden pending gradients and M3/M3a fused and calibrating optimizers

Adafactor, Automagic, Adam8bit, Prodigy8bit, Automagic3/4, AdamConvRot and AutomagicEXPERIMENT (unfused) move each
low-precision gradient into `p._accum_grad` and delete `p.grad`; `step()` moves it back.

| From | Event | Guard | Actions (current) | To | Original |
|------|-------|-------|-------------------|----|----------|
| none | backward | low-precision param | `_accum_grad = grad`, `grad = None` | hidden | same |
| hidden | `step()` | | buffer into `grad`, update | none | same |
| hidden | `zero_grad()` | | drops the buffer | none | buffer survived (E1, 8/8 optimizers) |
| hidden | OOM window reset | | `zero_grad` + `pending_grads.discard` | none | survived (P5) |
| hidden | RAWR `training_state()` | | joins offload copies; captures buffer + location | pair holds window | captured None (E1) |
| hidden | clipping | update, `max_grad_norm > 0` | materialize before `clip_grad_norm_` | grad | clipping saw nothing (P7) |

**M3** fused optimizers update inside backward; the generic observer marks the commit at the first gradient hook
(before the optimizer's post-accumulate hook). **M3a Automagic4**: `uncalibrated -> calibrated -> updating`. The first
step only measures second moments and the sensitivity reference (`_group_ref`, per-parameter `sens_ref`); later steps
move weights. `load_state_dict` rebuilds the group reference from saved `sens_ref`. Tests
(`tests/test_automagic4_calibration.py`, 3): first step does not move weights; a resume saved after calibration
continues the uninterrupted run exactly; negative control without `sens_ref` recalibrates and diverges. The phase02
low-precision continuity test dropped Automagic4 because a first-step movement expectation was invalid for it; that
was a fixture error, not a protocol bug. The OPUS-09 GPU transfer race is **not verified** (no GPU).

---

## 5. M4/M5 RAWR pair and resume

The two-slot pair (`begin` -> inactive slot written -> extras -> `commit`: fsync, hashes, atomic optimizer, manifest
replace -> alias publication with warnings) was read in full; no defect was found in it. Changes (current):

| From | Event | Guard | Actions | To | Failure |
|------|-------|-------|---------|----|---------|
| startup | `run()` | non-RAWR scheduler, manifest present | `ValueError`, manifest untouched | stopped | — |
| save | `rawr_pair.begin` | RAWR off, manifest present | same refusal before any write | | — |
| resume | `_load_optimizer_state` | RAWR, `step_num > 0`, optimizer not loaded | `RuntimeError` | stopped | — |

Original (E1): `opus-original-resume-probe.json` (rank change: paired optimizer present, not loaded, RAWR restarts
silently); non-RAWR save into a committed-pair folder accepted.

---

## 6. M6/M7/M20/M24 borrowed model state: sampling, EMA, prior prediction, validation

### 6.1 Sampling record (M6, current, `toolkit/sampling_state.py`)
Entry values recorded **before** any change: CPU/CUDA RNG; auxiliary LoRA activation and its single (device, dtype)
over parameters *and* buffers; trained network mode, multiplier, `is_active`, `is_merged_in`; the device snapshot.

| From | Event | Guard | Actions | To | Failure |
|------|-------|-------|---------|----|---------|
| training | `borrow` | auxiliary LoRA spread over several devices/dtypes, or a moved role (inverted assistant, inference) whose dtype != holder dtype | `ValueError`, nothing changed | training | — |
| training | `borrow` | | RNG captured; aux activation set (plain assistant off; inverted/inference on, moved to device); network `eval()`; merge only if allowed, not already merged, one multiplier; device preset via `enter_device_preset` | sampling | any step fails: `return_` undoes the steps done, error kept |
| sampling | generation body fails | | `borrowed()` calls `return_(primary=e)` | training | give-back failures are notes on `e` |
| sampling | `return_` | | each step attempted independently: RNG; device snapshot (the pre-move one); network mode/multiplier/`is_active`; `merge_out` only if this scope merged; legacy unet back to its entry device only if this scope moved it; aux activation and exact placement | training | first failure raised after all steps (no primary) |
| sampling | `merge_in` failed part-way | | no `merge_out` (it would subtract deltas never added) | | error notes "partly merged, must be reloaded" |

Original (E1/E2): device state, EMA, adapter flag and step hooks were not given back on failure (14 tests, phase02);
training previews merged the LoRA, drifting bf16 base weights (OPUS-08: 42/4096 elements after 20 events). The phase02
helper and the original teardown both hard-coded auxiliary activation/placement (Codex `sampling_entry_probe.json`:
inactive assistant returned active; active float64 inference LoRA returned inactive float32; a premerged network was
merged out) and cast a float32 auxiliary LoRA through bf16 and back (Codex `auxiliary_lora_roundtrip_probe.json`,
512/512 values changed). Ordinary initialisations matched the hard-coding (non-inverted assistant active,
inverted/inference inactive, all loaders cast to the holder dtype), so no ordinary-config failure on the original is
claimed for R02. The original teardown also ran an unconditional `unet.to(device_torch, dtype=torch_dtype)`; it is
gone (device restoration is `restore_device_state`'s job; a deliberately mixed-dtype unet is no longer recast; E3).
**Load-time parking** keeps the original memory behaviour: the original teardown moved inverted-assistant and
inference LoRAs to the CPU after the first sample; their loaders (`stable_diffusion_model.py`, `krea2.py`,
`z_image.py`) now park them on the CPU at load, and exact restoration keeps them there between samples. LoRA modules
keep their base module in a list, so `unet.to()` never moves them back.
Tests: `tests/test_sampling_state.py` (R02 group: every parameter/buffer value, dtype and device compared, nontrivial
values; mixed placement and dtype mismatch refused before mutation; partial preset failure put back once from the
pre-move snapshot; failing device restore keeps the generation error and still restores flags and RNG; partial merge;
logging failure in `settle`). Holder device-preset methods are recording stubs (one CPU device: module moves are not
exercised).

### 6.2 EMA (M20)
States: `training` (raw weights live, nothing stored), `evaluating` (EMA weights live, raw copies stored), and the
failure state `mixed / restore pending` (some parameters EMA, raw copies stored, flag says evaluating).

| From | Event | Actions (current) | To | Failure (current) | Original |
|------|-------|-------------------|----|-------------------|----------|
| training | `eval()` | store raw; copy EMA in | evaluating | copy fails: raw put back, stays `training`, error raised; rollback also fails: `evaluating` claimed with raw copies kept so `train()` retries (mixed weights, restore pending), note on the error | second copy failure left `[EMA, raw]` with `training` claimed; `train()` then did nothing (Codex `ema_entry_probe.json`, E1) |
| evaluating | `train()` | restore raw, drop copies | training | restore fails: stays evaluating with copies, retry possible | same (retry-safe on both) |
| any | `average_parameters()` | copy inside the try | | partial copy restored | partial copy left (E1) |
| training | `save()` (P-04) | EMA in only for the save body; entry mode given back; network multiplier restored in a `finally`; control-net adapter placement given back | training | save error kept, give-back failures noted | EMA weights and multiplier 1.0 left after a failed save (E1); ended in `train()` even if entered evaluating |
| training | `sample()` | EMA/adapter flag given back through `settle` | training | generation error kept | (fixed in phase02, now primary-preserving) |

Tests: R03/P-04 group in `tests/test_sampling_state.py`; original snapshot: partial eval x2, failed save x2,
`average_parameters` fail behaviourally (`r03-p04-original`, `r03-average-original`); `train()` retry passes on both.

### 6.3 Prior prediction (M7) and validation activation (M24)
Prior prediction restores network/adapter `is_active` and unet mode in a `finally` (phase02, E1). Validation
(`toolkit_validation.validate`) restored multiplier and unet mode but its `with network` forced `is_active` True then
False regardless of entry; it now gives back the entry value (E3 on the original; between training steps the entry is
normally False, so the ordinary path was unaffected). Test: `test_validation_gives_back_the_networks_entry_activation`.

---

## 7. M8/M16 caches and borrowed device presets

| From | Event | Guard | Actions (current) | To | Failure |
|------|-------|-------|-------------------|----|---------|
| file exists | caching pass | header opens (`_cache_file_is_usable`; latents: memory mode loads fully) | trust | cached | — |
| file exists | caching pass | unreadable/truncated | rebuild from source through the normal path | cached | — |
| encoding | publish | | `_publish_cache_file`: temp in the cache folder, `os.replace` | complete file or none | temp removed |
| latent pass | first item needing encode | | `enter_device_preset(sd, 'cache_latents')` (own pre-move snapshot) | borrowed | preset fails part-way: put back once from the snapshot, error raised |
| latent pass | end, error or interrupt | | `settle([executor.shutdown, pbar.close, leave_device_preset])` | given back | each step attempted; pass error kept |
| CLIP pass | start | | `enter_device_preset(sd, 'cache_clip')`; body `_cache_clip_vision_borrowed` | borrowed | body error: preset given back, error kept |
| TE pass | first encode | | `set_device_state_preset('cache_text_encoder')`, **not** given back | TE placed | deliberate: the trainer parks the TE once after all datasets |

Original (E1): latent disk cache trusted a truncated file and memory mode dropped the item forever; a killed write
left a partial file at the name (4/4); the latent preset was never restored after an encode/publish error (Codex
`cache_cleanup_probe.json`; 4 R04 tests fail on the original); text-embedding and CLIP caches had the same
truncation/publication defect and the CLIP preset was not restored on error (3 tests,
`tests/test_embedding_cache_recovery.py`). Not exercised: CLIP truncated-file rebuild with a real vision encoder.
Decided limits (P-05/P-06): crash-abandoned hidden temp files are harmless and not auto-deleted; a cache corrupted
externally after validation is a runtime I/O failure, not an internally reachable publication defect.

---

## 8. M9 UI trainer stop and M14 captions
* SM-08: `maybe_stop` sets `is_stopping` before the blocking status write (DiffusionTrainer, UITrainer,
  BaseCaptioner); original 4/4 fail.
* SM-14: caption publication is temp + fsync + `os.replace`; the original deleted the old caption first (a failed
  write left `''`, 1/2 fail).

---

### 8.1 M13 parameter swapping (Adafactor/Automagic `do_paramiter_swapping`; current, SM-13)
State: the trained subset (`requires_grad` per parameter), `paramiter_swapping_factor`. Invariant: every step trains a
nonempty subset; no gradient survives a reselection; the factor is in (0, 1].
| From | Event | Guard | Actions | To | Failure |
|------|-------|-------|---------|----|---------|
| setup | `enable_paramiter_swapping(f)` | `0 < f <= 1` | store factor | enabled | `ValueError` otherwise |
| startup | accumulation check | `gradient_accumulation_steps != 1` | `ValueError` (reselection clears gradients every step) | not started | — |
| step top | `swap_paramiters()` | | all off, grads cleared, shuffle (step-seeded RNG), switch on until the next would exceed `f` of the total; at least one | subset | — |
Original (E1; `tests/test_paramiter_swapping.py`: 14 behavioural failures, 3 more only for an absent API, of 19): a
factor below the smallest tensor selected nothing, and accumulation silently lost windows at every reselection.

## 9. M11/M12/M22 inference engine

### 9.1 Requests and engine (current)
| Machine | From | Event | Guard | Actions | To | Failure |
|---------|------|-------|-------|---------|----|---------|
| request | — | `submit` | arch a non-empty string, not closed | registered + enqueued under the engine lock | queued | `EngineClosed` (503), `ValueError` (400) |
| request | queued | `cancel` | `cancel_if_queued` under the job lock | error frame, `finish` | cancelled | — |
| request | queued | dequeue | `claim()` under the job lock | | running | lost claim: skipped |
| request | running | any exception in setup/generation | | error frame, outputs removed, `finish` | error | worker continues |
| request | any | `finish` | first call only | one end frame | finished | later emits dropped |
| engine | idle | `unload` | `_exec_lock` free | drop holder, clear pool | idle | `EngineBusy` 409 when held |
| engine | serving | `run_forever` exits (normal **or exception**) | | `_close_and_drain` (closed, queued work cancelled) | closed | primary error kept (C15) |
| LoRA stack | loaded | apply fails part-way | merge / hook | evict touched pool entries + drop holder / detach hooks | clean | — |

### 9.2 Process startup/shutdown and stop watcher (M22, current)
| From | Event | Actions | To | Failure |
|------|-------|---------|----|---------|
| starting | `run()` | `_serve()`: engine, server, endpoint file, signal handlers (previous recorded), watcher, preload, loop | serving | any exception or KeyboardInterrupt (e.g. stop during preload): `shutdown(primary)`, error re-raised |
| any | `shutdown()` | independent steps: stop watcher; restore signal handlers; close/drain admission (also when `run_forever` never ran); server stop/join; endpoint removal; unload; status | stopped | each step attempted; failures noted on the primary |
| watcher | flag read returns stop | dispatch under the callback lock, rechecking "stopped" | callback once | after `stop_stop_watcher` returns no callback starts; an in-flight one finishes first |
| startup | `serve_in_thread` timeout/interrupt before handoff | stop its own thread, close its socket | nothing owned | error kept |

Original (E1): 12/13 request/engine tests fail; preload interruption skipped `shutdown()` and a dead loop kept
admitting (Codex `inference_startup_probe.json`; tests fail on the original); requests admitted before a failed
startup were never settled; `serve_in_thread` left its thread/socket on a failed start. Watcher race (E2: Codex
`watcher_stop_probe.json` on the draft). The idle-status failure used in tests is injected: `UIJobStatus` swallows
ordinary DB write errors, so no observed production callback failure is claimed. The 200x cancel/claim stress test
also passes on the original; C04's evidence is the deterministic interleaving test.

---

## 10. M15 schedule-free optimizer modes

Layouts (source, every vendored class read): **attribute** `self.train_mode` (ScheduleFreeWrapper, ADOPT family,
FADOPT family, `_ADOPTAOScheduleFreeBase`, AMUSE; not serialized) and **per-group** `group['train_mode']`
(RAdamScheduleFree, AdamWScheduleFreePlus, NorMuonScheduleFree, SODA; serialized with the groups). FADOPT is a hybrid
(attribute used, an unused `'train_mode': True` group default), so the attribute wins in `training_mode()`.

| From | Event | Actions (current) | To | Original |
|------|-------|-------------------|----|----------|
| constructed (eval; SODA train) | `enter_training_mode` after `_load_optimizer_state` (and in MiniCPM after create/resume) | attribute: set flag only (toolkit checkpoints hold training weights y); group: `train()` (exact via serialized mode) | train | never entered: first `step()` raises "not in train mode" unless validation ran first (Codex `schedulefree-probe.json`, E1) |
| train | `eval()` / `train()` | every group converted, flag set after the loop | eval / train | attribute layout set the flag inside the first group: later groups never converted (C16, E1: `tests/test_schedulefree_groups.py` fails on the original; Codex `schedulefree-groups-probe.json`) |
| train | `evaluation_optimizer` (validation, MiniCPM eval) | exact CPU copies of every optimized parameter; optimizer's own `eval()`; RNG draws of `eval()` given back (CPU + CUDA devices holding the params only); body; copies written back in place, train flag set | train | `eval()`...`train()` on live weights: bf16 changed 79/256 values per pass, 214/256 after 20 (`opus-schedulefree-roundtrip-probe.json`); exit forced train even for a nested scope (Codex probe) |
| train | partial `eval()` or body failure | all copies restored, flag set | train | — |
| train | a copy cannot be restored | flag set **not** training; error "could not restore N of M... stop and reload" | out of train mode (mixed) | — |
| eval | nested `evaluation_optimizer` | nothing | eval | inner exit switched to train |
| resumed (attribute layout) | MiniCPM loop `opt.train()` | (now preceded by `enter_training_mode`, so a no-op) | train | moved the checkpoint's y weights as if they were x (E3; same transition as the negative-control test) |
| train (MiniCPM, after the last step) | finalization | final checkpoint in TRAINING coordinates first; then one `opt.eval()` kept for the final loss, validation and the exported adapter (no `train()`/`eval()` round trip) | eval | `eval()` -> final loss -> `train()` -> checkpoint -> `eval()` -> export: the checkpoint changed 79/256 bf16 values (Codex `minicpm_finalization_probe.json`, actual AST statements + real CPU ADOPTScheduleFree, E1); a resume with a larger `--steps` passes the guard (`--steps` is not in the options hash; `start_step >= a.steps` refuses only the same total) |

**Decision (Codex, option A):** checkpoints, exports and previews keep training coordinates (y); no eval-coordinate
save format and no save-frequency-dependent round trip. Exporting the schedule-free evaluation point is an
**unresolved design limitation**, not a fixed bug; the resume tests prove a small optimizer continuation, not exact
full-training resume (EMA, non-RAWR pairing, GPU/offload and trainer clocks have their own contracts). Tests:
`tests/test_schedulefree_modes.py` (real optimizers on CPU: fresh start, resume continuity + negative control,
evaluation trajectory identical with and without 3 scopes in fp32 and bf16 incl. RNG, evaluation point equals the
optimizer's own, identity/storage/pending grad, body/partial-eval/combined failures, CUDA RNG untouched for CPU
params) and `tests/test_schedulefree_groups.py`. Not runnable here: RAdamScheduleFree `step()` (CUDA-only), AMUSE
(torch.compile), EMAMix/Nesterov/MARS/AO variants (same corrected loop, not executed). MiniCPM's own path needs a CUDA
model: covered through the helper, and its finalization order by `tests/test_minicpm_finalization.py` (the actual
statements after `main()`'s loop, extracted by AST, run with a real CPU bf16 optimizer: checkpoint 0/256 values changed
against the trained values, export = evaluation weights, which are also the weights of `final_train_loss`; no MiniCPM
model or GPU run).

---

## 11. M17 checkpoint selection and pruning (OPUS-11, P-07)

`save_write_rank(path)` = (last write time, step); a Diffusers folder is ranked by its `aitk_meta.yaml`, which every
save path writes last; equal times rank by step, the unsuffixed final save last. Used by `get_latest_save_path`
(after the RAWR manifest branch, which still selects committed pairs) and every `clean_up_saves` sort.

| Event | Original (creation time) | Current |
|-------|--------------------------|---------|
| final `.pt` rewritten after a periodic one (NTFS keeps creation time) | resumes from the older periodic save | resumes from the final save |
| Diffusers folder rewritten (nested files overwritten) | same rewind | ranked by `aitk_meta.yaml` mtime |
| safetensors rewrite | correct (writer resets birth time; Codex) | correct |
| restart rewrote an old step's file in place | pruned the newest write | prunes the oldest write |
| deliberate reset: newer lower-step save | newest by creation | newest by write (step only breaks ties) |

Tests `tests/test_save_selection.py` (10, real writers, `os.utime`, no sleeps): `.pt` and Diffusers fail on the
original for resume and pruning (4); safetensors and tie cases pass on both (no tie defect claimed for the original).

---

## 12. Remaining families: states, invariants and transitions

### 12.1 M21 trainer run lifecycle, observer ownership, job entry (current; R01 E2)
States: `constructed -> running (_run_training) -> loop (observers installed) -> after loop (observers removed) ->
done | failed`. Invariant: commit observers exist only while this trainer's `run()` is on the stack; none survives
on an optimizer after `run()` returns or raises. Job entry (`run.py`): `config -> job -> run -> cleanup`.

| From | Event | Actions | To | Failure |
|------|-------|---------|----|---------|
| constructed | `run()` | `_run_training()` inside `try/finally` | running | any exception/KeyboardInterrupt: `finally` removes observers, error propagates |
| running | `_check_trainer_accumulation_capability` | (M27) | running | `ValueError` before any model loads |
| running | loop start | `_install_update_commit_observers` (removes old handles first, records each as made) | loop | partial install: recorded handles removed by `finally` |
| loop | loop end | remove observers, final validation/save/sample, `done_hook`, `release_accelerator_state` | done | save/sample failure: observers already removed; error propagates |
| run.py | job raises | `process[0].on_error(e)`; with `-r`: job dropped, `accelerator.free_memory()`, next config | next job | without `-r`: re-raised |

Original: observers did not exist; phase02 removed them only after a normal loop end (R01). Tests: 8 in
`tests/test_accumulation_window.py`; `tests/test_accelerator_teardown.py` checks the teardown order in
`_run_training`.

### 12.2 M27 alternative trainers' update policy (current; SM-20, E3 + tests)
States: per iteration `zero_grad -> forward/backward -> optimizer.step()` (one update per iteration). Invariant
(current): a trainer is never run with an accumulation setting its `hook_train_loop` does not implement.

| From | Event | Guard | Actions | To | Failure |
|------|-------|-------|---------|----|---------|
| startup | capability check | class `supports_gradient_accumulation_steps` False and `G != 1` | `ValueError` with guidance | not started | — |
| startup | capability check | `TrainSliderProcess` (uses `batch[0]`) and `B > 1` | `ValueError` | not started | — |
| iteration | `hook_train_loop` | | zero, step every call; step pre-hook observes the update (OOM after it aborts) | next | — |

Affected (read): ConceptReplacer, ImageReferenceSliderTrainerProcess, UltimateSliderTrainerProcess,
TrainSDRescaleProcess, TrainSliderProcess, TrainSliderProcessOld. Ordinary reachable boundary: `type: slider`
(`jobs/TrainJob.py` `process_dict`, shipped `config/examples/train_slider.example.yml`). Original: `G != 1` silently
trained one update per iteration (effective batch size different from the configuration). Other overrides keep
`B > 1` (not shown to mishandle lists; they were not read beyond `hook_train_loop`). Tests:
`tests/test_trainer_capabilities.py` (refused before model load; default accepted; SDTrainer keeps every setting).

### 12.3 M23 dataset scan / epoch / item retry (phase02 read of `toolkit/data_loader.py` 387-897; not re-read)
| From | Event | Actions | To | Failure |
|------|-------|---------|----|---------|
| config | dataset init | scan files, sizes (`.aitk_size.json`, rebuilt when unreadable) | scanned | unreadable size cache rebuilt |
| scanned | bucketing | buckets + batch indices | bucketed | — |
| bucketed | caching (M8/M16) | latents/embeddings | ready | see section 7 |
| epoch k | iterator exhausted | `_seed_epoch_sampler(dl, 1, epoch)`, new iterator, `setup_epoch` | epoch k+1 | — |
| item | load fails | replacement from the same bucket, bounded retries | item | retries exhausted: error |
Invariant: per-epoch sampler order derives from (seed, epoch); workers receive a model-free dataset. No confirmed
defect in inspected paths.

### 12.4 M25 component pool (read) and memory manager (partial)
States per entry: `absent -> resident(touched_gen) -> freed`; pool `generation` bumps per request. Invariant: an
entry is freed only under the pool lock and only when not touched by the current request (`release_unused`) or
explicitly evicted.
| From | Event | Actions | To | Failure |
|------|-------|---------|----|---------|
| — | `begin_request` | `generation += 1` | | — |
| absent | `put` | entry with this generation | resident | — |
| resident | `get` / `touch_module` | `touched_gen = generation` | resident | — |
| resident | `release_unused` | free entries not touched this generation | freed | `MemoryManager.free` fails: moved to `meta` |
| resident | `evict_where` / `clear` | free matching / all | freed | same |
Used only by the inference engine (serial generations; `unload` under `_exec_lock`). `MemoryManager.free/detach`
dropping per-device state is a hazard with no reachable training trigger (OPUS-20, E3, partial read).

### 12.5 M26 async writers
| Machine | From | Event | Actions | To | Failure |
|---------|------|-------|---------|----|---------|
| `UIJobStatus` | any | `_write(sql)` | retried connect/execute under a lock | written | ordinary DB errors swallowed (logged) |
| `UIJobStatus` watcher | watching | flag read true | `on_stop` under the callback lock, rechecking stop | done | read errors ignored, next poll |
| `UIJobStatus` watcher | watching | `stop_stop_watcher` | done flag under the callback lock | stopped | an in-flight callback finishes first |
| UI loss logger | buffering | step | append; periodic flush | flushed | (phase02 read; no defect found) |
Invariant: no status write raises into the engine today (the UI fencing in section 13 will change this, by design).

### 12.6 M28 SAEM step clock (partial read + `tests/test_saem_optimizer.py`)
| From | Event | Actions | To | Failure |
|------|-------|---------|----|---------|
| group step k | `step()` | `group['step'] = k+1` first, then per-parameter update, moments in storage dtype/device | k+1 | exception mid-step: clock already advanced, the step pre-hook marked the commit, the iteration aborts |
| saved | `load_state_dict` | moments restored from the source tensors in storage dtype/device | resumed | — |
| host-offloaded param | update | async D2H copy, group-end synchronize | visible | GPU path unverified |

### 12.7 M29 CPUOffloadOptimizer (partial; GPU-only, unverified)
`device grad -> event-queued copy -> host grad (copied, not accumulated) -> host step()`. Reachable through the generic
FQN resolver (its `optimizer_class` defaults to AdamW; phase02's "never constructible" wording was wrong); no shipped
config/index entry; construction needs CUDA/XPU. Multi-backward accumulation and RAWR pending-state compatibility are
unverified. No fix.

### 12.8 M30 multistage boundary cycling (read: `SDTrainer.hook_train_loop`, Wan22 14B constructor)
State: `current_boundary_index`, `steps_this_boundary`; sets `multistage_boundaries`, `trainable_multistage_boundaries`.
Invariant: training only runs microbatches on a trainable boundary; the trainable set is nonempty.
| From | Event | Guard | Actions | To | Failure |
|------|-------|-------|---------|----|---------|
| model init | Wan22 14B | neither high nor low noise trainable | `ValueError` | — | — |
| model init | base models | | boundaries `[0.0]`, trainable `[0]`, `is_multistage=False` | single stage | — |
| microbatch | before training it | `steps_this_boundary >= switch_boundary_every` or index untrainable | cycle the index (wrap) until trainable, reset count | boundary i | — (empty set rejected at init) |
| microbatch | trained | | `steps_this_boundary += 1` | | — |
Timestep range per boundary is derived in `BaseSDTrainProcess` (`[1] + multistage_boundaries`); the discrete
timestep-choice branches were not inspected. GPU execution untested.

### 12.9 M31 server request cache and model-arch loader
`apiCache.cached` (read): `absent -> pending(promise, timestamp at creation) -> fresh hit (same promise) -> stale`.
| From | Event | Guard | Actions | To |
|------|-------|-------|---------|----|
| absent / stale | `cached(key)` | | call fetcher, store promise + timestamp | pending |
| pending/fresh | `cached(key)` | age < TTL | return the stored promise | same |
| pending | `cached(key)` | age >= TTL (even if still pending) | replace with a new fetch (overlap by design) | pending (new) |
| pending | rejection | stored promise is still this one | delete the entry | absent |
| pending | rejection | replaced meanwhile | nothing (cannot delete the newer owner) | |
| any | `invalidateCache` | | delete key / all param variants | absent |
Not universal in-flight de-duplication: a fetch slower than its TTL can overlap the next. `modelArchs.loadModelArchs`
(Codex source read; not read by Opus): one in-flight load -> loaded/errors, partial module failures collected, a
failure keeps the previous archs, `finally` clears in-flight. No confirmed defect in inspected behaviour.

### 12.10 M32 manager migrations (read) and M33 UI launcher (read `launch.py`; SM-22)
`run_pending`: for each migration not in the applied set: run, then record its id (not in dry run). Invariant: an id
is recorded only after its migration returned. The registry is empty (`MIGRATIONS = []`), so a crash between run and
record is a future authoring contract, not a current reachable defect.

M33 `launch_ui` (current): preflight (venv, node, npm, deps) -> `stop_event` + `open_lock` -> inside one
`try/finally`: start the browser watcher thread, `Popen(npm run db_build_start)`, wait -> `finally: with open_lock:
stop_event.set()`. Invariant: once `launch_ui` has returned or raised, no browser is opened by its watcher.
| From | Event | Actions (current) | Original |
|------|-------|-------------------|----------|
| preflight ok | watcher start, Popen | both inside the `finally` scope | Popen before `try`: an injected Popen failure left the watcher running unsignalled (Codex `manager_startup_probe.json`: `watcher_stop_signalled: false`, E1) |
| watcher polling | readiness request returns | open the browser only under `open_lock` after re-checking `stop_event` | a readiness request blocked when the UI stopped still opened the browser (`manager-blocked-readiness-original.json`, original function, stub browser, E1) |
| watcher polling | request fails | `stop_event.wait(2)` (wakes at once on stop) | `time.sleep(2)` |
| running | Ctrl+C | terminate npm, kill after 10 s | same; whether npm's descendants (the UI servers) exit with it is NOT verified |
Tests: `tests/test_manager_launch.py` (4: injected Popen failure signals the watcher; blocked readiness released after
stop opens nothing; ready UI opens once; stop wakes the retry wait); no browser, service or install. `manager/env.py`
installer/sync and `nodejs.ensure_ui_deps` were read by Codex only (no new issue asserted); installer end-to-end,
simultaneous managers and platform provisioning remain runtime-unverified.

### 12.11 M34 log terminal emulator (read; SM-21, E1, implemented)
State: `lines` (<= 5000), cursor `row/col`, `pending` (an escape sequence split across chunks, <= 4096 chars).
Invariant: `len(lines) <= MAX_LINES` and `0 <= row < len(lines)` after every write.
| From | Event | Actions (current) | To | Original |
|------|-------|-------------------|----|----------|
| text | printable run | overwrite at cursor (pad with spaces) | text | same |
| text | `\n` | next row, prune the oldest beyond 5000 | text | same |
| text | `ESC[nB` / `ESC[ne` / `ESC[nE` | bounded move: rows scrolling off the top are dropped or never created | text | grew without bound (6000 -> 6001 lines, again -> 12001; Codex probe; 4 tests fail on the original) |
| text | `ESC[nB` with `n >= 2*5000` (incl. `Infinity`/NaN from an absurd parameter) | every existing row scrolls off whatever the cursor row: a blank 5000-row grid, cursor on the last row, decided before any arithmetic on huge numbers; smaller counts use the exact arithmetic (only rows that would scroll off are dropped, e.g. 5000 rows, cursor at row 0, down 5000 drops only the first) | text | (drafts: floating cancellation left `lines=[]`, Codex `terminal_overflow_probe`; the `n >= 5000` shortcut wiped retained rows, Codex `terminal_retention_probe`; parent repair reviewed) |
| text | `ESC[nA`, `ESC M`, `ESC[nD` | clamp at 0 | text | same |
| text | incomplete escape at chunk end | keep in `pending` (cap 4096) | pending | same |
| pending | next chunk | prepend and parse | text | same |
| text | `ESC[2J`/`ESC[3J` | reset | empty | same |
Column jumps are not capped (decided). Test: `ui/tests/terminalEmulator.test.cjs` (node:test on the transpiled source,
8 cases incl. 1e20 and 310-digit counts followed by text and the retention case).

---

## 13. M10/M18/M19/M35/M36 UI ownership, requests, uploads, deletion, queue

**Status: accepted within the stated test/platform scope and committed locally (phase04 interrupted, completed in phase05, two late fixes in phase06). Four review groups SM-15a-d are listed in section 14; section 19 maps them to commits.** Agreed basis: `ui-ownership-proposal.txt` rev. 2 with `codex-ui-implementation-decision.txt` (all seven
corrections) and `codex-ui-boundary-decision.txt` rev. 2 (Q1-Q4; run folder, supporting only). Everything below is
also stated in the source comments of the named files. Windows is the tested platform; Linux is source-only here
(no Linux host or container was available; Codex found the Docker daemon absent and started nothing); macOS and other
platforms refuse to launch.

### 13.1 Ownership model (M10)

**Run generation and claims.** Admission gives a queued job a fresh `run_id` (Prisma `Job.run_id`, plus nullable
launch/identity columns; additive schema) and, in the same SQLite transaction, one `RunClaim` row per owned
resource: `gpu:<n>` for each normalized index (`'1,0'` = `'0,1'`, duplicates/whitespace ignored), `gpu:mps`, `gpu:*`
for an opaque token (UUID/MIG name: exclusive against every GPU claim, no aliasing claimed) and
`folder:<canonical folder>`. A CPU-only job is `gpu_ids: '-1'` and owns only its folder; an empty `gpu_ids` is refused
at create/edit/start/admission ("No GPU is selected..."); `-1` mixed with GPUs is refused.

**Canonical folder** (`ui/cron/ownership/resources.ts`): `path.resolve(root, name)`; the nearest existing ancestor
resolved with `fs.realpathSync.native` (junctions, symlinks, short names) plus the not-yet-existing tail; on Windows
case-folded with trailing dots/spaces of each component dropped (NTFS semantics); on macOS NFC + case-folded. A
realpath failure other than a missing component is a `FolderIdentityError` (admission waits with the reason,
routes return 500), never a lexical substitute. Two folders **conflict when equal or when one lies inside the other**
at a path-component boundary (`a` vs `ab` disjoint), so a junction cannot let deletion of one job remove another's
folder. Job names must be direct children of the training root; `.aitk_locks` (the deletion-lock directory) is
reserved in every spelling Windows maps to it.

**Serialization.** With Prisma 6.3.1 an interactive SQLite transaction holds the database write lock from BEGIN
(`prisma-lock-probe.json`: a second client's transaction does not even read until the first ends; plain reads still
proceed under WAL), so start/admit/stop/edit/delete transactions from any worker or route run one after another and
each sees all earlier commits. The `RunClaim` primary key additionally makes a duplicate claim on one resource
impossible even if a check were stale (tested by injecting a claim after the read: P2002, whole admission rolled
back). Busy transactions are retried; a route that stays busy returns 503 without changing anything.

**Ownership predicate.** A job owns its resources while it has `RunClaim` rows of kind `run`, or, for a row last
launched before this schema ("legacy": `run_id` null), while it has an active status or any recorded `pid` (the old
code never cleared `pid` on a normal exit). Display status is not ownership: Python writes completed/error/stopped/
queued before its process exits.

**Run boundary** (`toolkit/ui_procctl.py`, stdlib only, run as `python -I -S`). Process chain: worker -> relay
(`ui_procctl.py launch`, a child of the worker that exits) -> per-run supervisor (`ui_procctl.py supervise`, detached,
so training survives normal UI worker shutdown as before) -> root (`pythonw run.py <per-run config>`) and its
descendants. The run id appears in every command line (per-run config `.job_config.<run_id>.json`), and the worker
records its own `pid@creation` and boot id on the run at admission.
* Windows: the supervisor first publishes an exclusive **boundary claim** (`.run.<id>.boundary.json`, a complete
  record hard-linked into place: never replaces an existing file), then creates a FRESH named job object
  `Global\aitk-run-<run id>` (`ERROR_ALREADY_EXISTS` = launch failure) with `KILL_ON_JOB_CLOSE` and no breakaway, and
  creates the root with `PROC_THREAD_ATTRIBUTE_JOB_LIST`, so the root and every descendant are members from their first
  instruction (an unassigned root never runs). It writes an atomic spawn receipt (`pid@creation` of root and
  supervisor, boot id), records members (completion-port messages plus pid-list polling; auxiliary only), and writes
  the **exit receipt** only after the kernel's `ActiveProcesses` count is 0.
* Linux (source only): the supervisor is `PR_SET_CHILD_SUBREAPER` (needs pidfd: Linux 5.3+, Python 3.9+, else the
  launch is refused) and writes the exit receipt only when `waitpid(-1)` reports no children at all. Force stop asks it
  (pidfd-signalled, identity-checked) to kill its **current direct children** repeatedly as orphans are adopted; no
  ancestry walk.
* Identity: `pid@creation` (Windows exact 64-bit FILETIME as a decimal string; Linux boot id + start ticks). A pid now
  showing another creation time proves the recorded process exited and is never signalled. `WAIT_FAILED`, denied or
  unreadable states are `unknown`.
* **Release evidence** (attached or unattached run): a valid exit receipt (bound to run id AND supervisor identity);
  or the job object still OPEN with `ActiveProcesses == 0` and its supervisor (the only possible spawner) proved gone;
  or a boot-id change since the run was recorded. **A job object that can no longer be opened is never evidence**: the
  name disappears when the last handle closes, before members finish terminating (measured: Opus
  `job_name_lifetime_probe.json`, ERROR_FILE_NOT_FOUND at 0.08 ms with the owned child alive until 5.5 ms; Codex
  `job-name-lifetime-acceptance.json` with a correctly escaped Global name, 1.4 ms, not an upper bound). Recorded
  members are incomplete by construction. **Limitation:** if a supervisor dies without its exit receipt (e.g. it is
  killed), the run's GPUs and folder stay reserved until a restart of the computer proves the exit; the job shows
  "The run's supervisor ended without confirming ... a restart of this computer clears this." No timer releases it.
* Force stop = `TerminateJobObject` on the run's job (Windows); no pid is signalled. Termination is asynchronous; only
  the kernel's count reaching 0 counts, and claims are released only on the receipt/evidence above.
* Command-line scans (needle = run id / legacy config path) are diagnostics only: a process that cannot be queried
  (denied) or read makes a scan incomplete, and no release decision uses a negative scan.

**Python fencing** (phase03, `toolkit/ui_run_fence.py`, now active because the launcher passes `AITK_RUN_ID`): every
Job read/write of DiffusionTrainer, UITrainer, BaseCaptioner and UIJobStatus is `WHERE id = ? AND run_id = ?`; a write
matching nothing or a missing row means the run was superseded/deleted and the process stops; a fenced process that
cannot confirm its row refuses to start.

### 13.2 Run states and transitions (M10, current)

Job statuses: `stopped | completed | error | queued | starting | running | stopping | deleting | delete_failed`.
Launch phases (`launch_phase`): `claimed -> spawn_requested -> attached -> exited`. Every release re-checks, inside its
transaction, the phase (and for `attached` the root identity) its evidence was gathered for; stale evidence never
releases a run that has moved on (`reconcile_phase_race_probe` reproduction; now kept).

| From | Event | Guard | Actions | To | Failure / retry / concurrency |
|------|-------|-------|---------|----|-------------------------------|
| stopped, error, completed (the UI offers it for completed only when steps were added) | Start (route) | not deleting; no claims; no legacy occupancy; valid gpu_ids and name | queued, new queue position, queue row ensured | queued | 409 while starting/running/stopping/deleting/delete_failed or while the last run's process is unverified ("reported completed but its process has not exited yet"); 400 bad gpu_ids/name; repeat Start on queued = 200 no-op |
| queued | admission (worker tick) | the job's own queue is running AT COMMIT; every resource free of run claims, deletion reservations and legacy occupancy (overlap rules above) | `run_id`, claims, launcher identity + boot id, `status=starting`, `phase=claimed` | starting/claimed | conflict: stays queued, `ownership_note` says what blocks it ("GPU 0 in use by job X (stopped, process still exiting)"); paused after the tick read: not admitted; P2002/busy: next tick |
| starting/claimed | launch preparation (settings, folder, log rotation, per-run config, run.py check, platform) | launch registered in-flight synchronously at admission | — | same | any failure: KNOWN pre-spawn, released as error (nothing spawned); unsupported platform refused here |
| starting/claimed | stop request | `stop` set before spawn | release "stopped before its process started" | stopped | — |
| starting/claimed | about to spawn | `stop=false`, phase still claimed | `phase=spawn_requested` committed BEFORE the relay is spawned | spawn_requested | DB failure: still pre-spawn |
| spawn_requested | relay cannot start (spawn error) | — | release | error | — |
| spawn_requested | spawn receipt `spawned` (relay exit, or reconciler) | receipt run id + valid identities | `phase=attached`, root/supervisor identity, boundary kind, boot id; `running` only if still `starting` | attached | attach DB write fails: launch reports "uncertain", claims kept; the reconciler attaches from the durable receipt |
| spawn_requested | receipt `failed`, or relay exit 2/3 | program never started / receipt failed and the supervisor terminated the whole job and saw it empty | release | error | — |
| spawn_requested, no receipt | reconciler | boundary unclaimed AND launcher proved unable to continue (relay exit recorded, or this worker with nothing in flight, or admitting worker proved gone) | FENCE (hard-link a fence record to the boundary claim): wins only if no supervisor claimed, and then no supervisor ever can | released (error) | launcher may still continue: kept; fence lost to a supervisor: next tick; fence I/O failure: kept |
| spawn_requested, no receipt | reconciler | boundary claimed by supervisor S | S alive: kept; S proved gone and job open with 0 active: release; S gone and job absent/active>0/unknown: kept ("restart clears this") | — | never by time |
| attached | Python progress/terminal writes | fenced on run_id | step/info/status | same; terminal status keeps claims | 0 rows: superseded, process stops itself |
| starting/running | Stop (route) | run owned | `stop=true`, `stop_requested_at`, `after_exit=stopped`, status `stopping` (not overwriting a terminal status) | stopping | nothing is signalled on any platform (the trainer's stop watcher raises SIGINT in-process) |
| stopping | stop older than 60 s (reconciler backstop) | boundary open, active > 0 | one force stop per 30 s | stopping | termination is not exit proof |
| attached | Force stop (route) | identified boundary | `TerminateJobObject`, wait for count 0, release on the exit receipt | stopped | not attached: 409 "still starting"; legacy: 409; unsupported platform: 501; partial: 500 with remaining count; claims kept in all these cases |
| running (queue paused) | queue tick | the queue is still paused AT COMMIT; only that queue's own jobs | `return_to_queue`, `after_exit=queued`, stopping | stopping | a queue started again after the read: untouched |
| attached | exit evidence (receipt / open+empty+supervisor gone / boot change) | phase + identity unchanged | delete claims, `phase=exited`, status: Python's completed/error stand, else `after_exit`, else error "exited unexpectedly. Last log line: ..." | final | DB failure: retried next tick |
| attached | supervisor gone, no receipt, job object absent | — | nothing | same | reserved until restart (limitation above) |
| claimed (another worker) | reconciler | admitting worker proved gone | release (nothing spawned) | error | worker alive or unknown: kept |
| any phase | reconciler | recorded boot id differs from the current one | release | final | boot id unreadable: no proof |
| queue running, nothing to do | idle stop | AT COMMIT: no queued job and no job of that queue that may own a process (active status, claimed/spawn_requested/attached run, legacy pid) | `is_running=false` (conditional) | stopped queue | an enqueue + start, or another worker's admission, after the read keeps it running |

Edit (`POST /api/jobs` with id): editable status AND no claims AND no legacy occupancy; rename checked for folder
collisions/reservations (409). Create: folder-key alias of any existing job, or a folder overlapping a run claim or
deletion reservation, is 409. Mark as Stopped: a display reset only for a row that owns nothing (409 otherwise; no
confirmation-click bypass).

**Legacy rows** (pre-update): occupy their normalized GPU set (empty/unparsable = all GPUs) and folder; are never
signalled; Stop still sets the flag the old process polls; force stop is refused. They are cleared ONLY by a boot-id
change since the row was first observed by this code (the boot id is recorded at first observation; writes to legacy
rows preserve `updated_at`). Dead-looking pids and empty scans are not proof (Codex Q1). **Deployment note:** stop all
jobs before upgrading; any legacy row with a recorded pid (every finished job of the old code) keeps its GPU and
folder reserved until the next restart of the computer, with the diagnostic "Started before the run-ownership
update...". The schema push itself (`npm run update_db`) is not run here; on this host the isolated `prisma db push`
failed with an empty "Schema engine error" unless `RUST_LOG=info` was set (test harness note; production behaviour not
verified).

### 13.3 Deletion (M35, `deleteJob` in `ui/cron/ownership/store.ts`)

| From | Event | Guard | Actions | To | Failure / retry |
|------|-------|-------|---------|----|-----------------|
| any | Delete request | acquire `<training root>/.aitk_locks/delete.<job>.lock` (crash-released, see 13.5) | — | holding | another Delete holds it: 409 "being deleted by another request" (no hang) |
| stopped/completed/error/queued/delete_failed, or deleting (a crashed or unfinished earlier attempt; only under the lock) | begin/resume | no claims, no legacy occupancy, no other owner of this folder or a nested/containing one, valid name | `status=deleting`, folder reservation `RunClaim(kind=delete)` (in one transaction) | deleting | 409 owned; 400 invalid/reserved name |
| deleting | remove folder | (no DB transaction held) | `fs.rm` recursive | removed | failure: `delete_failed`, reservation kept, retry allowed |
| deleting | finish | — | delete reservation + row | gone | DB failure: 500 "retry Delete", still deleting + reserved; next Delete resumes |

**Retry in the UI** (phase06, Codex `delete-ui-retry-probe.json`: the actual action helper returned every action
false for a persisted `deleting` row, so a deletion interrupted by a server stop could not be retried from the UI):
`getAvaliableJobActions` now offers Delete for `deleting` and `delete_failed` rows that own no run, flagged
`isRetryDelete`; the button is labelled and confirmed as "Retry Delete", and bulk delete uses the same helper (retries
are named in its confirmation, owned jobs skipped). The browser never guesses whether an earlier deleter is alive: if
one still holds the deletion lock, the server refuses the retry (409). Test: `ui/tests/jobActions.test.cjs` (actual
helper, and the actual `JobActionBar` evaluated under the simulated host with its UI libraries stubbed: button
enabled and labelled, confirmation text, Delete request sent).
While `deleting`/`delete_failed`: Start/Edit refused; creating any name whose canonical folder equals or overlaps the
reserved one refused; admission into it refused. Original: no reservation at all (row deleted after `rm`); the phase04
draft without the lock left a permanent `deleting` row after a crash (Codex `delete_crash_retry_probe.json`, real child
process); now resumed.

### 13.4 Browser request ownership (M18, `ui/src/utils/requestOwner.ts`)

Each hook owns a `RequestOwner`: `reset(key)` on identity change (job id, ref, list filter), `begin()` per request,
`mutate()` for a manual refresh or a mutation (invalidates everything issued before), `accept(ticket)` (same identity,
no mutation since issue, newer than anything applied). State setters, cursor writes, status and in-flight release run
only for an accepted/owned ticket, and a callback created for an older identity (stale closure, queued timer) cannot
start a request under the new identity (`isCurrent(key)`). `usePollLoop` passes `isCurrent()` to its callback.

| Hook / caller | Kind | Rule | Original defect (evidence) |
|---------------|------|------|----------------------------|
| `useJobLog` | cursor | one read in flight per job; cursor = byte offset + log-file generation (`ino`-`birthtimeNs` from the opened handle); generation mismatch or truncation => reset | old job's text and offset written into the new job's terminal; old finally released the new slot (hooks tests, original fails) |
| `useJobLossLog` | cursor + mutation | one read per job; `since_step` cursor tied to the run id returned by the loss route (another run => full reload, history kept); `deleteRange` bumps the epoch before and after the DELETE, then clears the cursor and reloads; a DELETE resolving after a job switch is not applied; the run-change reload timer is cancelled on switch/unmount and identity-checked | a GET issued before the DELETE put deleted points back; (draft only) a queued retry populated the next job, Codex `loss-retry-identity-probe.json`, accepted fixed |
| `useJob`, `useJobByRef`, `useSampleImages`, `useFilesList`, `useJobsList`, `useQueueList` | snapshot | newest issued wins; manual refresh supersedes earlier polls; previous identity dropped | out-of-order refreshes regressed status; a previous id's response replaced the new one |
| generate page status poll, job page plugin poll | loop | `isCurrent()` checked after every await | an old loop's response could clear a newer start and alert |
| inspected, unchanged | — | `useCaptionBatch` (AbortController per effect), `JobNotes` (cancelled flag; a late save causes at most one redundant save), `OstrisCloudBalance` (no identity, serialized loop), `DatasetImageCard` thumbnail fetch (abort + cancelled) | no confirmed defect |

Log route (`/api/jobs/[id]/log`, parent repair reviewed): opens first, `fstat`s and reads THAT handle, advances the
offset by `bytesRead`; the cursor never splits a UTF-8 character: a trailing sequence that is still a VALID prefix
(valid lead byte and, for E0/ED/F0/F4, a second byte inside its allowed range: no overlong, surrogate or above
U+10FFFF prefix) waits for the next read (at most 3 bytes); anything already invalid passes through as U+FFFD (phase06
fix: E0 80, ED A0, F0 80 and F4 90 tails were withheld, Codex review 22:25); a log that ends for good inside a valid
prefix keeps those bytes unshown (documented limit). Evidence is from the **simulated React host** (`ui/tests/harness/reactHost.cjs`, a
hand-rolled implementation of the hook semantics used here; no React test renderer or browser is installed) and is
not browser-scheduling evidence.

### 13.5 Operation lock and uploads (M19)

**Operation lock** (`ui/cron/ownership/opLock.ts`): `BEGIN EXCLUSIVE` on a per-operation sidecar SQLite file for the
whole operation; the OS releases it if the holder dies; waiting is bounded (`busy` result); there is no owner file to
judge stale and nothing is unlinked behind a holder. Durable state never lives in this transaction.

**Upload sessions** (`ui/src/server/uploadSession.ts`, route `ui/src/app/api/loras/upload/route.ts`): one directory per
upload under `<MODELS_PATH>/loras/.uploads/<id>/` with `state.json` (atomic temp+fsync+rename), `data.part` and
`op.lock`. Name, size and overwrite are fixed at start.

| From | Event | Guard (under the session lock) | Actions | To | Failure / retry |
|------|-------|-------------------------------|---------|----|-----------------|
| — | start | name/size valid; no existing file unless overwrite | session dir, empty part, state | receiving | 409 exists |
| receiving | chunk(offset, bytes) | `offset + len <= size`, `len > 0`; no overlap with a recorded chunk | positional write loop (handles short writes), fsync, THEN record `{len, sha256}` | receiving | same offset+len+sha256 = 200 duplicate; same offset different bytes / overlap = 409; beyond size = 400; an interrupted write is not recorded and its retry rewrites the range; lock busy = 503 (client retries) |
| receiving | finish | complete coverage `[0,size)`, part size = size, destination free unless overwrite | record intent `{dest, mode link/rename, part ino, size}` durably, then publish | publishing | holes: 409 with missing ranges |
| publishing | publish (link: no overwrite) | dest absent and part is the recorded file | `fs.link(part, dest)`, state published (result kept), unlink the part name | published | EEXIST: back to receiving + 409; EPERM/ENOTSUP/EXDEV/EINVAL/ENOSYS: `publish_unsupported`, data kept, explicit 500 (no copy fallback); other errors: still publishing, retry |
| publishing | publish (rename: overwrite) | part is the recorded file | `fs.rename(part, dest)`, published | published | — |
| publishing (interrupted after link/rename, before the state write) | next finish or cancel | dest has the recorded identity | finish the transaction | published | dest is another file and the data has moved: `indeterminate`, 500, nothing overwritten, evidence kept |
| published | finish (lost-reply retry) | — | return the recorded result | published | — |
| receiving/unsupported/indeterminate | cancel | — | write `cancelled` FIRST, then remove the data | cancelled | the cancelled write fails: nothing changed (500); removal fails: stays cancelled with `cleanup_error`, a later Cancel retries; note-write failure after cleanup keeps cancelled |
| cancelled | chunk/finish | — | — | cancelled | 410 |
| any old session | expiry (at start) | over 24 h, not busy, not publishing/unsupported/indeterminate | remove under its lock | gone | busy: left |

Client (`LoraBrowserModal.tsx`): retries chunks and finish on transport failures and 503 only. Original route evidence
(Codex `upload_probe.cjs`/`upload-probe.json`, real temporary files): two racing finishes both 200 with the loser's
bytes published; a 1-of-4-byte upload published zero-padded.

### 13.6 Tests (UI phase04)
Isolated Prisma client generated under the scratch folder from the current schema, temporary SQLite files, read-only
existing `node_modules`.

| File | Scope | Result |
|------|-------|--------|
| `ui/tests/ownership.test.cjs` | actual routes/store/reconcile/queue on temporary SQLite, two Prisma clients, scripted process layer (labelled), real Windows junctions, migration from the original f6c137b4 schema with `prisma db push` (no data-loss flag) | 29 passed (ownership-9) |
| `ui/tests/launch.test.cjs` | REAL hidden CPU processes through the actual relay/supervisor (`ui/tests/fixtures/fake_run.py`): attach from receipt, early Python ending keeps the GPU, orphaned grandchild after its parent died, delayed child, receipt failure kills the whole job, pre-spawn failures, a relay held 31 s past the old timeout, post-spawn DB failure, dead admitting worker, actual kill route, supervisor crash before receipt with a tokenless child (kept), delayed supervisor loses to the fence, unsupported platform | 12 passed (launch-5); after-hook scan finds nothing of any test run |
| `tests/test_ui_procctl.py` | helper classification: WAIT_FAILED, reused pid, `pid@None`, no parent-link membership, enumeration error, denied = incomplete, existing job name refused, boundary claim/fence exclusivity, supervisor refuses after a fence, Linux direct-children selection (deterministic table), boot id | 12 passed |
| `ui/tests/deletion.test.cjs` | REAL child Node processes: crashed deleter resumed, overlapping delete 409, finish-DB failure, removal failure | 4 passed |
| `ui/tests/upload.test.cjs` | REAL second Node processes on real temporary files (13 cases above) | 13 passed |
| `ui/tests/hooks.test.cjs` | actual hooks under the simulated host with scripted, explicitly ordered responses | 11 passed; on the original hooks 10 fail (the `usePollLoop` case uses the new helper; the loss-retry case is a draft regression, the original had no retry) |
| `ui/tests/logStream.test.cjs` (Codex), `ui/tests/logUtf8.test.cjs` | actual log route on real files: rotation before open, short read, split multi-byte characters, malformed bytes, a table of invalid vs valid prefixes | 5 passed |
| `ui/tests/jobActions.test.cjs` | actual action helper + `JobActionBar` (simulated host): Retry Delete | 3 passed |
| all nine UI files together (`ui-final-2`) and `typecheck-final-2` | settled phase06 tree | **85 passed**; 0 / 0 diagnostics |
| `ui/tests/terminalEmulator.test.cjs` | terminal emulator (M34) | 8 passed |
| TypeScript | `ui_typecheck.cjs` (Codex): both `tsconfig.worker.json` and `tsconfig.json`, installed TypeScript 5.7.3, isolated Prisma types | 0 / 0 diagnostics |

Agreed plan coverage: U-R1 start refusals and terminal-but-live, U-R2 concurrent/normalized/PK admission, U-R3 early
Python ending, U-R4 force-stop failure, U-R5 every launch phase, U-R6 stale writers and identities, U-R7 legacy +
original-schema migration, U-R8 delete interleavings and aliases, U-R9 edit refusals: `ownership.test.cjs` (named by
id) plus the real-process cases in `launch.test.cjs`; U-R10 Python fencing: `tests/test_ui_run_fence.py` (phase03,
committed); U-R11 `hooks.test.cjs`; U-R12 `upload.test.cjs`. Decision-specific cases (delayed relay, worker death
before/after spawn, post-spawn DB failure, crash before receipt, fence race, receipt failure, unsupported platform,
deletion crash/overlap, queue races, folder ancestry, realpath errors, cancel ordering) are in the same files.

Reproduce: see section 16 ("UI environment").

### 13.7 Platform and coverage limits (UI)
* Windows behaviour is tested with owned hidden CPU fixtures only (no GPU, no real training, no browser, no live UI
  service). Survival of a run when the UI worker is shut down follows the original design (the supervisor is outside
  the worker's process tree) but was not exercised against real UI shutdown or external tree killers.
* Linux supervisor, subreaper wait and pidfd signalling are source-only; macOS/other refuse before spawn.
* A supervisor that dies without its exit receipt leaves the run reserved until a restart (no automatic recovery).
* Legacy rows clear only after a boot change relative to the baseline recorded by the new worker. The new worker must first observe those rows; a reboot before that observation is not proof. Deployment requires the live schema migration (not done).
* Multi-session Windows use: the `Global\` job namespace is shared across sessions; a worker in another session of
  the same user has not been tested.
* Browser hook evidence is simulated-host evidence; the UI pages themselves were type-checked, not rendered.

---

## 14. Issue groups (impact-ranked), agreement and status

| Group | Impact | Issues | Agreement | Status | Regression (current / original) |
|-------|--------|--------|-----------|--------|---------------------------------|
| SM-15a | High: two processes on one GPU/folder, phantom rows, unverifiable kills, writes from superseded runs | ownership/launch/reconcile/force stop/legacy: C01, C09, OPUS-05/06/07, U1-U5; draft findings fixed: parent-link membership (U2 rejected), absent-job-name proof (Q4), stale-phase release, preparation before in-flight, realpath errors, Linux ancestry walk | Opus + Codex (rev. 2, implementation decision, boundary decision rev. 2) | accepted and committed (`4fe31be2`); independent phase-CAS/boundary review plus Windows process/SQLite regressions; Linux source-only | `ownership`, `launch`, `test_ui_procctl` / C01 probe, draft probes |
| SM-15b | High: folder deleted under a live job, stuck or concurrent deletion, queue paused/stopped under a new run | deletion reservation + resumable deletion + Retry Delete UI, canonical folder (junction aliases, nested folders, reserved lock name), three queue snapshot races | Opus + Codex | accepted and committed (`4fe31be2`); independent junction/nested-folder/three-queue-race/Retry-Delete probes plus real-process deletion regressions; browser interaction not run | `ownership`, `deletion`, `jobActions` / `delete-crash-retry-probe`, `delete-ui-retry-probe`, queue/folder probes |
| SM-15d | High (data): an upload overwrites another or publishes holes; a crash after publishing or a failed cancel corrupts the session | C08; publication intent and recovery, content-checked retries, cancel ordering, expiry retention | Opus + Codex | accepted and committed (`c1e972bb`); independent cancellation probe, full protocol source review and 13 real-file/cross-process regressions | `upload` / `upload-probe` |
| SM-01 | High: unresumable RAWR pairs, partial windows, post-update OOM continuation, fused accumulation | OPUS-01, C11, commit stage, R01 | Opus + Codex | committed (`b83505f9`) | loop/commit/fused/observer tests / P1-P4 |
| SM-02 | High (low-precision optimizers) | C10, OPUS-09 | Opus + Codex | committed (`b83505f9`) | 20 behavioural fails on original; P5, P7 |
| SM-09 | High (engine): worker death, stuck requests, unowned startup/shutdown | C02, C15 | Opus + Codex | committed (`b83505f9`) | lifecycle tests / original fails |
| SM-16 | High for schedule-free users | C12, C16, validation drift, MiniCPM resume, MiniCPM finalization order (phase04) | Opus + Codex (option A; finalization order agreed) | committed (`b83505f9`, finalization `edf320f2`); independent real-CPU-optimizer acceptance | modes/groups/finalization tests / probes + C16 test |
| SM-15c | Medium: another job's or an older response shown, deleted points restored, log text mislabelled or split | C07 hooks, log generation, log handle (Codex repair), UTF-8 split and invalid prefixes, loss retry timer, stale closures | Opus + Codex | accepted and committed (`f7a4ba14`); independent loss-retry/log-handle probes and UTF-8 source review; actual files and simulated React tests | `hooks`, `logStream`, `logUtf8` / original hooks 10 fail, log probes |
| SM-19 | Medium for `G=-1`: N+1-batch first window, phase shift, no update in an N-step run | C13 | Opus + Codex (decision rev. 2) | committed (`b83505f9`) | 13 + tiny-SD case / AST probe |
| SM-05 | Medium: borrowed state not given back or not exactly | OPUS-10, R02, R03/C14, P-04, validation activation | Opus + Codex | committed (`b83505f9`) | sampling/EMA/save tests / 14 + 5 behavioural |
| SM-06 | Medium: base-weight drift per sample | OPUS-08 | Opus + Codex | committed (`b83505f9`) | drift probe on original |
| SM-07 | Medium: truncated caches trusted/dropped; borrowed device preset not restored | OPUS-13, P-01, R04 | Opus + Codex | committed (`b83505f9`) | latent 13 + embedding 4 / 8 + 4 fail |
| SM-17 | Medium (Windows): resume rewinds; newest save pruned | OPUS-11, P-07 | Opus + Codex | committed (`b83505f9`) | 10 / 4 fail |
| SM-18 | Medium (Adafactor `stochastic_rounding: false`) | P-03 | Opus + Codex | committed (`b83505f9`) | 3 / 2 fail |
| SM-20 | Medium/unknown: `G` ignored by alternative trainers | M27 | Opus + Codex (decision) | committed (`b83505f9`) | 24 tests |
| SM-04 | Medium: silent RAWR restart/rewind | OPUS-02, OPUS-12 | Opus + Codex | committed (`b83505f9`) | 6 / behavioural + probe |
| SM-10/11/12 | Medium: dirty weights after failed LoRA apply; unload race; duplicate end frames, leaks | C03-C06, OPUS-16 | Opus + Codex | committed (`b83505f9`) | lifecycle tests |
| SM-13 | Medium for swapping users | swapping | Opus; Codex source/test acceptance | committed (`b83505f9`) | 19 / 17 |
| SM-14 | Medium: lost captions | captioner | Opus; Codex source/test acceptance | committed (`b83505f9`) | 2 / 1 |
| SM-22 | Low: manager browser watcher outlives a failed launch; opens a browser after the UI stopped | M33 | Opus + Codex (guidance) | accepted and committed (`7906592c`); targeted CPU/injected acceptance | 4 / original probes |
| SM-21 | Low (P3): log view memory growth on large cursor-down escapes | Codex terminal probes (overflow, retention) | Opus + Codex | accepted and committed (`f7a4ba14`); Codex retention correction reviewed by Opus | 8 / 4 fail |
| SM-08 | Low: second interrupt during graceful stop | OPUS-14 | Opus + Codex | committed (`b83505f9`) | 4 / 4 |
| — | Test harness created a CUDA context (empty `CUDA_VISIBLE_DEVICES`) | phase03 finding | Opus + Codex | test fixed (`-1` + assertion); manual tools set to `-1` (not run) | 15/15 probe records `cuda_initialized: false` |
| — | Non-RAWR exact continuation with `G > 1` | OPUS-15 | documented limit | `docs/RAWR_RESUME.md` | — |
| — | Schedule-free evaluation-point export | — | Codex decision A | unresolved design limitation | — |

---

## 15. Behaviour changes users can notice
* `G > 1`: saves and UI samples keep the unfinished window; an OOM before the update skips the whole window, after
  it aborts; fused optimizers (incl. Automagic3/4 defaults) refuse `G != 1` / `B > 1`; swapping refuses `G != 1`.
* RAWR: a resume whose paired optimizer cannot load is refused; non-RAWR training into a committed-pair folder is
  refused.
* Training previews sample through LoRA hooks (slower on unquantized, non-offloaded models; identical base weights).
  Inverted-assistant/inference LoRAs sit on the CPU from load (previously from the first sample); one in another dtype
  than the model is refused at sampling.
* Schedule-free optimizers train from the first step; validation no longer changes their weights or the RNG.
* Adafactor with `stochastic_rounding: false` now updates bf16/fp16 weights (round to nearest).
* Windows: resume and pruning follow the last write, not the creation time.
* Truncated text-embedding/CLIP caches are rebuilt; cache files appear atomically.
* Inference engine: 400 for non-string arch, 503 after shutdown, 409 for `/unload` during a request; a stop during
  model load shuts down cleanly; requests admitted before a failed start are cancelled.
* `gradient_accumulation_steps: -1`: each update is exactly one epoch (previously the first held one extra batch and
  later ones were shifted); the run's final partial epoch is applied; a dataset without a known batch count is
  refused.
* Slider/rescale/concept-replacer/reference-slider trainers refuse `gradient_accumulation_steps != 1` (it was ignored);
  the slider trainer also refuses `gradient_accumulation > 1` (only the first batch was used).
* Job log view: a log that moves the cursor far down no longer grows the page's memory beyond 5000 lines; a new
  run's log is shown from its start (not appended to the previous run's text); characters are never split.
* UI jobs: a job's GPUs and folder stay reserved until its processes are verified gone, so "completed (exiting)" can
  appear briefly; Start/Edit/Delete/Mark as Stopped wait for that and the reason is shown. Jobs on overlapping GPU
  sets (`0` and `0,1`, across queues) never run together; opaque GPU names are exclusive; an empty GPU selection is
  refused; a name colliding by case/junction with another job's folder is refused.
* Stop never sends a signal (the trainer's watcher stops it); a stop that is not honoured within 60 s gets a force
  stop. Force Stop terminates the whole run (all child processes), and is refused for jobs started before the update.
* Upgrade sequence for legacy rows: stop existing jobs, apply the schema/new code, let the new worker record its boot baseline, then restart the computer when other work permits. A reboot before the baseline is recorded does not clear those reservations. A new run whose supervisor died without an exit receipt also stays reserved until a boot change. No upgrade/reboot was performed for this audit.
* Launching jobs on macOS/other non-Windows/Linux platforms is refused in this version.
* Deleting a job that a previous Delete left unfinished completes it: such a row shows "Retry Delete"; two Deletes
  at once: the second is refused.
* LoRA uploads: a finish to an existing name is refused (unless Overwrite) even when two uploads race; a hole, a
  conflicting retry or a replaced destination is reported, never published or overwritten.
* Manager: a failed UI start no longer leaves the browser watcher running; no browser opens after the UI stopped.
* MiniCPM: the final checkpoint holds the trained values; the exported adapter equals the weights used for the final
  loss.

---

## 16. Regression runs (CPU)
| Log label | Tree | Result |
|-----------|------|--------|
| `baseline-tests` | original branch head | 422 passed, 52 skipped |
| `full-suite-after-python-fixes` | phase02, overlapped the last captioner edit (milestone only) | 556 passed, 52 skipped |
| `full-suite-final` | phase02 final, non-overlapping | 557 passed, 52 skipped |
| `r01-fixed-2` | R01 observer tests | 8 passed (pre-repair evidence: source only) |
| `r02-review-fixed-2` | sampling + sigma | 39 passed; original: hard-coded teardown (source), Codex probes on the draft |
| `r03-p04-fixed-2` / `r03-p04-original`, `r03-average-original` | EMA/save | 35 passed / 4 + 1 behavioural fails (`train()` retry passes on both) |
| `r04-cleanup-order` / `r04-original` | latent cache | 13 passed / 8 fail (all 4 R04 error paths) |
| `r01-r04-related-2` | 12 related suites | 230 passed (first run: 26 fails, one cause: my `save()` split broke unbound-namespace callers) |
| `c15-review-fixed` / `c15-original`, `c15-review-original` | inference lifecycle | 23 passed / 7 behavioural fails (+1 absent-API, not counted) |
| `evalscope-combined-failure-2` | schedule-free modes + sampling | 77 passed |
| `c16-fixed` / `c16-original` | two-group schedule-free | 2 passed / 2 fail |
| `opus11-fixed` / `opus11-original` | selection/pruning | 10 passed / 4 fail |
| `p03-fixed` / `p03-original` | Adafactor write-back | 3 passed / 2 fail |
| `p01-clip-loop` / `p01-original`, `p01-clip-loop-original` | embedding caches | 4 passed / 4 fail |
| `automagic4-calibration` | Automagic4 | 3 passed |
| `full-suite-phase03` | phase03 tree before two fixes | 667 passed, 3 failed, 52 skipped (empty-file cache-hit fixture; `save()` order); **this run's test_resume_rng probe created a CUDA context** (section 1) |
| `phase03-suite-failures-fixed` | the 6 affected files | 171 passed |
| `full-suite-phase03-2` | **overlapped my C13 edits** (milestone only, not a valid record) | 666 passed, 4 failed (test_resume_rng, from the in-progress edit) |
| `save-preflight` | RAWR refusal before EMA | 2 passed |
| `c13-numeric` | epoch + accumulation files | 100 passed |
| `resume-rng-whole-epoch` | tiny-SD resume probes incl. `whole_epoch` | 5 passed, 15/15 `cuda_initialized: false` |
| `m27-fixed` | capability guard + teardown | 24 passed |
| `terminal-fixed-2` / `terminal-original-2` | terminal emulator | 5 passed / 4 fail (1 control passes) |
| `terminal-fixed-5` | terminal incl. huge/overflowing counts (not run on the original: unbounded loop) | 7 passed |
| `fence-fixed` / `fence-related` | Python run fencing on temp sqlite3 / writers' other suites | 14 passed / 112 passed |
| `c13-scope-corrected` | epoch + capability files after removing the unreachable `B > 1` branches | 35 passed |
| probes | schedule-free round trip, C13 original window | sections 10 and 3.3 |
| `full-suite-phase03-final` | phase03 final tree (only this document edited during the run) | **722 passed, 52 skipped, 0 failed** (6 min 8 s); 15/15 resume-probe records `cuda_initialized: false` |
| `full-suite-phase04` | phase04 tree (Python changed: `toolkit/ui_procctl.py` new, `manager/launch.py`, `training_extras/minicpm_trainer.py`, new tests; only this document and UI files edited during the run) | **741 passed, 52 skipped, 0 failed** (6 min 12 s); 15/15 resume-probe records `cuda_initialized: false`. Afterwards one comment-only edit in `minicpm_trainer.py` (machine id), re-checked with `minicpm-final-after-comment` (3 passed) |
| `procctl-unit-2`, `manager-launch`, `minicpm-final` | helper / manager / MiniCPM finalization | 12 / 4 / 3 passed |
| `ownership-9`, `launch-5`, `deletion-1`, `upload-3`, `hooks-4`, `hooks-original-2`, `logutf8`, `terminal-logstream`, `typecheck-opus-6` | UI (section 13.6) | 29 / 12 / 4 / 13 / 11 passed; original hooks 10 of 11 fail; 4 / 10 passed; types 0/0 |
| `ui-final`, `typecheck-final` | phase05: all eight UI test files on the settled tree; both TypeScript configs | **81 passed**; 0 / 0 diagnostics |
| `jobactions-1`, `logutf8-2` | phase06: Retry Delete helper/component; log route incl. invalid-prefix table | 3 passed; 5 passed |
| `ui-final-2`, `typecheck-final-2` | phase06: all nine UI test files on the settled tree; both TypeScript configs (no Python changed in phase06, so the phase05 Python suite stands) | **85 passed**; 0 / 0 diagnostics |

**UI environment** (Windows, repository root): `AITK_UI_NODE_MODULES=<an existing ui/node_modules>` (read-only),
`AITK_TS_PATH=<that>/typescript`, `AITK_TEST_PYTHON=<toolkit venv python.exe>`, `AITK_TEST_SCRATCH=<a scratch dir>`,
`CUDA_VISIBLE_DEVICES=-1`, then `node --test --test-concurrency=1 ui/tests/<file>.test.cjs`. The harness
(`ui/tests/harness/`) generates a Prisma client from `ui/prisma/schema.prisma` into the scratch dir using a private
copy of `@prisma/client` (never into `node_modules`), pushes schemas only to temporary SQLite files (with
`RUST_LOG=info`, see 13.2), and transpiles the actual `.ts/.tsx` modules with the installed TypeScript.

Test-design corrections during phase03 (recorded so they are not read as flakiness): a duplicate fixture keyword; the
R01 save-failure injection first missed the non-pair save path; the schedule-free fixture first used a CUDA-only
optimizer (RAdamSF) and a compile-only one (AMUSE); my first mode helper read group keys before the attribute (the
FADOPT resume test caught it); `settle` did not carry secondary notes; an EMA rollback branch re-raised the rollback
error instead of the primary; epoch-test step pre-hooks returned a tuple (must return None) and missed the
`accum_scale` keyword; a cross-realm array comparison in the node test. Existing tests changed:
`tests/test_accelerator_teardown.py` (reads `_run_training`), `tests/test_latent_cache_recovery.py` fixture gained
`save_device_state`, `tests/test_random_crop_percent.py` (a cache hit was an empty file; now a tiny valid safetensors
file), `tests/test_accumulation_window.py` fixture `train_config.steps`, `tests/test_resume_rng.py` (`-1`, CUDA
assertion, `whole_epoch` case), `tests/test_rawr_stochastic_pair.py` (+1 preflight test). Code corrections found by
the suite: `save()` first took the EMA swap before `rawr_pair.begin`; `begin` runs first again and its result is passed
to `_save_checkpoint`.

---

## 17. Coverage and gaps

**Read in this audit (full):** `ui/src/utils/terminalEmulator.ts`, `ui/src/server/apiCache.ts`,
`manager/migrations.py`, `rawr_checkpoint.py`, `rawr_pair.py`, `DiffusionTrainer.py`, `toolkit/optimizer.py`,
`sample_step_hook.py`, `sampling_state.py`, `engine.py`, `server.py`, `InferenceEngine.py`, `ui_job_status.py`,
`toolkit/models/v2/pool.py` (ComponentPool), `toolkit/ema.py` 150-360, `toolkit_validation.py` 40-110, the
`eval`/`train` methods of every vendored schedule-free class, `run.py` 100-175, `jobs/BaseJob.py`, UI files listed in
the UI proposal (rev. 1 and 2 lists).
**Read in part:** `BaseSDTrainProcess.py` (sample, save, selection/pruning, loop, optimizer setup/load), `SDTrainer.py`
1290-1530, 2330-2476; the six alternative `hook_train_loop`s; `dataloader_mixins.py` latent/TE/CLIP caches;
`stable_diffusion_model.py` / `base_model.py` generate_images, device state, assistant loading; krea2/z_image assistant
loading; `automagic4.py` 38-50, 240-265; `adafactor.py` 300-370; `ademamix.py` SAEM 1166-1560 (partial);
`minicpm_trainer.py` 218-236, 344-540; `SDTrainer` multistage branch and the Wan22 14B constructor 208-232;
`config_modules.py` 494-516 (accumulation validation); `jobs/TrainJob.py` process map; `cpu_offload.py` 20-140; `toolkit/data_loader.py` 387-897 (phase02).
**Read in phase04 (full):** every changed UI file (sections 13, 19), all `usePollLoop`/poll callers (hooks, generate
and job pages, `OstrisCloudBalance`, `JobNotes`, `DatasetImageCard` fetches, `useCaptionBatch`), job/queue/log/loss/
upload routes, `inferenceEngine.ts`, `manager/launch.py`, `minicpm_trainer.py` finalization (505-545).
**Not read by Opus, Codex-read only:** `ui/src/extensions/modelArchs.ts`, `manager/env.py`, `manager/nodejs.py`.
**Searched only (not audited):** CaptionMonitor; `toolkit/guidance.py`
backward calls; other captioner subclasses; EMAMix/Nesterov/MARS/AO schedule-free variants at runtime;
`optimizer-index.json`/catalog (names).
**Not read (gaps):** most of `SDTrainer.train_single_accumulation` and loss paths; `BaseSDTrainProcess` network/adapter
setup beyond the listed lines; `MemoryManager` beyond the grad sync helper; model-specific files under
`toolkit/models` and `extensions_built_in/diffusion_models` except the assistant-LoRA sites; ExtractJob, MergeJob,
ModJob, GenerateJob, TrainVAE/ESRGAN processes; UI dataset/image/file/zip/caption routes, settings, `fileServer.ts`;
`toolkit/accelerator.py`, `toolkit/util/quantize.py`; the non-LoRA upload routes (`datasets/upload`, `img/upload`:
single-request writes, not part of the chunk protocol). GPU-only behaviour (offload races, pinned async copies,
CPUOffloadOptimizer, CUDA RNG with real devices, real model sampling) is unverified.

---

## 18. Open decisions and proposals
1. **C13 `B > 1`**: resolved (Codex rev. 2): unreachable branches deleted, `TrainConfig` refusal kept and tested.
2. **SM-15a-d**: independent acceptance complete within the evidence scope; all groups are committed locally (sections 14 and 19). Pre-merge production/GPU/browser/platform checks remain. Decided limits (boundary decision rev. 2): no absence-based exit
   proof; supervisor death without receipt and legacy rows clear only on a restart; macOS refused; Linux source-only.
3. **Schedule-free export point**: option C (out-of-place x for exports) remains future design.
4. **MiniCPM finalization**: resolved (final training-coordinate checkpoint before evaluation; one eval mode through
   export).

## 19. Committed change set (local branch; no main merge)

| Commit | Scope |
|---|---|
| `b83505f9` | Reviewed core Python/training/inference fixes |
| `7906592c` | Stop the manager browser watcher on launch failure and cancellation |
| `edf320f2` | Save MiniCPM training coordinates before final evaluation |
| `4fe31be2` | Fence UI runs and serialize queue, resource and deletion transitions |
| `f7a4ba14` | Keep browser requests and log cursors bound to their current stream |
| `c1e972bb` | Make chunked uploads and publication recoverable across retries |

The audit document is committed separately. The final branch HEAD, file hashes and test provenance are in the packaged audit manifest and `final-source-acceptance.json`.

* SM-01/02/03 + R01: `BaseSDTrainProcess.py` (loop methods, observers, `run()` wrapper, checks), `SDTrainer.py`,
  `DiffusionTrainer.py`, `toolkit/optimizers/pending_grads.py` (new) and the optimizer files, `rawr_checkpoint.py`,
  `tests/test_accumulation_window.py`, `tests/test_resume_reconcile.py`, `tests/test_accelerator_teardown.py`,
  `docs/RAWR_RESUME.md`.
* SM-19 (C13): `BaseSDTrainProcess.py` (`_fetch_batch_list`, cursor, final window, refusals, resume limit message,
  `_cleanup_batch_list`), `docs/RAWR_RESUME.md`, `tests/test_epoch_accumulation.py`, `tests/test_resume_rng.py`,
  `testing/resume_rng_probe.py` (device label).
* SM-20: `BaseSDTrainProcess.py` (capability flags/check), the six alternative trainers, `tests/test_trainer_capabilities.py`.
* SM-04: `BaseSDTrainProcess.py`, `rawr_pair.py`, `tests/test_resume_guards.py`, `tests/test_rawr_stochastic_pair.py`.
* SM-05/06 + R02/R03/P-04: `toolkit/sampling_state.py` (new), `base_model.py`, `stable_diffusion_model.py`,
  `krea2.py`, `z_image.py`, `sample_step_hook.py`, `toolkit/ema.py`, `BaseSDTrainProcess.py` (`sample`, `save`),
  `SDTrainer.py`, `training_extras/toolkit_validation.py`, `tests/test_sampling_state.py`,
  `tests/test_sample_sigma_cleanup.py`.
* SM-07 + R04/P-01: `toolkit/dataloader_mixins.py`, `tests/test_latent_cache_recovery.py`,
  `tests/test_embedding_cache_recovery.py`, `tests/test_random_crop_percent.py`.
* SM-08: `DiffusionTrainer.py`, `UITrainer.py`, `BaseCaptioner.py`.
* SM-09..12 + C15: `engine.py`, `server.py`, `InferenceEngine.py`, `toolkit/ui_job_status.py`, `inference_lora.py`,
  `tests/test_inference_engine_lifecycle.py`.
* SM-13: `adafactor.py`, `automagic.py`, `tests/test_paramiter_swapping.py`. SM-14: `BaseCaptioner.py`,
  `tests/test_caption_publication.py`.
* SM-16: `training_extras/optim.py`, `vendor/.../schedulefree.py`, `training_extras/minicpm_trainer.py`,
  `BaseSDTrainProcess.py`, `tests/test_schedulefree_modes.py`, `tests/test_schedulefree_groups.py`.
* SM-17: `BaseSDTrainProcess.py` (`save_write_rank`), `tests/test_save_selection.py`.
* SM-18: `toolkit/optimizers/adafactor.py`, `tests/test_adafactor_writeback.py`.
* SM-21: `ui/src/utils/terminalEmulator.ts`, `ui/tests/terminalEmulator.test.cjs`.
* SM-15 (Python part): `toolkit/ui_run_fence.py` (new), `DiffusionTrainer.py`, `UITrainer.py`, `BaseCaptioner.py`,
  `toolkit/ui_job_status.py`, `tests/test_ui_run_fence.py`.
* CPU boundary: `tests/test_resume_rng.py`, `tests/test_rawr_trainer_continuity.py` (docstring),
  `testing/rawr_cpu_probe.py`, `testing/rawr_review/review_variants.py`, `testing/rawr_review/review_variant_probe.py`,
  `testing/rawr_checkpoint_acceptance.py` (labels; manual tools, not run).
* M3a: `tests/test_automagic4_calibration.py`. This document.
* SM-15 UI, grouped for review (some files serve several groups and are listed once):
  * SM-15a ownership/launch: `ui/prisma/schema.prisma`; `ui/cron/ownership/{resources,store,launch,reconcile,procctl,
    forceStop}.ts`; `ui/cron/worker.ts`, `ui/cron/actions/startJob.ts`; routes `jobs/route.ts`,
    `[jobID]/{start,stop,kill,mark_stopped}`; `ui/src/server/inferenceEngine.ts`; `toolkit/ui_procctl.py`;
    `ui/src/utils/jobs.ts`, components `JobActionBar`, `JobOverview`, `JobsTable`, `ActiveJobWidget`; tests
    `ui/tests/{ownership,launch}.test.cjs`, `tests/test_ui_procctl.py`, `ui/tests/harness/*`, `ui/tests/fixtures/fake_run.py`,
    `ui/tests/fixtures/schema-f6c137b4.prisma`.
  * SM-15b deletion/queue/folders: `ui/cron/ownership/opLock.ts`, `deleteJob`/`beginDelete`/`stopIdleQueue` and the
    folder rules in `store.ts`/`resources.ts`, `ui/cron/actions/processQueue.ts`, `[jobID]/delete`, Retry Delete in
    `jobs.ts`/`JobActionBar`/`JobsTable`; tests `ui/tests/{deletion,jobActions}.test.cjs`,
    `ui/tests/fixtures/deleteChild.cjs`, the queue/folder cases in `ownership.test.cjs`.
  * SM-15c browser/log: `ui/src/utils/requestOwner.ts`, hooks (`usePollLoop`, `useJob`, `useJobByRef`, `useJobLog`,
    `useJobLossLog`, `useJobsList`, `useQueueList`, `useSampleImages`, `useFilesList`), `app/generate/page.tsx`,
    `app/jobs/[jobID]/page.tsx`, routes `[jobID]/{log,loss}`; Codex's terminal emulator and log-handle repairs
    (`ui/src/utils/terminalEmulator.ts`, `ui/tests/{terminalEmulator,logStream}.test.cjs`); tests
    `ui/tests/{hooks,logUtf8}.test.cjs`, `ui/tests/harness/reactHost.cjs`.
  * SM-15d uploads: `ui/src/server/uploadSession.ts`, `loras/upload/route.ts`, `generate/LoraBrowserModal.tsx`;
    tests `ui/tests/upload.test.cjs`, `ui/tests/fixtures/uploadChild.cjs`.
* SM-22: `manager/launch.py`, `tests/test_manager_launch.py`. SM-16 finalization: `training_extras/minicpm_trainer.py`,
  `tests/test_minicpm_finalization.py`.

## 20. Invariant hardening addendum (baseline `db8dafb5`, finalized 2026-11-07)

Sections 1-19 are historical and preserved. This section records the later invariant-hardening pass and supersedes the
relevant earlier behaviour and status. The baseline is `db8dafb53d7008db21f65e01c7d8aad2b3be1a3d` (`db8`). This is a
bounded record of the 24 named acceptance-matrix issues and the state-machine boundaries they touch, not an exhaustive
proof of every transition.

**Final integration status.** Core and CLI/UI lease integration: `964d79a1`; initial UI guards: `bb21cae1`; data/cache/inference: `cb692933`. Codex independently verified the combined CPU suite: **863 passed / 53 skipped**, with unchanged source hashes. Full UI **101 passed**, both TypeScript configurations **0 diagnostics**. Opus's 510/10 and the data worker's 123/1 focused runs are supporting subsets. Main is unmerged; no GPU throughput/full-model, production DB or live browser validation is claimed.

Evidence labels used below: **C** = confirmed on original code by a recorded probe/original-code behaviour; **S** =
source-supported; **T** = implemented source plus focused worker test (worker-reported; acceptance only where the
final status above says accepted); **H** = hypothesis/unproven. A focused test is not a full-suite, GPU, or production
claim.

### 20.1 What an invariant means here

An **invariant** is a condition the code needs to remain true for a safe result, whether or not any line of code checks
it. Examples: one trainer writes a save root at a time; a cache hit came from the current source bytes and encoder
recipe; an optimizer state belongs to the checkpoint and parameter order it is loaded onto.

An implemented guard or test establishes only a narrower claim: the guard rejects the fixtures/paths it actually
inspects, and a passing test shows those tested transitions behaved as asserted. It does **not** establish that every
writer, filesystem race, model architecture, GPU path, or legacy folder is safe, nor that an unchecked condition is
true. Source-supported means the failure path was read in source; confirmed means an original-code or real route/selector
probe reproduced the bad transition. Absence of a helper in the original tree is not behavioural evidence.

### 20.2 Acceptance-matrix disposition

All 24 acceptance-matrix IDs are listed once below. Counts and final commit mapping for the data work remain parent-owned;
the committed core and UI statuses above are the only accepted integration statuses stated here.

| ID | Group / impact | Evidence | Resulting rule/change or disposition | Test/reference |
|---|---|---|---|---|
| OPUS-RES-01 | Persistence/resume; wrong checkpoint or foreign-file pruning | C: filename-only selection probe; original-code bracket, foreign-folder and sibling-prune behaviours reported | Exact writer-name/suffix save sets, no glob; Wan split halves form one save; incomplete newest skipped; `optimizer.pt` and `training_extras.pt` excluded. Core committed `964d79a1`. | `tests/test_core_save_discovery.py` |
| OPUS-RES-02 | Persistence/resume; optimizer attached to another checkpoint | S; core focused tests | One generation stamped into every artifact; `optimizer.pt` records generation, step, optimizer class, ordered parameter manifest, recipe and metadata-less content hashes. Checked before `load_state_dict`; an interrupted save falls back to the last complete matching save or refuses. Core committed `964d79a1`. | `tests/test_core_resume_identity.py` |
| OPUS-PARAM-ID-01 | Resume/optimizer; moments on wrong parameter names | S; Codex reproduced RAWR + AdamW; strict refusal source/test-confirmed | Ordered `(name, shape, dtype)` manifest; `require_named` refuses free/unnamed optimizer parameters before resume, training or save; versioned positional records are refused before loading; module swap/reorder/class changes refuse; legacy record-less pairs warn only. Core committed `964d79a1`. | `tests/test_core_resume_identity.py` (`test_unnamed_parameters_are_refused_before_training`, `test_reordered_unnamed_parameters_are_refused_without_loading_moments`, `test_named_producers_are_certified`) |
| OPUS-SEC-01 | UI boundary; unauthenticated mutation | C: middleware probe showed mutation routes passing without a bearer token | Public reads are only GET/HEAD under media prefixes; all other `/api/*` require the `AI_TOOLKIT_AUTH` bearer. UI committed `bb21cae1`. | `ui/tests/uiInvariantRoutes.test.cjs`; `ui/tests/uiInvariantLaunchBinding.test.cjs` |
| OPUS-SEC-02 | UI boundary; delete/caption outside roots | C: traversal probe deleted an outside image/caption | Shared lexical + physical mutation resolver; both image and caption targets validated before mutation; caption extension restricted. UI committed `bb21cae1`. | `ui/tests/uiInvariantRoutes.test.cjs`; `ui/tests/uiInvariantLaunchBinding.test.cjs` |
| OPUS-UI-01 | UI launch binding; writes outside claimed run folder | S: stale training folder/name could differ from the admitted folder | Launch rewrites a single supported training process to `training_folder=parent` and `name=job name`; inference `job_folder` is bound; multi-process or unknown writer types refuse. UI committed `bb21cae1`. | `ui/tests/uiInvariantLaunchBinding.test.cjs` |
| OPUS-ONEWRITER-01 | Writer ownership; overlapping writers corrupt saves/prune | S; core focused and cross-language path tests | Exact-root SQLite writer lease acquired before config writes; same-job holders share, other jobs/processes/delete are busy; release on run/constructor/cleanup/finally; distributed start refuses. UI preparation uses the same path formula but the handoff to the spawned trainer is not an atomic lease transfer. Core committed `964d79a1`; UI lease evidence accepted. | `tests/test_core_save_root_lease.py`; `ui/tests/writerLease.test.cjs` |
| OPUS-PRUNE-01 | Save pruning; another job's or newer data deleted | C: mixed LoRA/adapter budget, prefix-sharing job and derived-export pruning reproduced | Per-artifact-kind budgets over exactly the kinds this run writes; split save counts once; final save never pruned; only saves older than the oldest kept complete save go; newer interrupted save remains; a failed optimizer write skips pruning. Core committed `964d79a1`. | `tests/test_core_save_discovery.py`; `tests/test_core_resume_identity.py` (`test_a_failed_optimizer_save_prunes_nothing_and_the_resume_falls_back`) |
| OPUS-CACHE-01 | Data/cache; stale latent from changed media or VAE | S; T focused worker tests | Cache schema v2 latent key includes source SHA-256, schema version, preprocessing policy, model latent-space override and a compact latent-encoder digest. Admission compares the stored key and lazily atomically rebuilds misses/legacy/unreadable files. Data committed `cb692933`; final CPU suite accepted within the limits below. | `tests/test_data_invariants_cache_final.py`; `tests/test_latent_cache_recovery.py` |
| OPUS-CACHE-02 | Data/cache; stale text embedding/control conditioning | S; T focused worker tests | Text key adds text-encoder identity, overrides, ordered reference content identities, first-frame/control flags, target size where applicable, and self-media scale/crop/flip policy. Lazy rebuild on miss; reference hashing is memoized per prepared item. Data committed `cb692933`; final CPU suite accepted within the limits below. | `tests/test_data_invariants_cache_final.py`; `tests/test_embedding_cache_recovery.py` |
| OPUS-MF-01 | Training loss; weighting/accumulation hit unweighted gradient | S; core focused tests | `get_mean_flow_loss` returns the differentiable, per-sample-weighted loss; shared `backward_weighted_loss` owns non-finite handling, reg weight, accumulation scale and the only backward. Core committed `964d79a1`. | `tests/test_core_mean_flow_mask.py` |
| OPUS-EMA-01 | Resume/EMA; raw weights and shadow lost on resume | S; core focused tests | Export remains EMA; `optimizer.pt` carries raw weights, shadow, update count and EMA options; the whole record validates before restore; changed options are printed and the new options used; legacy EMA is explicit "not exact". Core committed `964d79a1`. | `tests/test_core_resume_identity.py` |
| OPUS-UPL-01 | UI upload; acknowledged files overwrite each other | C: `a b.png` and `a_b.png` both accepted, only one survived | Sanitization collisions refused; destinations prechecked; staged publication is no-clobber/atomic with rollback; unsupported filesystem returns explicit failure. UI committed `bb21cae1`. | `ui/tests/uiInvariantRoutes.test.cjs` |
| OPUS-VAL-01 | Data/held-out validation; training sees validation files | S; T with real hardlink and directory-junction aliases | `_file_identity` now returns only a nonzero `(dev, ino)` identity; canonical path equality is separate. Equal/nested roots and actual admitted file-list path overlap refuse before validation admission; real hardlinks under disjoint roots and junction aliases are detected; true prefix siblings are admitted; nested `_controls` roots are conservatively refused. Data committed `cb692933`; final CPU suite accepted within the limits below. | `tests/test_data_invariants_runtime_final.py`; `tests/test_training_extras.py` |
| OPUS-INF-01 | Inference LoRA; reused weights do not match current path | S; T with real tiny safetensors and the real engine path | `InferenceLoRA.load` binds the file generation actually loaded and rejects a generation change during read; stack keys and `set_strengths` compare current generation so the engine reloads; merge-cleanup key stays stable and removal is reload-based. Metadata-only limitation remains. Data committed `cb692933`; final CPU suite accepted within the limits below. | `tests/test_data_invariants_runtime_final.py`; `tests/test_inference_engine_lifecycle.py` |
| OPUS-SAVE-01 | Save naming; two artifacts overwrite one path | S; core focused tests | Decorator gets `{job}_decorator`; startup refuses any two configured artifact kinds sharing a filename (case-folded on Windows). Core committed `964d79a1`. | `tests/test_core_save_discovery.py` |
| OPUS-RAWR-01 | RAWR writer; unsupported split layout aborts only at save | S; core focused tests | Startup and `rawr_pair.begin` refuse pair mode with a declared split LoRA layout; non-pair split layouts remain allowed. Core committed `964d79a1`. | `tests/test_core_save_discovery.py` |
| OPUS-DIST-01 | Distribution; duplicate ranks write one folder | S; core focused test | `BaseTrainProcess` refuses `num_processes > 1` before taking the lease or writing; the message directs a single-process launch. Core committed `964d79a1`. | `tests/test_core_save_root_lease.py` |
| OPUS-DADAPT-01 | Optimizer dispatch; name selected a different algorithm | S; T dispatch tests using a stub optional module | `dadaptationadam` now selects `DAdaptAdam`; `dadaptationlion` selects `DAdaptLion`; bare legacy `dadaptation` selects `DAdaptAdam`; `lion` remains Lion and bare `adam` remains `torch.optim.Adam`. The test uses a stub module and does not run actual DAdapt optimizer training. Legacy Lion state produced by the old `dadaptationadam` path is not certified for the corrected algorithm; use a fresh optimizer for affected legacy saves. Data committed `cb692933`; final CPU suite accepted within the limits below. | `tests/test_data_invariants_runtime_final.py` (`test_dadaptationadam_and_lion_dispatch_to_their_real_optimizers`, `test_bare_legacy_adam_stays_torch_adam`) |
| OPUS-GRAD-01 | Gradient finiteness; hypothesis only | H: no reachable non-finite-gradient path established | Deliberately not patched. The shared loss guard substitutes a zero loss only for a non-finite loss; it does not scan parameter gradients before every optimizer step. Remains unproven. | None |
| OPUS-CAP-01 | Caption randomness with cached text embeddings | S; T focused guard, source/test-confirmed only within the exact options | `cache_text_embeddings` refuses `shuffle_tokens` and `random_triggers` separately at cache admission; a concrete caption-draw test shows those options produce multiple captions across accesses. Static normal captions and the supported caption-dropout cache path complete. This is not acceptance-confirmed beyond those options. | `tests/test_data_invariants_cache_final.py` (`test_cached_embeddings_refuse_dynamic_caption_redraws`, `test_static_and_dropout_caption_cache_paths_remain_accepted`) |
| INV-DATA-01 | Data loader; bucket replacement changes shape | S; T focused worker tests | Bucket mode returns only a same-bucket index or a deliberate `RuntimeError`; no cross-bucket fallback. Failed-item retry remains bounded at 11 load attempts. Data committed `cb692933`; final CPU suite accepted within the limits below. | `tests/test_data_invariants_cache_final.py` (`test_replacement_index_stays_in_the_source_bucket`, `test_singleton_and_missing_bucket_fail_clearly`, `test_failed_item_retry_stays_bounded`) |
| INV-PATH-02 | UI boundary; nested junction escapes root | C: pre-existing directory junction deletion removed an outside file | Physical resolver rejects a nested link resolving outside a configured root while preserving an explicitly linked root; lexical component checks prevent sibling-prefix/traversal. Check-then-act swap remains outside the guarantee. UI committed `bb21cae1`. | `ui/tests/uiInvariantRoutes.test.cjs` |
| INV-MASK-01 | Training mask; empty support produced NaN weights | C: empty foreground and all-white inverted-prior helpers produced non-finite weights; caller consequence source-supported | Support is read from the pixel mask; zero-mass batches yield zero additive weights instead of 0/0; ordinary masks remain bit-identical; selection is device-side without a host sync. Core committed `964d79a1`. | `tests/test_core_mean_flow_mask.py`; `tests/test_sdtrainer_masks.py` |

### 20.3 Enforcement boundaries and transitions

#### 20.3.1 Exact-root writer lease (training, launch preparation, deletion)

**Invariant:** for one canonical save root, at most one writer/operation acts as owner; sequential trainer objects of the
same job in one Python process may share the training ownership. **Old behaviour superseded:** exclusive writing was assumed, not enforced;
UI launch preparation and CLI trainers did not share a lifetime lease. The lease is an SQLite `BEGIN EXCLUSIVE` on
`<canonical parent>/.aitk_locks/writer.<sha256(canonical root)>.lock`, outside the folder so deletion cannot unlink it.
Python `toolkit/save_root_lease.py` and TypeScript `ui/cron/ownership/writerLease.ts` compute the same path.

| State | Transition | Guard/result |
|---|---|---|
| Free | Training constructor acquires | `BaseTrainProcess.__init__` takes the lease before tensorboard/config writes. |
| Held by job J, holder H | Another holder of the same Python job object in the same OS process | Shared connection; holder added. |
| Held by job J | Another job/process or UI deletion acquires | SQLite busy -> `SaveRootBusy` / HTTP 409; no config, checkpoint, or deletion mutation. |
| Held by UI launch preparation | `withOpLock(writerLeasePath(folder))` around `mkdir`, log rotation and atomic config write | Lock released before spawning; the Python constructor must acquire the lifetime lease next. If another trainer wins, Python refuses before training writes. This handoff is not an atomic lease transfer. |
| Held by trainer | UI `deleteJob` takes `delete.<job_id>.lock` then the writer lock | Busy -> 409 preserves row, claims, folder and checkpoint bytes; deletes only after acquiring the writer lock and the `expectedFolder` admission check. |
| Held | Last holder releases, run ends, constructor fails, or process dies | Explicit release on run/cleanup/finally; SQLite OS lock drops on process death. Sidecar file remains and is never unlinked. |
| Free | Recreate deleted folder and acquire | Same canonical path maps to the same sidecar lock, so a stale/replacement folder does not create a second writer window. |

The lease covers the **exact canonical root** and cooperating writers using this protocol. It does not fence arbitrary
nested CLI roots, unrelated programs, or network filesystems that do not honour SQLite locks. Hostile filesystem swaps
remain outside all path/lease guarantees.

#### 20.3.2 Checkpoint + optimizer identity (including EMA and named manifests)

**Invariant for versioned records:** optimizer state must match the save generation, optimizer class and ordered trainable
parameters it was written with. Legacy record-less state follows the explicitly uncertified compatibility path below. **Old behaviour superseded:**
optimizer state was mapped positionally to whatever checkpoint was newest, and an EMA resume continued from the EMA
export without the raw weights/shadow. The new check certifies pairing, not bit-exact continuation of scheduler,
dtype, accumulated state, or every underlying tensor.

| State | Transition | Guard/result |
|---|---|---|
| New save | Write artifacts with one generation; write `optimizer.pt` record and EMA record | Record includes generation, completed step, optimizer class, ordered `(name, shape, dtype)`, base recipe, and content hashes for formats without metadata (embedding files). |
| Optimizer built, unnamed/free trainable parameter exists | Before resume, EMA, training or save | `require_named` refuses at startup: an optimizer parameter owned by no module could be paired only by position. Current in-tree producers are named; an out-of-tree extension adding a free parameter is refused. |
| Versioned record contains positional entries | Resume | Refuse before `load_state_dict`; move `optimizer.pt` aside for fresh optimizer state or restore the recorded checkpoint. |
| Newest complete save, identity matches every artifact | Resume | Load optimizer state after validation; with EMA restore raw weights + shadow; continue. |
| Newest complete save, optimizer belongs to an earlier save | Discover previous complete match | Print fallback and resume there; if none match, refuse rather than pair mismatched moments. |
| New checkpoint written, optimizer still carries an older valid generation | Resume discovery | Find the last complete matching artifact set; refuse if no match. A failed optimizer write skips pruning so that fallback remains available. |
| Optimizer absent or unreadable | Resume | Non-RAWR can continue with fresh optimizer state (a read failure prints a warning); RAWR continuation requires its paired optimizer and refuses. This is not certified continuation. |
| Compatible LoRA rank conversion changed loaded weights | Resume | Non-RAWR explicitly starts fresh optimizer state; RAWR continuation refuses. |
| Recorded optimizer beside recorded but different checkpoint/parameter list/class | Resume | Refuse before `load_state_dict`. |
| Optimizer present, no checkpoint of this job resumed | Resume | Refuse; do not attach old moments to fresh weights. |
| Legacy record-less optimizer + legacy checkpoint | Resume | Load with an explicit "not certified/exact" warning; next save records identity. |
| Record-less optimizer + recorded checkpoint | Resume | Refuse. |
| EMA record present | Resume | Validate whole record (lengths, shapes, dtypes, update count, options) before mutating; a bad late entry restores nothing. Changed EMA options are printed and the new settings used. |
| Legacy EMA checkpoint | Resume | Continue from EMA weights with fresh EMA and print that this is not an exact continuation. |
| Changed base-model recipe | Resume | Warn only; this record stores recipe fields, not a digest of the base weights. Exactness is not claimed. |

Strict named-manifest refusal is implemented and reviewed. The naming fallback depends on attribute names staying the
same between save and resume; a renamed attribute holding a trained module refuses loudly. There is **no absolute
guarantee of matching tensors**: identity covers generation, optimizer class, ordered name/shape/dtype and content only
for metadata-less artifacts; the checkpoint base-model record stores recipe fields only (large local cache-model files use metadata identity), legacy state is uncertified, and scheduler or
accumulated-state exactness is not promised. Wan 2.2 full fine-tune split files remain undeclared.

#### 20.3.3 Cache preparation, admission and trusted dataset lifetime

**Invariant:** a latent-cache hit corresponds to current source bytes plus the effective VAE/preprocessing recipe; a
text-embedding hit corresponds to the current text encoder/prompt pipeline, current ordered reference content, and the
consumed self-media policy where applicable. **Old behaviour superseded:** keys covered names, geometry and similar
flags but not source bytes, encoder identity, ordered reference bytes, or self-media crop/flip policy; a readable legacy
file was treated as reusable.

| State | Transition | Guard/result |
|---|---|---|
| Dataset preparation | Build compact encoder fingerprints once per prepared dataset | Fingerprints are built only for enabled latent/text cache kinds; caching-off initialization makes zero builder calls. Source media/reference files are hashed at preparation/admission. |
| Latent cache key | Build/read | Key includes source SHA-256, schema version, preprocessing policy, model latent-space override and compact latent-encoder digest; a same-length rewrite with restored mtime changes the key on a new preparation. |
| Text cache key | Build/read | Key includes text-encoder identity, overrides, ordered control-image/video content identities, first-frame/control flags, target size where applicable, and self-media policy. |
| D-OPSD still image or i2v first frame | Self-media key | Includes the actual consumed pixel policy: scale, crop and flip plus dataset preprocessing policy. |
| D-OPSD video | Self-media key | Uses a raw-path reference policy with dataset temporal settings and the model's 2-fps video-reference presentation, not loader crop/flip flags. A random start-frame draw is temporal policy, not a content hash. |
| Reference bytes | Preparation/admission and explicit recalculation | Hashed through a per-item memo refreshed on `recalculate=True` or new admission, not on each training step. |
| Cache miss / schema mismatch / unreadable / legacy file | Re-encode | Latent/text reads compare stored keys; mismatch rebuilds lazily and atomically. Old files remain; no eager global invalidation. |
| Cache hit | Admit to prepared dataset | Item carries the preparation's encoder fingerprint and source/control digest; it is not rehashed per item/step. |
| Prepared dataset active | Source, control, or reference file replaced on disk | Not observed by already-admitted normal steps; a fresh preparation or explicit recalculation is required. This is a trust boundary, not a filesystem-watch guarantee. |
| `cache_text_embeddings` + `shuffle_tokens`/`random_triggers` | Text-cache admission | Refuse: per-access caption redraws cannot all be represented by one cached file. |
| Encoder/model identity changes | Next preparation | Fingerprint changes the key, so old compatible-format files are not admitted. No full-model weight hash is claimed. |

Model-path identity hashes small files (at most 1 MiB each, first 256 small files in a directory) and records local
metadata for large weights; it is not a content hash of multi-GB checkpoints. Active prepared input immutability is a
documented trust boundary: bytes are snapshotted at admission and memoized, not watched or rehashed on every step.

#### 20.3.4 Held-out validation overlap refusal

**Invariant:** no file admitted as validation data is also admitted as training data. **Old behaviour superseded:** setup
could compare only resolved folder strings and miss aliases/nesting.

| State | Transition | Result |
|---|---|---|
| Validation disabled or no validation datasets | Setup | Return without a new check. |
| Validation configured | `toolkit_validation.setup` starts | `_check_validation_disjoint` runs before validation dataset admission. |
| Equal or nested training/validation roots | Check | Conservative `ValueError`; put validation outside the training tree. |
| Disjoint roots but an actual admitted validation file equals an admitted training path | Check | Refuse before training-validation use. |
| Hardlink under a different path | Check | `_file_identity` uses only a nonzero `(dev, ino)` identity, with canonical path equality kept separate, so real hardlinks under disjoint roots are detected. |
| Directory junction or canonical alias | Check | Resolved canonical overlap is detected; true prefix siblings such as `data` and `data2` are admitted. |
| Nested `_controls` or hidden validation root | Check | Conservatively refused rather than relying on loader skipping rules. |
| Disjoint data | Validation cache/setup | Continue to frozen-text-encoder/plain-conditioning checks and validation loading. |

This is a refusal boundary, not proof that every file in a complex layout leaked, and not a substitute for exact
filesystem-walk or concurrent-swap protection. File-symlink creation was unavailable in the Windows worker environment
(`WinError 1314`); canonical path handling is implemented, and a real directory-junction alias test passes.

#### 20.3.5 Inference loaded-file generation

**Invariant:** a reused loaded LoRA stack reflects the file currently at the requested path. **Old behaviour
superseded:** path-only keys allowed an in-place strength update or merge cleanup to keep old weights after the file
was replaced.

| State | Transition | Guard/result |
|---|---|---|
| Desired spec unloaded | Engine asks key | Reads current local file generation (metadata) into the key. |
| Loading | Capture before/after generation | If the file changed during `load_file`, refuse to bind an unverified snapshot. |
| Loaded | Bind `_loaded_file_generation` | Stack key uses the bound generation; merge-cleanup tag stays stable for the exact prior set. |
| Loaded, file path replaced | Engine computes desired key | Key differs -> evict/reload affected components. |
| Hook mode, same paths | In-place `set_strengths` requested | Return false if current generation differs from bound generation; engine detaches the old hook generation and reloads. |
| Merge mode | Remove requested | Reload-based removal; does not unmerge weights. |
| Same size/mtime/inode rewritten in place | Any check | Undetectable: generation is metadata-only, not a content hash. No absolute matching-tensor guarantee is made. |

#### 20.3.6 Optimizer dispatch

**Invariant:** the configured optimizer name selects the named algorithm. `dadaptationadam` now actually selects
`DAdaptAdam`; `dadaptationlion` selects `DAdaptLion`; the bare legacy `dadaptation` alias selects `DAdaptAdam`; `lion`
remains Lion and bare `adam` remains `torch.optim.Adam`. The focused test uses a stub optional module and does not run
actual DAdapt optimizer training. Legacy Lion optimizer state produced by the old `dadaptationadam` path is **not
certified** for the corrected algorithm; use a fresh optimizer if using affected legacy saves.

### 20.4 Compatibility and performance boundaries

- Cache source/reference hashing happens during preparation/admission, not on the normal training-step path. Encoder
  fingerprints are built once per preparation and only for enabled cache kinds; caching-off initialization makes zero
  builder calls. No GPU throughput measurement was made, and none should be inferred.
- Active prepared input immutability is not enforced: media/reference bytes are admission-time snapshots memoized on
  the prepared DTO. A file changed after preparation is not re-hashed during a normal training step.
- Cache schema v2 is lazy: incompatible/legacy files are misses and are rebuilt atomically; old files are retained
  rather than eagerly deleted. Resume refuses generation/identity mismatches instead of trying a lossy repair.
- Large model weight files are metadata-only (small-file content hashing is bounded as above); inference LoRA
  generations are metadata-only. There is no full-model content hash and no absolute guarantee that two same-metadata
  tensors are identical.
- EMA resume adds CPU copies of raw weights and the EMA shadow to `optimizer.pt`, so save/load RAM, disk and I/O grow
  with trainable parameter count beyond the optimizer state itself. This is an untested overhead policy, not a
  benchmark.
- RAWR pair mode with a split LoRA layout and distributed multi-process training are explicitly unsupported and refuse
  at startup; non-pair split layouts still work. Wan 2.2 full fine-tune split remains undeclared.
- Root/ancestor junction spellings are canonicalized, but concurrent hostile filesystem replacement after validation
  remains outside the guarantee.
- The exact-root lease does not serialize arbitrary nested CLI roots, unrelated programs, or network filesystems
  without working SQLite locks.
- Unknown/free optimizer parameters owned by no module are refused. Legacy checkpoint/optimizer/EMA state remains
  uncertified, and legacy DAdapt-Lion state from the old dispatch is not certified for corrected DAdaptAdam.
- GRAD-01 remains an unproven hypothesis; no per-step gradient scan was added.

### 20.5 Evidence and remaining limits

- Core evidence is the committed `964d79a1` focused CPU run: 510 passed / 10 skipped across 27 files, including a real
  tiny CPU SDTrainer resume with bit-identical uninterrupted versus resumed output. No final combined CPU suite, GPU,
  offload, compile, full-model, or full-training validation is claimed.
- UI evidence covers commits `bb21cae1` and `964d79a1`: 101 UI passed and TypeScript 0 diagnostics. These are CPU/TypeScript fixtures
  and mocked/real route helpers, not a live HTTP/browser deployment.
- Data/cache/validation/inference/dispatch is committed as `cb692933` and accepted in the combined CPU suite above. The focused worker result is 123 passed/1 skipped (Windows file-symlink privilege); real hardlink/junction tests pass. CAP-01 is source/test-confirmed within its exact options. DAdapt dispatch uses a stub module, without real DAdapt optimizer training.
- Original-code behavioural evidence is bounded to the detached `db8` source snapshot and harness additions. Tests
  that only fail because a new helper is absent are absence-only, not behavioural proof.
- Legacy checkpoint/optimizer/EMA state is not certified. The checkpoint base-model recipe warns rather than enforcing weight identity; large local cache-model files use metadata identity; inference generation is metadata-only. Active prepared input immutability is not guaranteed.
- Wan 2.2 full fine-tune split files remain undeclared; exact-root leases exclude arbitrary nested roots and
  non-cooperating programs; no full GPU validation was performed.
- No universal "every invariant found" or exhaustive-proof claim is made. This addendum records the 24 named
  acceptance-matrix issues and the named boundaries only; the historical sections retain their own scope and gaps.

### 20.6 Final integration acceptance

Codex independently ran the complete `tests/` suite after all Python writers finished: **863 passed, 53 skipped**. The source hash manifest was unchanged through the run and rechecked before committing. Full UI **101/101**, TypeScript **0/0 diagnostics**. Core commit `964d79a1`, UI `bb21cae1`, data `cb692933`; documentation is committed separately. No merge/push or GPU test was performed. The data worker's 123/1 focused result and Opus's 510/10 focused result are supporting subsets, not counts to add to the full suite. One new Windows file-symlink case is skipped for missing privilege; real hardlink and directory-junction cases execute. Earlier overlapping/duplicate-log runs are excluded from acceptance; exact failures and replacements are in the implementation run's failure ledger.
