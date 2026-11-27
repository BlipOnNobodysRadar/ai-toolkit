# MOSS step distillation

Use **Step distillation** in the AI Toolkit sidebar. This is an experimental
text-to-image LoRA trainer for MOSS-Image SFT; it is not a Wan/video trainer,
an 8-step adapter retune, or a reproduction of online DMD.

1. Use the original SFT model or a **BF16 checkpoint with your content adapters
   already merged**. Use the same base at inference; do not apply baked-in
   adapters twice. The initial implementation does not train on a quantized DiT.
2. Paste or import a UTF-8 list of prompts, one per line. Set seeds per prompt,
   image dimensions, teacher CFG, step counts and student sampler/schedule.
   Teacher: Euler, default native shift-5 MOSS grid, layer 22 skipped on the
   negative CFG pass. Paste the actual teacher sigma list if matching a specific
   Comfy workflow. The student uses one conditional prediction per Euler step,
   no CFG or negative prompt; Heun uses two predictions except on the last step.
3. **No review:** leave the review box off, save the job, then start it from
   its overview. One queue slot renders the teacher cache and trains the student.
   There are no PNGs, ratings, preference updates, or review pause in this path.
4. **With review:** turn the box on and use two or more seeds per prompt. Save
   and start the teacher job. When complete it exits and releases the GPU.
   Return to Step distillation, refresh finished teachers, and select it.
   Compare A/B within each prompt. Ties add no directional label; Skip casts
   no vote. Exclude obvious defects explicitly from regression if desired.
   Browser-local review progress is scoped to this job and cache. Saving a
   separate student job copies votes and exclusions into its ordinary job config.
5. Leave DPO off for curated regression alone, or enable it to interleave a
   flow-DPO preference loss with trajectory updates. Rank is calculated separately
   in each connected set of comparisons (Elo, initial 1000, K=32). Pair the
   highest with lowest, then next highest with next lowest; no unvoted, equal-score
   or cross-prompt pairs, no sample reuse within that export. Excluded targets
   can still be DPO losers. An entirely excluded set or DPO without a usable
   pair is an error, not silent fallback. This uses **hard** preferences; Elo
   is a pairing rule, not an estimate of calibrated preference probability.

## What is actually trained

The frozen teacher produces one trajectory per prompt/seed, using FP32 noise
and solver arithmetic. Cache its latent at each student sigma: inside a teacher
Euler interval use that segment's linear interpolation, without changing the
teacher's final image. Each optimizer update samples a trajectory and an interval,
starts from its cached teacher latent, and runs the configured student solver.
Minimize the squared error of the resulting vs target displacement divided by
interval width (velocity MSE). This is **teacher-forced trajectory distillation**:
it never sees its own long-run drift. Judge full student rollouts on held-out
prompts and seeds, not just training loss.

Only low-rank updates on the last 18 MOSS blocks' projection linears train.
The backbone is shared and frozen, with block gradient checkpointing. Optional
DPO uses two clean teacher endpoints, the same fresh noise and timestep for both,
the fork's flow preference logit (beta 100), and the frozen personalized backbone
with the speed adapter disabled as reference. It is an auxiliary loss on the same
adapter, not a second model. The review UI is built in; no blind-ranker installation
or external service is required.

## Files, resume and evaluation

- Teacher folder: `teacher-cache/` contains prompt embeddings, per-candidate
  FP32 latent trajectories, optional PNGs, and the completed cache manifest.
  Expect ~42 MiB per 1024-square, 20-step candidate (128 packed channels).
  256 candidates at that size need ~10.5 GiB before PNGs. Keep the teacher
  folder in place for subsequent students.
- Student folder: `moss-speed.safetensors`, `target-sampler.json`,
  `distill-state.pt` (optimizer + adapter), `student-recipe.json` and
  `preference-pairs.json`. Reviewed pairs are also exported to
  `preferences-export/{chosen,rejected}` as same-stem PNG/caption pairs
  for the fork's existing offline DPO trainer. The same pair labels need
  not be applied twice.
- Restart the same job to reuse completed teacher candidates and the last
  atomic student checkpoint (default every 50 updates and at completion).
  Up to one save interval of training may repeat. Start a new student job
  if you change the votes, exclusions, model, rank, loss weight or learning
  rate. GPU selection uses the existing queue; no GPU starts on form save.
- Inference: use the same base + LoRA strength 1, CFG 1, solver from
  `target-sampler.json`, and its exact sigmas. That file includes the clean
  terminal zero; omit the zero in the Toolkit preview's sample-item `sigmas`
  field. Set `model.model_kwargs.sample_solver: heun` for a Heun student
  (default Euler). Don't assume a similarly named Comfy schedule is identical.
- Start small: eight prompts × two seeds, 512², 50 → 20 Euler, 50–100 updates.
  Confirm generation, voting, export and restart before a quality run. Compare
  with non-distilled 20-step and guided 50-step A/B on the 3060, not just on the
  training card. No measured quality, 4090 peak VRAM or training-time claim.
