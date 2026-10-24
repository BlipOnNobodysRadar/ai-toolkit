# Independent review regressions

`review_tests/test_review_contracts.py` is an unchanged copy of the independent
Astra review at c84fbfd (source task-7 on Blindows). Its SHA256 is
`d665c53094f7fa2c2aee5dfb76ffebf2316ed3c0d745470677d12dade401e87f`.
All four assertions failed before these repairs. The tests do not relax the
requirement for parameter-exact supported resume, including learnable SNR.

The two review drivers retain the review's configuration variations and loop
comparisons. Plumbing now points at this checkout and existing local fixtures
through `AITK_RAWR_CPU_FIXTURES`. Added observations compare all CPU RNG states,
elementwise pending gradient tensors, and SNR parameters/optimizer/history.
They never change production updates, reseed per step, download models, or use
CUDA. Evidence is ignored by Git and remains local.

With the existing environment and `AITK_RAWR_CPU_FIXTURES` pointing at a directory
containing `template.yaml`, `tiny-sd/`, and `tiny-data/`, run serially:

```powershell
python -B testing/rawr_review/review_variants.py fp16-export
python -B testing/rawr_review/review_variants.py learnable-snr
```

Both drivers reject an already-used control/resume folder. Supply a fresh
`--evidence-root` for reruns; the unchanged assertions read the default
`testing/rawr_review/evidence/variants` location. Run the assertions and
`tests/test_rawr_review_fixes.py` with CPU environment variables set before
framework imports and CUDA initialization blocked, as with the other RAWR CPU
tests. Without local variant evidence, the evidence-dependent assertions skip.

The standard integration suite additionally covers an abrupt pre-manifest exit
with both learnable SNR and the default FP16 export, then corrupts only the owned
fixture's legacy scalar JSON. Recovery must use the previous committed pair and
match the control's complete snapshots before and after every resumed update.
