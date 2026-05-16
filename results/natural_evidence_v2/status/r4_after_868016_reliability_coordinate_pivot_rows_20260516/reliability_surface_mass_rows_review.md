# R4 After 867621 Reliability Surface-Mass Rows

Status: `PASS_R4_AFTER_867621_RELIABILITY_SURFACE_MASS_ROWS_BUILT_ARTIFACT_ONLY`

This artifact builds teacher-forced score rows for the coordinate-unique
reliability surface bank after job `867621` failed with no selected surface
matches in free generation.

```text
rows: 3072
prompt offset: 256
selected prompts: 256
selected coordinates: 12
selected coordinate ids: [6, 22, 26, 1, 17, 19, 15, 31, 8, 4, 7, 23]
excluded coordinate ids: [3, 10, 20, 24]
surface entries: 128
surface bank sha256: 4a0f07af15ade41d51655352976e18d17e095b60a4850814ade231ec9f5fe1ac
codebook sha256: aa277c813bbd58b893aa8e75fa1e3132f4cd3cb4cd2a742fbee4ec49356214cc
expected codeword bits: [1, 0, 1, 0, 0, 1, 0, 1]
```

No tokenizer, model, Slurm, training, generation, Llama, sanitizer, FAR, payload
diversity, or paper-facing claim action was started.

Next allowed action: actual Qwen tokenizer-boundary preflight route preparation
for these rows. Do not submit scoring or generation until tokenizer boundary
passes and a reviewed scoring route is recorded.
