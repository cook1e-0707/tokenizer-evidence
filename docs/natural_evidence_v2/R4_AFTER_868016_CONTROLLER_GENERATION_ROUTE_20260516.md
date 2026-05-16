# R4 After-868016 Controller-Aware Generation Route

Status: `ARTIFACT_ONLY_ROUTE_REVIEW`

This route follows the reviewed job `868114` teacher-forced controller pass. It
does not treat `868114` as a natural-output result. It prepares the first
generation/decode wrapper for the coordinate-filtered controller path.

## Inputs

```text
tokenizer preflight: 868103 PASS
teacher-forced controller scoring: 868114 PASS
score rows: results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_rows_20260516/reliability_surface_mass_rows.jsonl
filtered codebook: results/natural_evidence_v2/precommit/r4_after_868016_reliability_coordinate_pivot_codebook_precommit_20260516/
```

The filtered codebook removes the four post-868016 weak coordinates from the
generation/decode contract:

```text
excluded coordinates: 3,10,20,24
selected coordinates: 6,22,26,1,17,19,15,31,8,4,7,23
```

## Generation Scope

This is a small Qwen-only row-cylinder diagnostic:

```text
generation unit: prefix_native_row_cylinder
conditions: protected, raw, task_only
protected mechanism: base Qwen + first-step committed soft controller
raw: base Qwen, no controller
task_only: task-only adapter, no controller
blocks: 4
prompts per block: 64
row cylinders per shard: 768
format-scrub primary decode: all
```

The protected generation condition uses the best reviewed `868114` controller
grid:

```text
bonus_nats: 4.0
penalty_nats: 0.5
max_target_mass: 0.5
max_kl_budget: 0.5
```

Wrong-key and wrong-payload remain decoder controls over protected transcripts.

## Claim Control

This route may only produce a small diagnostic. It does not unlock training,
Llama, same-family null, sanitizer, FAR, payload diversity, or paper-facing
positive claims.

The route must pass local and remote preflight before any Slurm submission:

```text
route validator PASS
wrapper plan-only PASS
allowlist zero-enabled PASS
local/remote file hashes match
exactly one allowlist entry enabled for submission
allowlist disabled immediately after sbatch
```

If submitted and completed, the review gate is:

```text
protected accepts with format_scrub=all >= 3/4
raw accepts = 0/4
task_only accepts = 0/4
wrong_key accepts = 0/4
wrong_payload accepts = 0/4
forbidden public surface count = 0
```

This gate is intentionally diagnostic-scale; passing it would only justify a
larger locked dev generation route, not paper-facing claims.
