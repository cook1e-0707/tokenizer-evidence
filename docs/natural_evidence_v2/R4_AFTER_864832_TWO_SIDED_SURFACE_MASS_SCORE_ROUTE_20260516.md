# R4 After 864832 Two-Sided Surface-Mass Score Route

Canonical phase:
`V2_R4_AFTER_864832_TWO_SIDED_SURFACE_MASS_SCORE_ROUTE_VALIDATED_NO_SUBMIT`

## Decision

The two-sided rows passed actual Qwen tokenizer-boundary validation in job
`865210`. The next compute action is teacher-forced surface-mass scoring only.

## Route

```text
allowlist entry: v2_r4_after_864832_two_sided_surface_mass_score_h200
wrapper: scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_surface_mass_score_h200.sbatch
rows: results/natural_evidence_v2/status/r4_after_864832_two_sided_cover_bank_rows_20260516/cover_bank_aligned_target_only_rows.jsonl
arms: base, protected, task_only
max rows: 8192
```

The route uses the existing protected/task-only adapters from job `864761` and
does not train or generate.

## Gate

```text
protected lift vs base >= +0.15
protected lift vs task-only >= +0.10
protected rank1 >= 0.75
protected median margin > 0
boundary failures = 0
target/other overlap = 0
```

## Not Unlocked

This route does not unlock generation, training, Qwen E2E, Llama, same-family
null, sanitizer, FAR, payload diversity, or paper-facing claims.
