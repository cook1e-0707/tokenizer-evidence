# Repaired target-mass probe submission blocker

Checked at: `2026-05-08T07:13:43Z`

## Status

`BLOCKED_REPAIRED_TARGET_MASS_PROBE_SUBMISSION_NOT_UNAMBIGUOUS`

Hermes notification succeeded before this Codex worker started:

- Telegram: sent
- Email: sent
- Notification JSON:
  `results/natural_evidence_v1/status/hermes_reports/20260508_0711_scheduled_tick_notification.json`

Model scoring is still needed because the repaired teacher-forced target-mass
probe design is complete but not scored:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_design/
```

## Blocker

The next allowed action requires exactly one Slurm-scored repaired
teacher-forced target-mass probe that consumes the Option R scoring plan:

```text
results/natural_evidence_v1/status/qwen_natural_e2e_eval_846699_repaired_teacher_forced_target_mass_probe_design/repaired_teacher_forced_target_mass_probe_scoring_plan.jsonl
```

No dedicated repaired target-mass scoring wrapper or scorer exists in the
current repo state. The only teacher-forced Slurm wrapper found is:

```text
scripts/natural_evidence_v1/slurm/qwen_teacher_forced_bucket_mass_probe.sbatch
```

That wrapper calls the older committed-prefix probe,
`scripts/natural_evidence_v1/probe_qwen_teacher_forced_bucket_mass.py`, and
does not consume the repaired scoring plan. Submitting it would score the wrong
artifact path, not the Option R repaired plan.

The run allowlist currently enables only:

```text
qwen_846699_teacher_forced_bucket_mass_probe
```

with command:

```text
sbatch scripts/natural_evidence_v1/slurm/qwen_teacher_forced_bucket_mass_probe.sbatch
```

Submitting an unlisted GPU wrapper would violate
`forbid_unlisted_gpu_jobs: true`. Modifying the allowlist and implementing a
new scorer/wrapper is a separate state-changing action and was not combined
with this tick.

## Actions Not Taken

- no Slurm job submitted
- no model scoring started
- no training
- no generation
- no Qwen E2E rerun
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no paper-facing positive claim

## Next Safe Action

In a later supervised tick, implement and allowlist a dedicated repaired
teacher-forced target-mass scorer/wrapper that consumes the Option R scoring
plan, validates non-overwrite output paths, and then submit exactly one Slurm
job. Do not submit the older committed-prefix teacher-forced wrapper for this
gate.
