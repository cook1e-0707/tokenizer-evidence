# WP5-R2 Train-and-Score Review: Job 851481

Review time: 2026-05-09T16:58:00Z

## Scope

This review reconciles Hermes state after WP5 job `851373` failed and the
WP5-R2 margin-lambda retry job `851481` completed.

This is still teacher-forced target-bucket mass evidence only. It is not
free-generation E2E, not payload recovery, not FAR, not Llama, and not a
paper-facing positive claim.

## Slurm Status

```text
job_id = 851481
job_name = nat-ev-v2-wp5r2
state = COMPLETED
elapsed = 00:17:47
exit_code = 0:0
```

Local artifacts:

```text
results/natural_evidence_v2/status/wp5_r2_teacher_forced_train_and_score_851481/
```

Key files:

```text
protected_train/wp5_micro_slot_lora_train_summary.json
task_only_train/wp5_micro_slot_lora_train_summary.json
teacher_forced_score/wp5_teacher_forced_bucket_mass_summary.json
```

## Teacher-Forced Score Summary

Rows:

```text
base rows = 8192
protected rows = 8192
task_only rows = 8192
total rows = 24576
```

Condition means:

| Condition | mean_target_mass | mean_target_margin | median_target_margin | rank1_rate |
|---|---:|---:|---:|---:|
| base | 0.18369929203450752 | -0.057893098408331234 | 0.0012832073516619857 | 0.521484375 |
| protected | 0.735353565186017 | 0.688442271253743 | 0.7632847931236029 | 0.9820556640625 |
| task_only | 0.19897803151220247 | -0.08308507613994176 | 0.0000037924229587815717 | 0.513671875 |

Observed lifts:

```text
protected_target_bucket_mass_lift_vs_base = 0.5516542731515095
protected_target_bucket_mass_lift_vs_task_only = 0.5363755336738145
task_only_target_bucket_mass_lift_vs_base = 0.015278739477694953
```

Configured WP5 gate:

```text
protected_target_bucket_mass_lift_vs_base >= +0.15
protected_target_bucket_mass_lift_vs_task_only >= +0.10
protected_target_bucket_rank1_rate >= 0.70
protected_median_target_margin > 0
task_only_target_bucket_mass_lift_vs_base < +0.05 diagnostic cap
```

Result:

```text
teacher_forced_gate_status = PASS
teacher_forced_gate_pass = true
```

## Reconciled State

`851373` remains a valid failed WP5 diagnostic. `851481` supersedes it as the
current WP5-R2 teacher-forced gate result.

However, WP6/free-generation E2E is still blocked by the latest Hermes conflict
reports because the controlling tick prompt explicitly forbids generation and
Qwen E2E reruns, and the WP6 generator/decoder/wrapper path is not locally
implemented/reviewed.

Latest conflict blocker:

```text
results/natural_evidence_v1/status/hermes_reports/20260509_1645_wp6_e2e_generation_conflict_blocker.md
```

The inconsistent `v2_wp6_e2e_eval` allowlist entry has been disabled until a
later controlling action explicitly removes the `no generation` and
`no Qwen E2E rerun` hard constraints and the WP6 implementation is reviewed.

## Decision

```text
WP5_R2_TEACHER_FORCED_GATE_PASS__WP6_E2E_BLOCKED_BY_GENERATION_CONSTRAINT
```

Next safe state:

```text
current_phase = V2_WP5_R2_GATE_PASSED_WP6_E2E_BLOCKED_BY_GENERATION_CONSTRAINT
next_allowed_action = artifact-only WP6 implementation review only, or explicit later WP6 Slurm permission after generation/E2E constraints are removed
```

Still forbidden:

- no new training job;
- no generation;
- no Qwen E2E submission;
- no Llama;
- no same-family null;
- no sanitizer benchmark;
- no FAR aggregation;
- no paper positive claim.
