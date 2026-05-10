# WP5 Teacher-Forced Train-and-Score Review: Job 851373

Review time: 2026-05-09T16:55:00Z

## Scope

This review reconciles Hermes state after Slurm job `851373` completed.
It covers only the natural_evidence_v2 WP5 teacher-forced Qwen protected/task-only
LoRA training and bucket-mass scoring gate.

This is not free-generation E2E, not payload recovery, not FAR, not Llama, and
not a paper-facing positive claim.

## Slurm Status

```text
job_id = 851373
job_name = nat-ev-v2-wp5tf
state = COMPLETED
elapsed = 00:16:27
exit_code = 0:0
```

Synced local artifacts:

```text
results/natural_evidence_v2/status/wp5_teacher_forced_train_and_score_851373/
```

Key files:

```text
protected_train/wp5_micro_slot_lora_train_summary.json
task_only_train/wp5_micro_slot_lora_train_summary.json
teacher_forced_score/wp5_teacher_forced_bucket_mass_summary.json
teacher_forced_score/wp5_teacher_forced_bucket_mass_rows.jsonl
```

## Training Sanity

Protected training did locate margin positions:

```text
protected final_margin_loss = 0.08222173154354095
protected final_loss = 1.1151220798492432
protected metrics_tail margin_count = 16.0
```

Task-only training had no bucket margin, as intended:

```text
task_only final_margin_loss = 0.0
task_only final_loss = 0.25307193398475647
task_only metrics_tail margin_count = 0.0
```

This means the run is not invalid because of missing protected margin-position
localization. The failure is in the measured teacher-forced bucket-mass effect.

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
| protected | 0.08716496688584074 | 0.009017284726605457 | 0.0034794392995536327 | 0.535888671875 |
| task_only | 0.21478994174473212 | -0.07346284649315613 | 0.000005882290452063899 | 0.5069580078125 |

Observed lifts:

```text
protected_target_bucket_mass_lift_vs_base = -0.09653432514866678
protected_target_bucket_mass_lift_vs_task_only = -0.12762497485889138
task_only_target_bucket_mass_lift_vs_base = 0.0310906497102246
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
teacher_forced_gate_status = FAIL
teacher_forced_gate_pass = false
```

## Conflict Reconciliation

Hermes produced repeated `wp6_e2e_generation_conflict_blocker` reports at
16:00, 16:15, 16:30, and 16:45 UTC. The latest blocker is:

```text
results/natural_evidence_v1/status/hermes_reports/20260509_1645_wp6_e2e_generation_conflict_blocker.md
```

The blocker is correct. WP6/E2E must not start because:

- WP5 teacher-forced gate failed on job `851373`;
- generation and Qwen E2E are still forbidden in the controlling status files;
- the WP6 wrapper/generator/decoder scripts are not present locally;
- the enabled `v2_wp6_e2e_eval` allowlist entry was inconsistent with the above.

The allowlist has therefore been reconciled:

```text
v2_wp6_e2e_eval.enabled = false
enable_condition = blocked_20260509_1645_wp6_e2e_generation_conflict_and_wp5_teacher_forced_gate_failed_on_851373
```

## Decision

```text
WP5_TEACHER_FORCED_GATE_FAIL_NO_WP6
```

This result means Qwen v2 controlled micro-slot WP5 did not yet demonstrate the
required teacher-forced protected target-bucket mass lift. It does not prove
natural evidence impossible, but it blocks WP6 proof-of-life E2E.

## Next Safe State

```text
current_phase = V2_WP5_TEACHER_FORCED_GATE_851373_FAIL_WP6_BLOCKED
next_allowed_action = artifact-only WP5 failure diagnosis / objective repair planning
```

Still forbidden:

- no second WP5 training job without a new reviewed repair plan;
- no Qwen E2E;
- no generation;
- no Llama;
- no same-family null;
- no sanitizer benchmark;
- no FAR aggregation;
- no paper positive claim.
