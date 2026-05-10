# WP5 Teacher-Forced Launch Plan Review

## Scope

This is an artifact-only review of the natural_evidence_v2 WP5
teacher-forced target-mass training/scoring launch plan in:

```text
results/natural_evidence_v2/status/wp5_teacher_forced_launch_plan_20260509_0225/
```

It does not start training, model scoring, generation, Qwen E2E, Llama,
same-family nulls, sanitizer benchmarks, FAR aggregation, or paper-facing
positive claims.

## Inputs

The plan uses the reviewed v2 route artifacts:

```text
primary_bank = Set|Plan vs Create|Prepare
bucket_bank_id = qwen_v2_wp3_r2_primary_set_plan_vs_create_prepare_v1
wp4_contract_sha256 = 69d1feb2b63f52db7cf1ca82bb9ccfcbeb056f2f4f5945b230fc8c44923ada07
selected_split = wp3_r1_dev
selected_source_response_count = 512
payload_plus_checksum_bits = 16
```

The protected plan rows rewrite complete Step-label responses so the first
action word at each of the 16 micro-slots targets the WP4 payload/checksum
bucket. The task-only rows preserve the same source responses without enabling
bucket loss. The score plan contains one teacher-forced slot row for each
planned micro-slot.

## Artifacts

```text
wp5_protected_training_rows.jsonl = 512 rows
wp5_task_only_training_rows.jsonl = 512 rows
wp5_teacher_forced_score_rows.jsonl = 8192 rows
wp5_training_examples_preview.jsonl = 16 rows
wp5_teacher_forced_launch_plan_summary.json
```

The summary reports:

```text
launch_gate_status = FAIL_NOT_READY_TO_TRAIN
training_started = false
model_scoring_started = false
model_generation_started = false
e2e_eval_started = false
paper_claim_allowed = false
not_payload_recovery = true
not_full_far = true
```

## Launch Gate

The WP5 pre-training launch gate is not satisfied. The recorded blockers are:

```text
missing_v2_margin_trainer
missing_v2_teacher_forced_scorer
missing_v2_wp5_slurm_wrapper
missing_enabled_allowlist_entry
```

Because the gate failed, no allowlisted Qwen WP5 training job may be submitted
from this review. The current tick also explicitly forbids training.

## Teacher-Forced Gate Targets

After a future reviewed training/scoring implementation exists, the
post-training teacher-forced gate remains:

```text
protected target bucket mass - base >= +0.15
protected target bucket mass - task-only >= +0.10
target bucket rank-1 rate >= 0.70
median target margin > 0
task-only target bucket mass - base must not be materially positive
```

Qwen E2E remains forbidden unless that future teacher-forced gate passes.

## Decision

The WP5 training/scoring plan artifact exists and fixes the protected/task-only
row sets and teacher-forced score rows, but it is not a launch approval. The
next allowed project-advancing step is to implement and review the missing v2
WP5 margin trainer, teacher-forced scorer, Slurm wrapper, and allowlist entry
without launching training until the full gate is satisfied under the active
hard constraints.
