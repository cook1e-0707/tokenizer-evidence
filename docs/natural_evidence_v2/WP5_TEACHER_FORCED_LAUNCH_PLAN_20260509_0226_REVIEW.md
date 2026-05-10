# WP5 Teacher-Forced Launch Plan Review

Review time: 2026-05-09T06:38:00Z

## Scope

This review covers the natural_evidence_v2 WP5 teacher-forced target-mass
launch plan:

```text
results/natural_evidence_v2/status/wp5_teacher_forced_launch_plan_20260509_0226/
```

This is a Qwen teacher-forced training/scoring gate only. It is not Qwen E2E,
not payload recovery, not FAR, not Llama, and not a paper-facing positive
claim.

## Inputs

The launch plan is built from existing reviewed artifacts:

```text
results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_audit_850885/
results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl
results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/
```

Primary bank:

```text
bucket_0 = Set | Plan
bucket_1 = Create | Prepare
min_bucket_mass = 0.06311000335572636
combined_bank_mass = 0.14328484169406375
mass_ratio = 1.2703982582035882
```

WP4 payload bits used for the prompt-local 16-slot target:

```text
payload+checksum bits = [1,0,1,0,0,1,0,1,0,1,0,1,1,1,1,0]
```

## Generated Plan Artifacts

```text
wp5_protected_training_rows.jsonl        512 rows
wp5_task_only_training_rows.jsonl        512 rows
wp5_teacher_forced_score_rows.jsonl      8192 rows
wp5_training_examples_preview.jsonl      16 rows
wp5_teacher_forced_launch_plan_summary.json
```

The protected rows use a local suffix repair policy:

```text
surface_specific_bridge_to_original_action_phrase_v1
```

This masks the exact target surface CE at evidence slots and trains the
remaining response text with task CE. The protected arm adds bucket margin loss
at micro-slots; the task-only arm has no bucket loss.

## Launch Gate

The 0226 summary reports:

```text
launch_gate_status = PASS_READY_TO_SUBMIT_ONE_ALLOWLISTED_WP5_SLURM_JOB
launch_blockers = []
trainer_exists = true
scorer_exists = true
slurm_wrapper_exists = true
allowlist_entry_exists = true
repair_warning_counts = {}
```

Implemented/reviewed execution pieces:

```text
scripts/natural_evidence_v2/build_wp5_teacher_forced_launch_plan.py
scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py
scripts/natural_evidence_v2/score_wp5_teacher_forced_bucket_mass.py
scripts/natural_evidence_v2/slurm/wp5_teacher_forced_train_and_score.sbatch
configs/natural_evidence_v2/run_allowlist.yaml
```

Validation:

```text
pytest tests/test_natural_evidence_v2_wp5_launch_plan.py \
  tests/test_natural_evidence_v2_wp4_contract.py \
  tests/test_natural_evidence_v2_restricted_density.py

10 passed
```

Local wrapper dry-run passed:

```text
results/natural_evidence_v2/status/wp5_wrapper_dry_run_20260509_0228/
```

## Submitted Job

Exactly one allowlisted Slurm job was submitted:

```text
job_id = 851373
job_name = nat-ev-v2-wp5tf
partition = DGXA100
node = chimera12
command = sbatch scripts/natural_evidence_v2/slurm/wp5_teacher_forced_train_and_score.sbatch
```

Initial Slurm status:

```text
RUNNING
elapsed = 00:00:10
exit_code = 0:0
```

## Review Decision

Decision:

```text
WP5_TEACHER_FORCED_TRAIN_AND_SCORE_JOB_SUBMITTED
```

Next allowed action is monitoring job `851373`. After completion, sync and
review:

```text
protected_train/wp5_micro_slot_lora_train_summary.json
task_only_train/wp5_micro_slot_lora_train_summary.json
teacher_forced_score/wp5_teacher_forced_bucket_mass_summary.json
teacher_forced_score/wp5_teacher_forced_bucket_mass_rows.jsonl
```

Do not submit another WP5 training job, Qwen E2E, Llama, same-family null,
sanitizer benchmark, FAR aggregation, or paper-facing positive claim until this
teacher-forced gate is reviewed.
