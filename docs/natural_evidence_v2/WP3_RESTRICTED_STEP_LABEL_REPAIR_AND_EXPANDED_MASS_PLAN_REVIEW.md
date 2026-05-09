# WP3 Restricted Step-Label Repair And Expanded Mass Plan Review

## Scope

This review covers the artifact-only repair work after Slurm job `850434`.
It does not approve WP4, training, Qwen E2E, Llama, same-family nulls,
sanitizer benchmarking, FAR aggregation, or paper-facing positive claims.

## Repair Plan Review

Repair plan artifacts:

```text
scripts/natural_evidence_v2/build_wp3_restricted_step_label_repair_plan.py
results/natural_evidence_v2/status/wp3_restricted_step_label_repair_plan_20260508_2134/
```

The plan is approved as an artifact-only planning step. It addresses the two
observed 850434 blockers:

- exact Step-label adherence: writes `256` stricter prompts that require exactly
  sixteen lines beginning with literal `Step 1:` through `Step 16:`;
- narrow bank coverage: writes `8` expanded Step-label action-verb candidate
  banks derived from frequent base-Qwen openers in 850434.

This is not a WP3 pass. It only prepares inputs for the next scoring audit.

## Expanded Mass Score Plan

Score-plan artifacts:

```text
scripts/natural_evidence_v2/build_wp3_restricted_step_label_expanded_mass_plan.py
results/natural_evidence_v2/status/wp3_restricted_step_label_expanded_mass_plan_20260508_2148/
```

The generated score plan has:

```text
score_plan_rows=128
expanded_bank_candidate_count=8
prefixes=Step 1: ... Step 16:
casing_variant=sentence_case
compatible_wrapper=scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
```

Rows are compatible with `scripts/natural_evidence_v2/score_wp3_context_mass.py`
and passed local virtual-environment validation:

```text
.venv/bin/python scripts/natural_evidence_v2/score_wp3_context_mass.py \
  --score-plan results/natural_evidence_v2/status/wp3_restricted_step_label_expanded_mass_plan_20260508_2148/qwen_v2_wp3_restricted_step_label_expanded_context_mass_score_plan.jsonl \
  --validate-plan-only

status=PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION
score_plan_rows=128
```

The Slurm wrapper syntax also passed:

```text
bash -n scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
```

The plan records that Chimera must use the configured virtual environment:

```text
/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3
```

Local Python validation used:

```text
.venv/bin/python
```

## Allowlist Status

The existing allowlist entry remains disabled:

```text
name=v2_wp3_context_mass_score
enabled=false
enable_condition=pending_review_and_explicit_submission_approval_for_restricted_step_label_expanded_mass_plan_20260508_2148
```

Do not submit automatically. A later explicit approval may temporarily enable
this one entry and submit at most one Chimera Slurm job with:

```text
SCORE_PLAN=results/natural_evidence_v2/status/wp3_restricted_step_label_expanded_mass_plan_20260508_2148/qwen_v2_wp3_restricted_step_label_expanded_context_mass_score_plan.jsonl
OUTPUT_DIR=<fresh scratch output dir>
sbatch scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
```

## Decision

The repair plan and expanded mass score plan are ready for review. The next
allowed project action is explicit review/approval for one Slurm scoring job.
Until that approval exists, no Chimera job should be submitted from this plan.

WP4 and training remain blocked.

## 2026-05-09 Submission Approval

The existing restricted Step-label expanded action-verb context-mass score plan
was reviewed and approved for exactly one future Chimera Slurm scoring
submission:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_expanded_mass_plan_20260508_2148/qwen_v2_wp3_restricted_step_label_expanded_context_mass_score_plan.jsonl
```

The approval report is:

```text
results/natural_evidence_v1/status/hermes_reports/20260509_0155_restricted_step_label_expanded_mass_submission_approval.md
```

No Slurm job was submitted during the approval tick. The allowlist entry remains
disabled and may be temporarily enabled only by a later explicit submission
tick. The submission, if run, must use `wp3_context_mass_score.sbatch`, the
configured Chimera virtual environment, and a fresh output directory or the
wrapper's job-id default. WP4 and training remain blocked.
