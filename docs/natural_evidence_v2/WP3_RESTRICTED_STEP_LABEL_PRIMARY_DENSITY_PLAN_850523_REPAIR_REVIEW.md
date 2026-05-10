# WP3 Restricted Step-Label Primary Density Plan 850523 Repair Review

## Scope

This review covers the artifact-only repaired strict density plan created after
Slurm job `850523`:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_primary_policy_density_plan_850523_repair_20260509_0310/
```

It was reviewed under:

```text
docs/natural_evidence_v2/V2_EXECUTION_STANDARD_AFTER_850523_EXPERT_REVIEW.md
```

This review did not submit Slurm, train, generate protected transcripts, run
Qwen E2E, run Llama, run same-family nulls, run sanitizer benchmarks, aggregate
FAR, or make a paper-facing positive claim.

## Findings

The repair correctly keeps the strict line-start detector contract and does not
reclassify the `850523` inline-paragraph response as passing.

The repaired prompt file contains `192` prompts:

```text
strict_literal_16_step_lines=64
strict_no_heading_16_step_lines=64
strict_numbered_step_label_lines=64
```

The failed `strict_compact_step_label_lines` variant is removed from the prompt
JSONL. The plan validation command passed:

```text
PASS_RESTRICTED_STEP_LABEL_DENSITY_PLAN_VALIDATION
prompt_count=192
```

The Slurm review artifact records:

```text
allowlist_enabled=false
explicit_approval_required_before_submission=true
slurm_submitted=false
```

## R1 Standard Check

The repaired plan is a useful prompt-side repair seed, but it is not sufficient
as a WP3-R1 gate plan under the new execution standard.

The new standard requires:

```text
dev outputs >= 512
eval outputs >= 2,048
dev complete_step_label_response_rate >= 0.995
dev mean_detected_slots_per_response >= 15.9
eval complete_step_label_response_rate >= 0.995
eval oracle_prompt_local_frame_completion_rate >= 0.95
forbidden_public_surface_rate = 0
```

The current repaired artifact has only `192` dev prompts and no separate eval
prompt set. Submitting it as-is would not produce an R1-compliant density gate,
even if the 192-prompt model-output audit passed.

## Decision

Status:

```text
REVIEWED_R1_REPAIR_SEED_NOT_APPROVED_FOR_SLURM_OR_GATE
```

The repaired 850523 plan is reviewed as a valid artifact-only repair seed:

- it removes the failing compact prompt variant;
- it preserves the strict line-start detector;
- it leaves `850523` as a close fail, not a pass;
- it keeps Slurm submission disabled pending explicit approval.

It is not approved as a standalone WP3-R1 density submission plan because it
does not meet the new dev/eval volume requirements.

Next allowed action:

```text
Prepare an artifact-only WP3-R1 strict density expansion plan with dev >=512,
eval >=2048, the strict line-start detector, and an explicit oracle prompt-local
frame completion field for eval review. Do not submit Slurm without explicit
approval.
```

Still blocked:

- WP4
- training
- Qwen E2E
- Llama
- same-family null
- sanitizer benchmark
- FAR aggregation
- paper-facing positive claims
