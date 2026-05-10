# WP3-R1 Strict Density Expansion Plan Review

## Scope

This review covers the artifact-only WP3-R1 strict Step-label density expansion
plan:

```text
results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/
```

It follows the expert execution standard:

```text
docs/natural_evidence_v2/V2_EXECUTION_STANDARD_AFTER_850523_EXPERT_REVIEW.md
```

This plan does not train, run WP4, run Qwen E2E, run Llama, run same-family
nulls, run sanitizer benchmarks, aggregate FAR, decode payloads, or make a
paper-facing positive claim.

## Plan Summary

```text
status=WP3_R1_STRICT_DENSITY_EXPANSION_PLAN_READY_ARTIFACT_ONLY
source_density_diagnostic_job_id=850771
source_primary_bank_id=step_label_recombined_create_develop_vs_choose_make_v1
dev_prompt_count=512
eval_prompt_count=2048
total_prompt_count=2560
```

Prompt variants:

```text
strict_literal_16_step_lines=854
strict_no_heading_16_step_lines=854
strict_numbered_step_label_lines=852
```

The plan keeps the strict line-start detector contract. It does not reclassify
the 850523 inline-paragraph failure as passing.

## Gate Fields

The expansion gate records:

```text
dev_outputs_min=512
dev_complete_step_label_response_rate_min=0.995
dev_mean_detected_slots_min=15.9
dev_forbidden_public_surface_rate_required=0.0
eval_outputs_min=2048
eval_complete_step_label_response_rate_min=0.995
eval_oracle_prompt_local_frame_completion_rate_min=0.95
eval_forbidden_public_surface_rate_required=0.0
```

Oracle prompt-local frame completion is defined as observing one ordered strict
line-start `Step 1:` through `Step 16:` slot sequence in a response, which
would expose all 16 prompt-local 2-way coordinates under no-erasure
substitution.

## Validation

Local validation passed:

```text
PASS_RESTRICTED_STEP_LABEL_DENSITY_PLAN_VALIDATION
prompt_count=2560
```

Additional checks passed:

```text
py_compile run_wp3_restricted_step_label_density_audit.py
py_compile build_wp3_r1_strict_density_expansion_plan.py
bash -n wp3_restricted_step_label_density_audit.sbatch
pytest -q tests/test_natural_evidence_v2_restricted_density.py
```

## Decision

The plan is valid for one user-approved Chimera Slurm WP3-R1 strict density
audit. The job remains a base-Qwen model-output density audit only. It is not
training, E2E, payload recovery, FAR, or a positive claim.

Still blocked:

- WP4;
- training;
- Qwen E2E;
- Llama;
- same-family null;
- sanitizer benchmark;
- FAR aggregation;
- paper-facing positive claims.
