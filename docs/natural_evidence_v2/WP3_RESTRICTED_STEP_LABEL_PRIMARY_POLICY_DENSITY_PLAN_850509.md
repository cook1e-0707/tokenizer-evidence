# WP3 Restricted Step-Label Primary Policy and Strict Density Plan

## Scope

This artifact-only plan follows the 850509 mass-aware context-mass score review.
It selects a primary restricted Step-label bank and prepares a strict density
repair audit plan. It does not submit Slurm, train, generate outputs, run E2E,
compute FAR, or make a paper-facing positive claim.

Artifacts:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_primary_policy_density_plan_20260509_0225/
```

Primary files:

```text
restricted_step_label_policy.json
restricted_step_label_bucket_bank.json
restricted_step_label_detector_contract.json
restricted_step_label_density_design.json
restricted_step_label_strict_density_audit_prompts.jsonl
restricted_step_label_strict_density_audit_slurm_review.json
restricted_step_label_primary_policy_summary.json
```

## Selected Primary Bank

The selected primary bank is:

```text
candidate_bank_id=step_label_recombined_create_develop_vs_choose_make_v1
bucket_0=[Create, Develop]
bucket_1=[Choose, Make]
min_bucket_mass=0.0125512375
mass_ratio=1.0047399181
source_job=850509
```

This bank is selected because it has the strongest combination of high minimum
bucket mass and near-unity full-vocabulary mass ratio among the passing
recombined banks.

Backup banks are recorded but not active in the primary policy:

```text
step_label_recombined_choose_make_vs_determine_define_v1
step_label_recombined_create_develop_vs_determine_define_v1
step_label_recombined_check_review_vs_identify_assess_v1
step_label_recombined_determine_define_vs_check_review_v1
```

## Density Repair Plan

The plan uses the existing 256 strict repair prompts that require exactly
sixteen labeled lines:

```text
Step 1:
...
Step 16:
```

Local validation passed:

```text
PASS_RESTRICTED_STEP_LABEL_DENSITY_PLAN_VALIDATION
prompt_count=256
```

The density wrapper was also repaired to accept reviewed recombined restricted
Step-label banks instead of hard-coding the two old candidate bank ids.

## Decision

The mass subgate is now strong enough for a strict density repair audit, but
WP3 overall is still not passed. The next step requires explicit approval before
submitting another Slurm job.

Next allowed action:

```text
Review this primary policy and strict density plan. If approved, submit exactly
one Chimera Slurm restricted Step-label density audit using the strict repair
prompts and this policy directory, then disable the allowlist entry immediately.
```

Still forbidden:

```text
WP4, training, Qwen E2E, Llama, same-family null, sanitizer, FAR aggregation,
and paper-facing positive claims.
```
