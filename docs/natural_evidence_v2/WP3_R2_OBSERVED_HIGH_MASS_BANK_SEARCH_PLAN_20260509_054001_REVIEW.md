# WP3-R2 Observed High-Mass Bank Search Plan Review

## Scope

This review covers the artifact-only WP3-R2 bank search plan:

```text
results/natural_evidence_v2/status/wp3_r2_observed_high_mass_bank_search_plan_20260509_054001/
```

The plan was built from detected Step-label first words in job `850885` model
outputs. It does not load a tokenizer/model, submit Slurm, train, generate
text, run E2E, aggregate FAR, or make a paper-facing claim.

## Inputs

```text
source_density_dir=results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_audit_850885
source_detected_slot_rows=40945
config=configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml
```

The observed high-frequency sentence-case action-word pool is:

```text
Set, Plan, Create, Prepare, Encourage, Ensure, Use, Review,
Assign, Identify, Schedule, Establish, Keep, Check, Choose, Develop,
Gather, Organize, Start, Define, Take, Determine, Train, Send
```

Top observed counts include:

```text
Set=2466
Plan=1958
Create=1711
Prepare=1653
Encourage=1277
Ensure=1166
Use=1162
Review=1146
Assign=1143
Identify=990
```

These are observed surface counts only. They are not bucket mass scores and do
not imply trainability or payload recovery.

## Candidate Plan

The plan writes:

```text
qwen_v2_wp3_r2_observed_high_mass_bank_candidates.jsonl
qwen_v2_wp3_r2_observed_high_mass_context_mass_score_plan.jsonl
qwen_v2_wp3_r2_observed_high_mass_bank_search_summary.json
qwen_v2_wp3_r2_observed_high_mass_context_mass_slurm_review.json
```

Plan size:

```text
candidate_bank_count=26
score_plan_rows=416
score_plan_rows_per_bank=16
```

The candidates include both one-surface-per-bucket probes such as `Set` vs
`Plan`, and two-surface bucket probes such as `Set/Plan` vs
`Create/Prepare`. This is a search plan, not a primary bank approval.

## Validation

Local validation passed:

```text
.venv/bin/python -m py_compile scripts/natural_evidence_v2/build_wp3_r2_high_mass_observed_bank_search_plan.py scripts/natural_evidence_v2/score_wp3_context_mass.py
.venv/bin/python -m pytest -q tests/test_natural_evidence_v2_restricted_density.py
.venv/bin/python scripts/natural_evidence_v2/score_wp3_context_mass.py \
  --score-plan results/natural_evidence_v2/status/wp3_r2_observed_high_mass_bank_search_plan_20260509_054001/qwen_v2_wp3_r2_observed_high_mass_context_mass_score_plan.jsonl \
  --validate-plan-only
```

Validation status:

```text
PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION
score_plan_rows=416
empty_prefix_rows=0
template_preflight_only=false
```

## Decision

Review status:

```text
READY_FOR_SINGLE_SLURM_CONTEXT_MASS_SCORING_REVIEW_NOT_SUBMITTED
```

This plan is the correct next WP3-R2 step because the current
`Create/Develop` vs `Choose/Make` bank remains below the pilot absolute mass
threshold (`0.0125512375 < 0.03`). The new plan tests candidates built from
the actual observed Step-label surface distribution.

No Slurm job has been submitted from this plan. If approved, enable exactly one
allowlist entry and submit one Chimera Slurm context-mass scoring job using:

```text
scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
```

WP4, training, Qwen E2E, Llama, same-family null, sanitizer, FAR aggregation,
and paper-facing positive claims remain forbidden.
