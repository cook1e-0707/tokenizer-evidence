# WP3-R2 Prompt-Conditioned Bank Search Plan Review

## Scope

This review covers the artifact-only prompt-conditioned R2 repair plan:

```text
results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_bank_search_plan_20260509_055137/
```

The plan was created after Slurm job `851233` showed that scoring candidate
banks under the generic `Step N:` prefix is too weak (`best min_bucket_mass`
about `0.00597`, below the pilot threshold `0.03`). This plan does not load a
tokenizer/model, submit Slurm, train, generate text, run E2E, aggregate FAR, or
make a paper-facing positive claim.

## Repair Rationale

The `851233` search scored only:

```text
Step N:
```

That omits both the owner prompt and the generated assistant prefix. The
prompt-conditioned repair plan instead scores each candidate surface under:

```text
Qwen chat template(user prompt) + assistant prefix before the Step slot
```

This directly tests whether the candidate bucket has sufficient mass in the
actual controlled-natural context where the model generated the Step-label
surface.

The existing context-mass scorer was updated to support rows with:

```text
chat_prompt_text
assistant_prefix_before_candidate
scoring_context_kind=chat_prompt_plus_assistant_prefix
```

Old raw-prefix rows remain supported.

## Plan Summary

```text
source_density_dir=results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_audit_850885
previous_score_dir=results/natural_evidence_v2/status/wp3_r2_observed_high_mass_context_mass_score_851233
source_detected_slot_rows=40945
selected_context_count=512
candidate_bank_count=20
score_plan_rows=10240
recommended_max_length=1536
```

Selected context coverage:

```text
wp3_r1_dev=272
wp3_r1_eval=240
strict_literal_16_step_lines=176
strict_no_heading_16_step_lines=176
strict_numbered_step_label_lines=160
each Step index 1..16 = 32 contexts
```

The plan excludes surfaces that were tokenizer-invalid in `851233`:

```text
Define, Encourage, Ensure, Gather, Organize, Review, Start, Use
```

The retained observed surface pool starts with:

```text
Set, Plan, Create, Prepare, Assign, Identify, Schedule, Establish,
Keep, Check, Choose, Develop, Take, Determine, Train, Send
```

## Validation

Local checks passed:

```text
.venv/bin/python -m py_compile scripts/natural_evidence_v2/build_wp3_r2_prompt_conditioned_bank_search_plan.py scripts/natural_evidence_v2/score_wp3_context_mass.py
.venv/bin/python -m pytest -q tests/test_natural_evidence_v2_restricted_density.py
.venv/bin/python scripts/natural_evidence_v2/score_wp3_context_mass.py \
  --score-plan results/natural_evidence_v2/status/wp3_r2_prompt_conditioned_bank_search_plan_20260509_055137/qwen_v2_wp3_r2_prompt_conditioned_context_mass_score_plan.jsonl \
  --validate-plan-only
```

Validation status:

```text
PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION
score_plan_rows=10240
empty_prefix_rows=0
template_preflight_only=false
```

## Decision

Review status:

```text
READY_FOR_SINGLE_SLURM_PROMPT_CONDITIONED_CONTEXT_MASS_SCORING_REVIEW_NOT_SUBMITTED
```

No Slurm job has been submitted from this plan. If approved, enable exactly one
allowlist entry and submit one Chimera Slurm context-mass scoring job using:

```text
scripts/natural_evidence_v2/slurm/wp3_context_mass_score.sbatch
```

Recommended scoring setting:

```text
MAX_LENGTH=1536
```

WP4 remains blocked until R2 passes. Training, Qwen E2E, Llama, same-family
null, sanitizer, FAR aggregation, and paper-facing positive claims remain
forbidden.
