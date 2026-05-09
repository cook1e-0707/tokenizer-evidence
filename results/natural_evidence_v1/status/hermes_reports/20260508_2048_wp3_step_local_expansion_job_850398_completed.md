# Hermes/Codex progress report

## Status

Slurm job `850398` completed and was reviewed.

## Result

```text
job_id=850398
state=COMPLETED
exit_code=0:0
score_plan_rows=72
context_score_rows=63
invalid_tokenization_rows=9
mass_rows=21
mass_gate_status=FAIL
```

Outputs:

```text
results/natural_evidence_v2/status/wp3_step_local_expansion_mass_score_850398/
docs/natural_evidence_v2/WP3_STEP_LOCAL_EXPANSION_REVIEW.md
```

## Passing Banks

Two `Step N: ` sentence-case action-verb banks passed:

```text
step_local_step_label_seed_check_review_choose_make_v1
side0=[Check, Review]
side1=[Choose, Make]
min_bucket_mass=0.0100467710
ratio=1.8203

step_local_step_label_start_begin_create_set_v1
side0=[Start, Begin]
side1=[Create, Set]
min_bucket_mass=0.0071791444
ratio=3.8920
```

## Interpretation

The step-local `Step N: ` direction now has a real base-Qwen mass signal.
Overall WP3 still fails because density is only structural and most expanded
banks failed or were tokenizer-invalid.

## Next action

Build an artifact-only restricted step-label policy from the two passing banks
and plan density measurement for either a sixteen-step/list prompt family or an
8-step family with additional non-step slots.

## Guardrails

No training, generation, Qwen E2E, Llama, same-family null, sanitizer, FAR
aggregation, or positive paper claim was started.
