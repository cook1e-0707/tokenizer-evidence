# WP3 Restricted Step-Label Mass-Aware Repair Plan from 850483

## Scope

This is an artifact-only repair plan after Slurm job `850483` completed and the
expanded restricted Step-label mass gate failed. It uses only existing 850483
context-mass artifacts and the 850434 density artifacts. It does not load a
tokenizer or model, submit Slurm, train, generate transcripts, run E2E, compute
FAR, or make any positive paper-facing claim.

Artifacts:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_mass_aware_repair_plan_20260509_0211/
```

Primary files:

```text
mass_aware_repair_summary.json
original_bank_reviews.jsonl
bucket_group_candidates.jsonl
mass_aware_recombined_bank_candidates.jsonl
qwen_v2_wp3_restricted_step_label_mass_aware_context_mass_score_plan.jsonl
```

The scoring plan was locally validated with the project virtual environment:

```text
.venv/bin/python scripts/natural_evidence_v2/score_wp3_context_mass.py \
  --score-plan results/natural_evidence_v2/status/wp3_restricted_step_label_mass_aware_repair_plan_20260509_0211/qwen_v2_wp3_restricted_step_label_mass_aware_context_mass_score_plan.jsonl \
  --validate-plan-only
```

Validation status:

```text
PASS_CONTEXT_MASS_SCORE_PLAN_VALIDATION
score_plan_rows=192
model_scoring_started=false
training_started=false
```

## Source Result

Job `850483` completed successfully at the Slurm level but failed the business
gate:

```text
mass_gate_status=FAIL
score_plan_rows=128
context_score_rows=112
invalid_tokenization_rows=16
mass_rows=7
```

No original expanded bank passed. The invalid rows all came from
`step_label_arrange_schedule_organize_plan_v1`, because `Organize` is not a
single Qwen next token in this context.

## Repair Logic

The failure was not uniform across all bucket groups. Six individual bucket
groups had mean full-vocab mass above the predeclared `0.005` minimum:

| Bucket group | Mean mass | Contexts below 0.005 |
|---|---:|---:|
| `Use, Take` | 0.0396515901 | 0 |
| `Create, Develop` | 0.0126107293 | 0 |
| `Choose, Make` | 0.0125512375 | 0 |
| `Determine, Define` | 0.0106194712 | 7 |
| `Check, Review` | 0.0062405573 | 7 |
| `Identify, Assess` | 0.0053986572 | 12 |

The repair plan recombines these already-scored bucket groups into candidate
two-way banks whose inferred mean masses meet the same rough mass and ratio
requirements. This is not a gate pass: every recombined bank must still be
freshly tokenizer-validated and context-mass-scored by Slurm before it can be
considered for WP3.

## Top Recombined Candidates

The strongest inferred candidate is:

```text
step_label_recombined_create_develop_vs_choose_make_v1
bucket_0 = [Create, Develop]
bucket_1 = [Choose, Make]
inferred_min_bucket_mass = 0.0125512375
inferred_mass_ratio = 1.0047399181
```

Other high-priority candidates:

```text
step_label_recombined_check_review_vs_identify_assess_v1
step_label_recombined_choose_make_vs_determine_define_v1
step_label_recombined_create_develop_vs_determine_define_v1
step_label_recombined_determine_define_vs_check_review_v1
```

Candidates involving `Use, Take` are included in the artifact output because
they pass inferred mass-ratio constraints against some high-mass groups, but
they should be reviewed more cautiously because their reference mass is much
higher than the other bucket groups.

## Decision

WP3 remains blocked. WP4, training, Qwen E2E, Llama, same-family null,
sanitizer, FAR aggregation, and paper-facing positive claims remain forbidden.

Next allowed action:

```text
Review the mass-aware repair plan. If approved, enable exactly one allowlist
entry and submit one Chimera Slurm context-mass scoring job for the 192-row
mass-aware recombined score plan in a fresh output directory.
```
