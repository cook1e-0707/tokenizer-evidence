# WP3 Restricted Step-Label Mass-Aware Score 850509 Review

## Scope

Slurm job `850509` scored the 192-row mass-aware recombined restricted
Step-label context-mass plan under base Qwen. This was tokenizer/model
context-mass scoring only: no text generation, no training, no E2E, no payload
recovery, no FAR aggregation, and no paper-facing positive claim.

Local artifacts:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_mass_aware_score_850509/
```

Remote output directory:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/wp3_restricted_step_label_mass_aware_score_20260509_021947
```

## Slurm Result

```text
job_id=850509
job_name=nat-ev-v2-wp3ctxm
partition=DGXA100
node=chimera13
state=COMPLETED
exit_code=0:0
runtime=00:00:44
```

The wrapper used the Chimera virtual environment:

```text
/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3
```

## Summary

```text
status=WP3_CONTEXT_MASS_SCORED_NOT_TRAINING_NOT_GENERATION
mass_gate_status=PASS_REVIEW_REQUIRED
score_plan_rows=192
context_score_rows=192
invalid_tokenization_rows=0
mass_rows=12
wp4_allowed=false
```

Tokenizer boundary validation passed with `192` valid rows. All rows used the
prefix-boundary repair policy and no candidate surface was dropped.

## Mass Gate Results

Gate thresholds:

```text
min_bucket_mass >= 0.005
max_bucket_mass_ratio <= 5.0
```

All `12/12` recombined candidate banks passed the context-specific model-mass
gate:

| Bank | Min mass | Ratio | Decision |
|---|---:|---:|---|
| `step_label_recombined_check_review_vs_identify_assess_v1` | 0.0053986572 | 1.1559 | pass |
| `step_label_recombined_choose_make_vs_check_review_v1` | 0.0062405573 | 2.0112 | pass |
| `step_label_recombined_choose_make_vs_determine_define_v1` | 0.0106194712 | 1.1819 | pass |
| `step_label_recombined_choose_make_vs_identify_assess_v1` | 0.0053986572 | 2.3249 | pass |
| `step_label_recombined_create_develop_vs_check_review_v1` | 0.0062405573 | 2.0208 | pass |
| `step_label_recombined_create_develop_vs_choose_make_v1` | 0.0125512375 | 1.0047 | pass |
| `step_label_recombined_create_develop_vs_determine_define_v1` | 0.0106194712 | 1.1875 | pass |
| `step_label_recombined_create_develop_vs_identify_assess_v1` | 0.0053986572 | 2.3359 | pass |
| `step_label_recombined_determine_define_vs_check_review_v1` | 0.0062405573 | 1.7017 | pass |
| `step_label_recombined_determine_define_vs_identify_assess_v1` | 0.0053986572 | 1.9671 | pass |
| `step_label_recombined_use_take_vs_choose_make_v1` | 0.0125512375 | 3.1592 | pass |
| `step_label_recombined_use_take_vs_create_develop_v1` | 0.0126107293 | 3.1443 | pass |

The strongest candidate remains:

```text
step_label_recombined_create_develop_vs_choose_make_v1
bucket_0 = [Create, Develop]
bucket_1 = [Choose, Make]
min_bucket_mass = 0.0125512375
mass_ratio = 1.0047399181
candidate_normalized_mass_ratio = 1.0731978929
```

## Interpretation

This repairs the previous 850483 mass bottleneck for restricted Step-label
action-verb banks. The mass result is much stronger than the original expanded
bank result because it recombines high-mass bucket groups instead of preserving
the original hand-paired low-mass buckets.

This is still not a full WP3 pass. The prior model-output density audit for the
restricted Step-label policy still failed its structural gate, even though it
was close:

```text
total_responses=256
complete_step_label_response_count=253
complete_step_label_response_rate=0.98828125
mean_detected_structural_slots_per_response=15.8125
```

Before WP4 or any training can be considered, the project still needs a reviewed
policy selection and a density repair/check that confirms the chosen prompt
family reliably yields the required slot count under base-Qwen generation.

## Decision

The mass-aware recombined context-mass subgate passes review, but WP3 overall
does not pass yet.

Recommended next allowed action:

```text
Artifact-only select a primary restricted Step-label bank set from the 12
passing banks, then prepare a strict density repair/audit plan for prompts that
force exactly 16 Step N: lines. Do not submit another Slurm job without review
and explicit approval.
```

Still forbidden:

```text
WP4, training, Qwen E2E, Llama, same-family null, sanitizer, FAR aggregation,
and paper-facing positive claims.
```
