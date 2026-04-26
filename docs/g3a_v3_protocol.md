# G3a-v3 Protocol

G3a-v3 is a principled held-out repair for G3a-v2. It does not overwrite G3a-v1 or G3a-v2, and it must not use the final G3a-v3 test matrix for hyperparameter selection.

## Motivation

G3a-v2 completed all 36 final cases and produced 32 exact/RS successes with 4 valid method failures. The failures are semantic slot/bucket substitutions with no erasures, matching contracts, and no parser evidence. The v2 diagnostics do not confirm a single root cause because per-slot target-vs-wrong bucket logmasses were not saved.

The v3 repair therefore targets bucket separation directly rather than post-hoc rerunning only the four failed v2 cases.

## Objective

The current bucket loss is:

```text
L_bucket = mean_j -log P(target_bucket_j | prefix_j)
```

G3a-v3 adds a hinge margin over bucket log-scores:

```text
s(B) = logsumexp_{w in B} logits(w)
L_margin = mean_j max(0, gamma + max_{B != B+} s(B) - s(B+))
L_total = L_task + lambda_set * L_bucket + lambda_margin * L_margin + lambda_reg * L_reg
```

The implemented compiled objective is `margin_aware_bucket_mass`. It logs `normalized_L_set_mean`, `normalized_L_margin_mean`, `total_evidence_loss_mean`, target-bucket mass, and slot margins per step.

## Validation Set

Validation is for hyperparameter selection only.

- Block counts: `B1`, `B4`
- Seed: `41`
- Payloads: `U00`, `U03`, `U12`, `U15`
- Fixed hyperparameters: `lora_r=16`, `learning_rate=3e-5`, `epochs=96`, `lambda_set=2.0`, `lambda_reg=0.0`
- Sweep: `margin_gamma in {0.5, 1.0}`, `lambda_margin in {0.25, 0.5}`, `checkpoint_selection_metric in {training_total_evidence_loss_mean, training_min_slot_margin}`
- Validation target: `64 train + 64 eval`

V2 failures may be used as diagnosis and unit-test motivation, but not as final G3a-v3 success claims.

## Selection Rule

Select one operating point before final launch:

1. Highest validation exact-gate success rate.
2. Highest validation RS-gate success rate.
3. Highest validation mean bucket margin.
4. Lower validation total evidence loss as tie-breaker.

Forbidden selection inputs:

- final reported G3a-v3 test acceptance
- manual failed final-test payload inspection
- post-evaluation threshold changes

## Frozen Operating Point

Status: `pending_validation`.

Final manifests must not be generated or launched until `configs/reporting/g3a_block_scale_v3.yaml` records:

- `selected_operating_point.status: frozen_before_final_launch`
- `selected_operating_point.final_launch_allowed: true`
- `margin_gamma`
- `lambda_margin`
- `checkpoint_selection_metric`

## Final Held-Out Matrix

The final matrix is larger than G3a-v2 and is distinct from validation by seed.

- Block counts: `B1`, `B2`, `B4`
- Payloads: `U00..U15`
- Seeds: `17`, `23`, `29`
- Final eval target: `144`

The current train/eval handoff stores one generated text path per eval payload. Unless eval-time generation is refactored to reuse a single checkpoint across payloads, launch planning assumes `144 train + 144 eval` jobs. At 24 requested GPU-hours per job, the final matrix requests `6912` GPU-hours. Approval is required before launching final runs.

## Required Outputs

- `results/tables/g3a_v3_block_scale.csv`
- `results/tables/g3a_v3_block_scale.tex`
- `results/tables/g3a_v3_slot_margin.csv`
- `results/tables/g3a_v3_failure_cases.csv`
- `results/processed/paper_stats/g3a_v3_summary.json`
- `results/processed/paper_stats/g3a_v3_run_inclusion_list.json`
- `results/processed/paper_stats/g3a_v3_compute_accounting.json`

## Paper-Ready Gate

G3a-v3 can be artifact-paper-ready only if all conditions hold:

- real contract hash checks pass
- `valid_completed_count == target_count`
- `invalid_excluded_count == 0`, unless a concrete non-method artifact failure is documented
- exact and RS-aware gates are both reported
- all failures, if any, are valid method failures and remain in the denominator
- validation set and final set are distinct
- hyperparameters were frozen before final matrix launch
- no threshold changes occurred after final evaluation

G3a-v3 is not claim-paper-ready until the final held-out matrix is complete and reviewed.
