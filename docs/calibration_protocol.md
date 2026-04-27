# Calibration Protocol

Status: B0 frozen on 2026-04-27.

This document freezes how false-accept thresholds and utility budgets are chosen
before B1/B2 baseline execution. It is a protocol artifact, not an experimental
result.

## Calibration Objective

Every method must be evaluated at a pre-final operating point. The operating
point is selected on a calibration split only, then frozen before final baseline
evaluation.

Score convention:

```text
higher ownership_score = stronger evidence for the claimed owner payload
```

Final decision:

```text
accepted = ownership_score >= frozen_threshold
```

Frozen false-accept target:

```text
target_far = 0.01
```

## Query Budget

The final ownership decision uses:

```text
M = 4
```

This is the maximum number of verifier queries per owner-payload decision in the
primary baseline table. A method may use fewer queries, but it may not use more
queries and still be labeled matched-budget.

Calibration samples are used only to select thresholds. They do not increase the
final decision query budget.

## Calibration Split

The calibration split must be distinct from the final baseline matrix.

Frozen calibration split:

| Field | Value |
|---|---|
| backbone | `Qwen/Qwen2.5-7B-Instruct` |
| block count | `2` |
| prompt family | standing exact-slot prompt family (`PF1`) |
| calibration payloads | `U01`, `U05`, `U09`, `U13` |
| calibration seed | `41` |
| final payloads | `U00`, `U03`, `U12`, `U15` |
| final seeds | `17`, `23`, `29` |

The calibration split may include diagnostic hard cases, but those diagnostics
must not be used to select final thresholds unless they are listed in the frozen
calibration split above.

## Negative Sets For FAR

FAR is measured against null ownership claims. B1/B2 calibration must use all
available negative sets below and report each separately.

1. `foundation_null`: unadapted Qwen 7B under the same prompt contract.
2. `wrong_payload_null`: a valid adapted run tested against the wrong claimed
   payload.
3. `organic_prompt_null`: non-evidence prompts where no owner payload should be
   recoverable.

The aggregate calibration threshold must satisfy `target_far = 0.01` on the
pooled negative set. If no threshold satisfies the target, use the most
conservative threshold and mark the method `far_unmatched`.

## Positive Set

Positive calibration examples are valid owner-payload pairs on the frozen
calibration split. Positives may be used to report calibration sensitivity, but
they must not override the FAR constraint.

## Threshold Selection

For each method:

1. Collect calibration scores and labels.
2. Enumerate candidate thresholds from observed calibration scores.
3. Select the lowest threshold whose pooled negative FAR is at most `0.01`.
4. If multiple thresholds have the same FAR and sensitivity, select the larger
   threshold.
5. Write the selected threshold before final evaluation.

The existing helper `scripts/calibrate.py` follows the same score direction and
target-FAR convention, but B1/B2 must use real calibration scores rather than
synthetic defaults.

Required threshold record:

- `method`
- `baseline_family`
- `score_name`
- `score_direction`
- `target_far`
- `frozen_threshold`
- `calibration_observed_far`
- `calibration_false_accept_count`
- `calibration_negative_count`
- `calibration_positive_count`
- `calibration_sensitivity`
- `calibration_split_hash`
- `threshold_candidates_hash`
- `threshold_frozen_at`

## Utility Budget

Primary utility metric:

```text
utility_acceptance_rate
```

The metric is computed on the frozen organic/utility prompt set using
`src/evaluation/utility_eval.py` semantics: the fraction of prompts accepted by
the utility evaluator.

Matched utility rule:

```text
baseline utility degradation <= primary method degradation + 0.02
```

If Wilson confidence intervals are available, overlapping intervals may also be
used to mark the utility comparison as matched.

If a baseline cannot produce utility outputs for the frozen utility prompt set,
it must be labeled `utility_unavailable` and cannot support a matched-utility
claim.

## Reporting FAR And Utility

Every final baseline row must report:

- `ownership_score`
- `frozen_threshold`
- `accepted`
- `target_far`
- `calibration_observed_far`
- `final_observed_far`
- `final_far_false_accept_count`
- `final_far_negative_count`
- `final_far_wilson_low`
- `final_far_wilson_high`
- `utility_acceptance_rate`
- `utility_delta_vs_foundation`
- `utility_delta_vs_primary`
- `utility_match_status`

Counts must be reported directly. Do not hide low sample counts behind rounded
rates.

## Inclusion And Exclusion Rules

The calibration package uses the same accounting semantics as the baseline
package.

Valid completed failures remain in the denominator. Invalid exclusions are
allowed only for artifact or contract failures:

- missing required score file
- corrupted score file
- incomplete run with no final score
- wrong config or calibration split
- train/eval contract mismatch
- missing checkpoint where required
- path violation that stores raw artifacts in home/repo

Not allowed as exclusions:

- high FAR
- low utility
- low sensitivity
- failed ownership recovery
- failed watermark/provenance detection
- method instability

## Required Calibration Artifacts

B1/B2 must write:

- `results/processed/paper_stats/baseline_calibration_summary.json`
- `results/tables/baseline_calibration.csv`
- `results/tables/baseline_far_summary.csv`
- `results/tables/baseline_utility_summary.csv`

These files must be generated before final baseline evaluation and referenced by
the final `baseline_summary.json`.

## Frozen Guardrails

- Do not inspect final baseline outcomes before freezing thresholds.
- Do not change `target_far` after calibration.
- Do not change the final payload/seed matrix after calibration.
- Do not use final failures to choose a new threshold.
- Do not report a baseline as matched-budget if it violates query, FAR, or
  utility constraints.
