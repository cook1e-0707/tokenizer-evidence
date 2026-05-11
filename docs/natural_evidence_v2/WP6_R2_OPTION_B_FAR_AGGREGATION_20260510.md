# WP6-R2 Option B FAR Aggregation

## Decision

Aggregated False Accept Rate from WP6-R2 Option B robust-block scale eval (job 852426).
All 128 null trials rejected at budget=64.

## FAR Summary

- Total null trials: 128
- Total false accepts: 0
- FAR: 0.000000
- Wilson 95% CI upper bound: 0.029138
- Rule-of-three upper bound: 0.023438

## Per-Condition FAR (budget=64)

- **raw**: 0/8 false accepts, FAR=0.000000, Wilson upper=0.324416
- **task_only**: 0/8 false accepts, FAR=0.000000, Wilson upper=0.324416
- **wrong_key**: 0/8 false accepts, FAR=0.000000, Wilson upper=0.324416
- **wrong_payload**: 0/8 false accepts, FAR=0.000000, Wilson upper=0.324416

## Per-Budget Breakdown (all conditions combined)

- **budget=8**: 0/32 false accepts, FAR=0.000000, Wilson upper=0.107183
- **budget=16**: 0/32 false accepts, FAR=0.000000, Wilson upper=0.107183
- **budget=32**: 0/32 false accepts, FAR=0.000000, Wilson upper=0.107183
- **budget=64**: 0/32 false accepts, FAR=0.000000, Wilson upper=0.107183

## Claim Boundaries

Allowed claim:
> WP6-R2 Option B robust-block coordinate-majority decoder shows zero false accepts across 128 null trials at budget=64, with Wilson 95% CI upper bound < 0.0291.

Forbidden claims:
- Full FAR (requires organic prompt nulls and non-owner probes)
- Stealth guarantee
- Cross-family generality (requires Llama positive recovery)
- Robustness (requires sanitizer benchmark pass)
- Paper-facing positive claim (requires all gates pass)

## Next Allowed Actions

1. Llama-3.1-8B migration (WP5 training + WP6 E2E)
2. Same-family null experiments
3. After Llama passes: sanitizer benchmarks

## Validation

Artifact-only aggregation; no Slurm, no training, no generation.

