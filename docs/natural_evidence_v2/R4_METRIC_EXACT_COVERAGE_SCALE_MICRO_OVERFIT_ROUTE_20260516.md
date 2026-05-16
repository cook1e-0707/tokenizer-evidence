# R4 Metric-Exact Coverage-Scale Micro-Overfit Route

Canonical status: `V2_R4_METRIC_EXACT_COVERAGE_SCALE_MICRO_OVERFIT_ROUTE_ARTIFACT_ONLY`

This route follows the reviewed failure of job `864705`. It is not a
generation route and does not unlock Qwen E2E, Llama, same-family null,
sanitizer, FAR, payload diversity, or paper-facing claims.

## Source Failure

Job `864705` completed cleanly but failed the teacher-forced surface-mass gate:

```text
protected mean target mass:   0.0847697
protected lift vs base:      +0.0799378
protected lift vs task-only: +0.0830972
protected rank1 rate:         1.000000
protected median margin:     +0.0772580
```

The result shows that floor-dominant pressure is directionally effective, but
not enough. The main gap is absolute mass, not rank ordering or tokenizer
boundary validity.

## Repair Hypothesis

Job `864705` trained on only `512` rows for `128` one-example steps while
scoring `8192` rows. This route tests whether the same metric-exact objective
works when the protected adapter sees the full candidate-v3 score row set and a
slightly stronger target-mass floor.

## Planned Route

```text
allowlist entry: v2_r4_candidate_v3_coverage_scale_micro_overfit_h200
wrapper: scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
surface margin loss mode: logsumexp_softplus
task CE weight: 0.0
target mass floor: 0.25
target mass floor lambda: 75.0
target mass ceiling: 0.50
target mass ceiling lambda: 5.0
margin lambda: 1.0
max train rows: 8192
max score rows: 8192
max steps: 4096
batch size: 2
gradient accumulation steps: 8
learning rate: 1e-4
```

This yields one coverage pass over `8192` examples at the training batch level.
The route remains teacher-forced training/scoring only.

## Gate

```text
protected lift vs base >= +0.15
protected lift vs task-only >= +0.10
protected rank1 rate >= 0.75
protected median margin > 0
task-only lift anomaly = false
target/other overlap = 0
scorer boundary failures = 0
```

If this fails, do not run generation. The next decision should inspect whether
the failure is still absolute mass, coverage, collapse, or objective mismatch.
