# R4 Metric-Exact Coverage-Scale 864761 Review

Canonical status: `PASS_R4_METRIC_EXACT_COVERAGE_SCALE_864761_TEACHER_FORCED_GATE_WITH_TRAIN_SPLIT_CAVEAT`

Job `864761` completed cleanly on H200/pomplun:

```text
job_name: nat-ev-v2-r4mof
node: chimera21
elapsed: 00:09:35
exit_code: 0:0
```

The run did not start generation, Qwen E2E, Llama, same-family null,
sanitizer, FAR, payload-diversity, or paper-facing claim work.

## Gate Result

The teacher-forced surface-mass gate passed:

```text
protected mean target mass:      0.1564211
base mean target mass:           0.0048318
task-only mean target mass:      0.0016724
protected lift vs base:         +0.1515893   required >= +0.15
protected lift vs task-only:    +0.1547487   required >= +0.10
protected rank1 rate:            1.000000    required >= 0.70
protected median margin:        +0.1542559   required > 0
task-only lift vs base:         -0.0031594
```

This is the first R4 candidate-v3 metric-exact adapter run in this sequence to
pass the teacher-forced surface-mass gate.

## Coverage Caveat

The route was named coverage-scale, but the train artifact contains `512`
unique rows:

```text
train input row count: 512
max steps: 4096
batch size implied by route: 2
score rows: 8192
```

So the run should be interpreted as repeated-cycled 512-row training with
stronger floor pressure that generalizes to the 8192-row scoring set, not as an
8192 unique train-row coverage experiment. This caveat does not invalidate the
teacher-forced score pass, but it must be preserved before any downstream
route.

## Training Summary

```text
surface_margin_loss_mode: logsumexp_softplus
task_ce_weight: 0.0
target_mass_floor: 0.25
target_mass_floor_lambda: 75.0
target_mass_ceiling: 0.5
target_mass_ceiling_lambda: 5.0
final_floor_loss: 0.0
final_margin_loss: 0.006309905555099249
final_loss: 0.006309905555099249
```

## Artifacts

```text
results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864761/
results/natural_evidence_v2/status/r4_metric_exact_coverage_scale_864761_review/
```

Remote hashes:

```text
adapter_model.safetensors: 21dae40ef91222da43cf211cd54ef7b02715daafe7f107f387c0ab8b39c85da4
train summary:             b6f1932789431e8afb0b1e99ec5557eefdbe6f05456396aa4b9aaf04f4244024
score summary:             f1940df3391485a22abe0af7089a348f5ec8aa165817fe0a422c19699bc2a024
score rows:                a87e7bf97e039b339e12f25c83dbd30ee5530dfd9b71dc68d9f596843b9e031c
```

## Next State

The next allowed action is an artifact-only route decision for a small Qwen dev
generation diagnostic using the reviewed adapter and the same candidate-v3
surface contract. No generation should start until that route records its
preconditions, wrapper contract, allowlist entry, Hermes notification, remote
hash preflight, and post-submit allowlist shutdown policy.
