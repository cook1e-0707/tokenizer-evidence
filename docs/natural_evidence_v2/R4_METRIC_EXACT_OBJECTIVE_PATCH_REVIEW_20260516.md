# R4 Metric-Exact Objective Patch Review

Date: 2026-05-16

## Status

`PASS_R4_METRIC_EXACT_OBJECTIVE_PATCH_STATIC_REVIEW_NO_TRAINING`

This patch is artifact-only. It does not train, score a model, generate text, submit Slurm, run Llama, aggregate FAR, run sanitizer, test payload diversity, or make a paper-facing claim.

## Change

`scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py` now supports an explicit protected surface margin mode:

```text
--surface-margin-loss-mode mass_relu
--surface-margin-loss-mode logsumexp_softplus
```

The default is still:

```text
mass_relu
```

so existing routes have no behavior change unless a future reviewed route opts into `logsumexp_softplus`.

The new mode computes a margin over exact prefix-native target/other first-token id sets:

```text
target_score = logsumexp(logits[target_ids])
other_score = logsumexp(logits[other_ids])
loss = softplus(margin_floor + other_score - target_score)
```

This is closer to the teacher-forced surface-mass gate than the older probability-mass ReLU surrogate, while preserving the existing target-mass floor, ceiling, and reviewed stratum-weight controls.

## Guardrails

The patch keeps the current safety contract:

```text
disabled by default
protected arm only
task-only path unchanged
base/task-only scoring unchanged
exact target/other token ids
target/other overlap fails closed
no generated transcript phrase mining
no threshold changes after locked outputs
```

## Validation

Local validation:

```text
uv run pytest tests/natural_evidence_v2/test_r4_metric_exact_objective_helpers.py tests/natural_evidence_v2/test_r4_training_objective_disabled_by_default.py tests/natural_evidence_v2/test_r4_target_mass_floor_loss.py tests/natural_evidence_v2/test_r4_stratum_weighting_controls.py -q
14 passed

uv run python -m py_compile scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py
PASS
```

## Next

The next allowed action is artifact-only route planning for a small H200 micro-overfit train-and-score job using the reviewed candidate-v3 split and explicit `logsumexp_softplus` mode.

No Slurm submission is unlocked by this review alone.
