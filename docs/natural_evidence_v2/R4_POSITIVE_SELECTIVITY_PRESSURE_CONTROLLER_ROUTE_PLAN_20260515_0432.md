# R4 Positive Selectivity Pressure-Controller Route Plan

Timestamp UTC: `2026-05-15T04:32:00Z`

## Decision

Status:
`PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_NO_COMPUTE`.

The teacher-forced pressure-controller route plan is now statically validated.
This is still artifact-only: no Slurm job was submitted, no model scoring was
started, no generation was started, and no training was started.

## Bound Route

Config:
`configs/natural_evidence_v2/r4_positive_selectivity_pressure_controller_route.yaml`

Validator:
`scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py`

Validation summary:
`results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_route_plan_20260515_0432/pressure_controller_route_plan_validation_summary.json`

The future route is:

```text
Qwen only
same-contract a55e
teacher-forced scoring only
no generation
no training
conditions: base, task_only, controlled_protected, wrong_key_controlled, wrong_payload_controlled
score rows: 8192 candidate-v3 prefix-native rows
```

The route plan explicitly requires controller/scorer integration review before
any Slurm submission. The current repository has a pure arithmetic controller
helper, but this plan does not claim that the model scorer has already been
integrated.

## Controller Grid

The future scoring route is bounded to:

```text
bonus_nats: [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
penalty_nats: [0.00, 0.25]
max_target_mass: [0.25, 0.35, 0.45]
max_kl_budget: [0.05, 0.10]
mode: additive
```

The validator rejects target-mass caps above `0.50`, generation unlocks,
missing score rows, candidate-row hash drift, and missing controller helper.

## Future Gate

Before any later generation route can be reviewed, the teacher-forced route
must show:

```text
protected lift vs base >= +0.15
protected lift vs task-only >= +0.10
protected rank1 >= 0.75
protected median margin > 0
wrong-key accepts = 0
wrong-payload accepts = 0
max single-surface mass <= 0.50
target/other overlap rate = 0
scorer boundary failures = 0
```

## Validation

```text
uv run python scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py --output-dir results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_route_plan_20260515_0432
PASS

uv run pytest tests/natural_evidence_v2/test_r4_positive_selectivity_pressure_controller_route.py tests/natural_evidence_v2/test_r4_prefix_native_soft_logit_controller.py
10 passed

uv run python -m py_compile scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py scripts/natural_evidence_v2/r4_prefix_native_soft_logit_controller.py
PASS
```

## Current Blocker

`BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_SCORER_INTEGRATION_NEXT`

## Next Allowed Action

Artifact-only scorer/controller integration review and patch planning. The
model-facing scorer must remain disabled for compute until wrapper review,
local/remote hash preflight, Hermes notification, and exactly-one allowlist
submission gates pass.

No Slurm, model scoring, generation, training, Llama, same-family null,
sanitizer, FAR aggregation, payload-diversity work, or paper-facing positive
claim is unlocked by this plan.
