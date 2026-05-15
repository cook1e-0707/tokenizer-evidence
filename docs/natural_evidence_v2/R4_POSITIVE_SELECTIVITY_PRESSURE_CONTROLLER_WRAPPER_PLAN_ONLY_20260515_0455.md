# R4 Positive Selectivity Pressure-Controller Wrapper Plan-Only Review

Timestamp UTC: `2026-05-15T04:55:00Z`

## Decision

Status:
`PASS_R4_PRESSURE_CONTROLLER_SCORING_WRAPPER_PLAN_ONLY`.

The H200/pomplun teacher-forced pressure-controller scoring wrapper now has a
local plan-only path. No Slurm job was submitted, no model scoring was started,
no generation was started, and no training was started.

## Wrapper

Wrapper:
`scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch`

Future disabled allowlist entry:
`v2_r4_positive_selectivity_pressure_controller_score_h200`

The wrapper is bound to:

```text
partition=pomplun
qos=pomplun
account=cs_yinxin.wan
gres=gpu:h200:1
time=30-00:00:00
array=0-71%4
score rows=8192 candidate-v3 prefix-native rows
same-contract a55e
teacher-forced scoring only
```

The array grid is:

```text
bonus_nats: [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
penalty_nats: [0.00, 0.25]
max_target_mass: [0.25, 0.35, 0.45]
max_kl_budget: [0.05, 0.10]
```

Plan-only validation for `grid_00` ran:

```text
VALIDATE_PLAN_ONLY=1 ... bash scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

Output:
`results/natural_evidence_v2/status/r4_pressure_controller_wrapper_plan_smoke_20260515_0455/grid_00/wrapper_plan_only_summary.json`

The plan-only wrapper:

```text
validates the route config
py-compiles scorer/controller/validator code
runs scorer dry-run with controller enabled
records no model scoring
records no generation
records no training
records no Slurm submission
```

## Full Mode Status

Full scoring mode remains fail-closed:

```text
R4_PRESSURE_CONTROLLER_FULL_SCORING_REQUIRES_WRONG_CONTROL_WRAPPER_REVIEW
```

This is intentional. The scorer integration currently supports controlled
protected pressure over the reviewed target token ids, but the full route still
needs a reviewed wrong-key / wrong-payload controlled mapping before an actual
H200 scoring submission can be safe.

## Validation

```text
bash -n scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
PASS

plan-only wrapper run
PASS_R4_PRESSURE_CONTROLLER_SCORING_WRAPPER_PLAN_ONLY
```

## Current Blocker

`BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_WRONG_CONTROL_MAPPING_NEXT`

## Next Allowed Action

Artifact-only wrong-key / wrong-payload controller mapping design and wrapper
review. The next patch must define how wrong-key and wrong-payload controlled
conditions are represented without post-hoc key/payload remapping, then update
the wrapper full-mode path and tests.

No Slurm job, model scoring, generation, training, Llama, same-family null,
sanitizer, FAR aggregation, payload-diversity work, or paper-facing positive
claim is unlocked by this plan-only wrapper review.
