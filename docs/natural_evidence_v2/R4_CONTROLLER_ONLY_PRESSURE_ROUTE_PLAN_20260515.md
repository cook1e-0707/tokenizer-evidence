# R4 Controller-Only Pressure Route Plan

Date: 2026-05-15

Status: `PASS_R4_CONTROLLER_ONLY_ROUTE_PLAN_LOCAL_VALIDATION_NO_SUBMISSION`

## Purpose

This route repairs the `859672` wrong-control semantics failure by evaluating a provider-side keyed controller on the base model, without loading the protected adapter for controller arms.

It is still Qwen-only, same-contract `a55e`, teacher-forced scoring-only, and does not run generation, training, Llama, FAR, sanitizer, payload-diversity work, or paper-facing claims.

## Config

```text
configs/natural_evidence_v2/r4_positive_selectivity_controller_only_route.yaml
```

Condition set:

```text
controller_only_controls
```

Conditions:

```text
base
task_only
controlled_base
wrong_key_controlled_base
wrong_payload_controlled_base
```

The controller arms have `adapter_path = None`; `task_only` remains a diagnostic adapter arm.

## Wrapper

Existing wrapper:

```text
scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

New reviewed environment controls:

```text
CONTROLLER_CONDITION_SET=controller_only_controls
ROUTE_CONFIG=configs/natural_evidence_v2/r4_positive_selectivity_controller_only_route.yaml
```

Future allowlist entry remains disabled:

```text
v2_r4_positive_selectivity_controller_only_score_h200
```

Future command pattern:

```text
sbatch --export=ALL,ALLOW_PRESSURE_CONTROLLER_SCORING=1,CONTROLLER_CONDITION_SET=controller_only_controls,ROUTE_CONFIG=configs/natural_evidence_v2/r4_positive_selectivity_controller_only_route.yaml scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

## Local Validation

Commands run:

```text
uv run pytest tests/natural_evidence_v2/test_r4_surface_teacher_forced_controller_integration.py tests/natural_evidence_v2/test_r4_positive_selectivity_pressure_controller_route.py -q
uv run python -m py_compile scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py
REPO_HOME=/Users/guanjie/Documents/tokenizer_alignment RUN_ROOT=results/natural_evidence_v2/qwen_micro_slot_pilot VALIDATE_PLAN_ONLY=1 CONTROLLER_CONDITION_SET=controller_only_controls ROUTE_CONFIG=configs/natural_evidence_v2/r4_positive_selectivity_controller_only_route.yaml OUTPUT_DIR=results/natural_evidence_v2/status/r4_controller_only_wrapper_plan_smoke_20260515 GRID_INDEX=0 bash scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

Results:

```text
pytest: 19 passed, 2 skipped
py_compile: pass
route validator: PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_NO_COMPUTE
wrapper plan-only: PASS_R4_PRESSURE_CONTROLLER_SCORING_WRAPPER_PLAN_ONLY
```

Wrapper plan-only summary:

```text
results/natural_evidence_v2/status/r4_controller_only_wrapper_plan_smoke_20260515/grid_00/wrapper_plan_only_summary.json
```

## Next Allowed Action

Artifact-only remote sync and remote plan-only preflight for the controller-only route.

Before any H200 scoring submission, the route still requires:

```text
local/remote hash preflight
remote wrapper plan-only validation
remote zero-enabled allowlist safety
active-job preflight
Hermes TG/email pre-submit notification
exactly one allowlist entry enabled
immediate allowlist disablement after sbatch returns
```

No Slurm scoring submission is unlocked by this local plan-only review alone.

