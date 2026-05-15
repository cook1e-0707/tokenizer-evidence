# R4 Controller-Only Single Submission Route

Date: 2026-05-15

Status: `READY_R4_CONTROLLER_ONLY_SINGLE_H200_SUBMISSION`

## Scope

Submit exactly one H200/pomplun Slurm array for the controller-only pressure route.

This is Qwen-only, same-contract `a55e`, teacher-forced scoring-only. It does not run free generation, training, Llama, same-family null, sanitizer, FAR, payload diversity, or paper-facing claims.

## Reviewed Preconditions

Completed:

- `859672` reviewed as failed selectivity diagnostic.
- wrong-control repair patch validated.
- controller-only route config local validation passed.
- controller-only wrapper plan-only validation passed locally.
- reviewed files synchronized to Chimera.
- local/remote hashes matched for reviewed route files.
- remote zero-enabled allowlist safety passed.
- remote wrapper plan-only validation passed.
- active Chimera job preflight observed no active jobs.

Reviewed artifacts:

```text
docs/natural_evidence_v2/R4_PRESSURE_CONTROLLER_SCORE_859672_REVIEW_20260515.md
docs/natural_evidence_v2/R4_PRESSURE_CONTROLLER_WRONG_CONTROL_REPAIR_PLAN_20260515.md
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_PRESSURE_ROUTE_PLAN_20260515.md
results/natural_evidence_v2/status/r4_controller_only_remote_preflight_20260515/remote_preflight_summary.json
```

## Authorized Single Command

Exactly one allowlist entry may be enabled:

```text
v2_r4_positive_selectivity_controller_only_score_h200
```

Authorized command:

```text
sbatch --export=ALL,ALLOW_PRESSURE_CONTROLLER_SCORING=1,CONTROLLER_CONDITION_SET=controller_only_controls,ROUTE_CONFIG=configs/natural_evidence_v2/r4_positive_selectivity_controller_only_route.yaml scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

Postcondition:

```text
The allowlist entry must be disabled immediately after sbatch returns.
```

## Review Gate For Completed Job

The completed scoring job must be reviewed before any generation or further scoring route.

Primary expected summary fields:

```text
controller_only_summary.controlled_basic_gate_pass
controller_only_summary.wrong_key_basic_gate_pass
controller_only_summary.wrong_payload_basic_gate_pass
controller_only_summary.overall_selective_gate_pass
```

Pass condition for a grid:

```text
controlled_basic_gate_pass = true
wrong_key_basic_gate_pass = false
wrong_payload_basic_gate_pass = false
overall_selective_gate_pass = true
```

No downstream route is unlocked until terminal artifacts are synced and reviewed.

