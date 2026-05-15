# Hermes Sync: R4 Pressure-Controller Route Plan

phase:
`V2_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_PASS_NO_COMPUTE`

blocker:
`BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_SCORER_INTEGRATION_NEXT`

summary:

```text
Codex continued from the reviewed 859491 failure and completed the artifact-only
route chain:

1. Recorded 859491 repair/pivot route.
2. Validated pressure/selectivity pivot package.
3. Selected teacher-forced protected-pressure / soft-controller scoring route.
4. Implemented and statically validated its route plan.

No Slurm job was submitted.
No allowlist entry was enabled.
No model scoring, generation, training, Llama, null/FAR, sanitizer, payload
diversity, or paper-claim action started.
```

new_route_artifacts:

```text
docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_859491_REPAIR_PIVOT_ROUTE_20260515_0412.md
docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_PACKAGE_STATIC_VALIDATION_20260515_0418.md
docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_SELECTION_20260515_0424.md
docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_20260515_0432.md
configs/natural_evidence_v2/r4_positive_selectivity_pressure_pivot_package.yaml
configs/natural_evidence_v2/r4_positive_selectivity_pressure_controller_route.yaml
scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_pivot_package.py
scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py
tests/natural_evidence_v2/test_r4_positive_selectivity_pressure_pivot_package.py
tests/natural_evidence_v2/test_r4_positive_selectivity_pressure_controller_route.py
```

validation:

```text
pressure-pivot package validator: PASS
pressure-controller route validator: PASS
focused pytest: 10 passed for pressure-controller route + soft-controller helper
py_compile: PASS
JSON syntax: PASS
allowlist safety: PASS with zero enabled entries
```

next_allowed_action:

```text
Artifact-only scorer/controller integration review and patch planning.
```

not_unlocked:

```text
Slurm submission
model scoring
free generation
training
Llama
same-family null
sanitizer
FAR aggregation
payload-diversity work or claim
paper-facing positive claim
```

standing authorization note:

```text
The user has standing authorization for Codex/Hermes to continue when a route's
recorded prerequisites pass. Do not ask for repeated manual approval on the
same clear route. This does not waive route gates, allowlist safety,
Hermes/Codex state sync, Slurm-only Chimera execution, or H200/pomplun policy.
```
