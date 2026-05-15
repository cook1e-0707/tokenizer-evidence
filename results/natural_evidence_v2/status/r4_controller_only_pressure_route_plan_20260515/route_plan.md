# R4 Controller-Only Pressure Route Plan

Status: `PASS_R4_CONTROLLER_ONLY_ROUTE_PLAN_LOCAL_VALIDATION_NO_SUBMISSION`

The controller-only route config and wrapper path passed local artifact-only validation. Controller arms use the base model and do not load the protected adapter. This repairs the `859672` null semantics failure before any future scoring rerun.

Local validation:

```text
pytest: 19 passed, 2 skipped
py_compile: passed
route validator: PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_NO_COMPUTE
wrapper plan-only: PASS_R4_PRESSURE_CONTROLLER_SCORING_WRAPPER_PLAN_ONLY
```

Next allowed action:

```text
artifact-only remote sync and remote plan-only preflight
```

No H200 scoring, generation, training, Llama, FAR, sanitizer, payload-diversity work, or paper-facing claim has been started or unlocked.

