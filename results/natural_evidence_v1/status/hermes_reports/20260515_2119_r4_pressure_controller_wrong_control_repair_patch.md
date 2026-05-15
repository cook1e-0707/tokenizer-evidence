# Hermes Sync: R4 Wrong-Control Repair Patch Validated

Phase:

```text
V2_R4_PRESSURE_CONTROLLER_WRONG_CONTROL_REPAIR_PATCH_VALIDATED_NO_SUBMISSION
```

Blocker:

```text
BLOCK_R4_PRESSURE_CONTROLLER_CONTROLLER_ONLY_ROUTE_CONFIG_WRAPPER_REVIEW_NEXT
```

Summary:

- `859672` remains a failed diagnostic: wrong-key and wrong-payload controls passed, so no positive claim or generation unlock.
- The scorer now supports `--controller-condition-set controller_only_controls`.
- New conditions are `base`, `task_only`, `controlled_base`, `wrong_key_controlled_base`, and `wrong_payload_controlled_base`.
- Controller arms do not load the protected adapter.
- Focused validation passed: `12 passed, 2 skipped`; scorer `py_compile` passed.

Artifacts:

```text
docs/natural_evidence_v2/R4_PRESSURE_CONTROLLER_WRONG_CONTROL_REPAIR_PLAN_20260515.md
results/natural_evidence_v2/status/r4_pressure_controller_wrong_control_repair_plan_20260515/
docs/natural_evidence_v2/CURRENT_STATE.md
```

Next allowed action:

```text
Artifact-only controller-only route config/wrapper review and plan-only validation.
```

No H200 scoring, generation, training, Llama, FAR, sanitizer, payload-diversity work, or paper-facing claim is unlocked yet.

