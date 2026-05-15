# R4 Pressure-Controller Wrong-Control Repair Plan

Status: `PASS_ARTIFACT_ONLY_WRONG_CONTROL_REPAIR_PATCH_VALIDATED_NO_SUBMISSION`

The scorer has been patched with `--controller-condition-set controller_only_controls`, which emits base/task-only diagnostics plus committed, wrong-key, and wrong-payload controller arms on the base model. The protected adapter is not loaded for controller arms.

This directly addresses the `859672` failure mode: wrong controls passed because they inherited protected adapter pressure while the scorer measured committed target ids.

Validation:

```text
uv run pytest tests/natural_evidence_v2/test_r4_surface_teacher_forced_controller_integration.py -q
12 passed, 2 skipped

uv run python -m py_compile scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py
passed
```

Next allowed action:

```text
artifact-only controller-only route config/wrapper review and plan-only validation
```

No Slurm, scoring, generation, training, Llama, FAR, sanitizer, payload-diversity work, or paper-facing claim was started.

