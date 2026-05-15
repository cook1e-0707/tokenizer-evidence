# R4 Pressure-Controller Wrong-Control Repair Plan

Date: 2026-05-15

Status: `PASS_ARTIFACT_ONLY_WRONG_CONTROL_REPAIR_PATCH_VALIDATED_NO_SUBMISSION`

## Motivation

Job `859672` showed that the soft controller can raise committed target mass, but it failed keyed selectivity: wrong-key and wrong-payload controlled arms also passed the basic teacher-forced mass/rank gate.

The artifact-only diagnosis found that the wrong-control mappings were present and disjoint, but all controller arms still loaded the protected adapter. Because the protected adapter already pushes committed target ids, wrong-key and wrong-payload controller pressure did not make the committed verifier reject.

## Repair

The scorer now supports a new controller condition set:

```text
--controller-condition-set controller_only_controls
```

This condition set emits:

```text
base
task_only
controlled_base
wrong_key_controlled_base
wrong_payload_controlled_base
```

Semantics:

- `controlled_base`: base Qwen plus committed-key controller pressure.
- `wrong_key_controlled_base`: base Qwen plus deterministic wrong-key controller pressure.
- `wrong_payload_controlled_base`: base Qwen plus complement/wrong-payload controller pressure.
- `task_only`: unchanged task-only adapter diagnostic arm.
- No controller arm loads the protected adapter.

This turns the next pressure-controller scoring route into a provider-side keyed controller selectivity test, rather than a protected-adapter-plus-controller test. It preserves the same committed target scorer while removing the adapter bias that invalidated the wrong controls in `859672`.

The old `pressure_controls` condition set remains available only as the historical `859672` diagnostic path.

## Validation

Local artifact-only validation completed:

```text
uv run pytest tests/natural_evidence_v2/test_r4_surface_teacher_forced_controller_integration.py -q
uv run python -m py_compile scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py
```

Result:

```text
14 tests passed or skipped as expected: 12 passed, 2 torch-dependent skipped
py_compile passed
```

The tests verify:

- `controller_only_controls` requires enabled controller mode.
- `controller_only_controls` requires task-only adapter for the diagnostic arm.
- controller arms do not load the protected adapter.
- controller applies to the new controller-only condition names.
- summary output reports controller-only selectivity without requiring a `protected` condition.

## Gate State

No Slurm job was submitted. No allowlist entry was enabled. No model scoring, generation, training, Llama, sanitizer, FAR, payload-diversity work, or paper-facing claim was started.

Next allowed action:

```text
artifact-only controller-only route config/wrapper review and plan-only validation
```

Not unlocked by this patch:

```text
H200 scoring submission
generation
training
Qwen E2E
Llama
same-family null
sanitizer
FAR
payload-diversity claim
paper-facing positive claim
```

