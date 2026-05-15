# R4 Positive Selectivity Pressure-Pivot Package Static Validation

Timestamp UTC: `2026-05-15T04:18:00Z`

## Decision

Status:
`PASS_R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_PACKAGE_STATIC_VALIDATION_NO_COMPUTE`.

The pressure/selectivity pivot package is now statically validated. This is an
artifact-only package; it does not submit Slurm, start generation, start model
scoring, start training, or unlock paper-facing claims.

## Inputs Bound

Config:
`configs/natural_evidence_v2/r4_positive_selectivity_pressure_pivot_package.yaml`

Validator:
`scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_pivot_package.py`

Validation summary:
`results/natural_evidence_v2/status/r4_positive_selectivity_pressure_pivot_package_validation_20260515_0418/pressure_pivot_package_validation_summary.json`

The package binds the four relevant failed diagnostics:

| Job | Role |
| --- | --- |
| `857795` | pressure-relaxation-B free-generation failure |
| `858019` | transfer-gap prompt repair failure |
| `859277` | phrase-event bank zero-support failure |
| `859491` | selectivity prompt-policy common-support failure |

## Validation Checks

The validator confirms:

- all source review and failure-analysis artifacts exist;
- current permissions keep Slurm, generation, model scoring, training, Llama,
  null/FAR, sanitizer, payload-diversity, and paper claims disabled;
- `859491` reuse policy forbids post-hoc phrase mining, threshold tuning,
  key/payload remapping, decoder relaxation, and unchanged resubmission;
- candidate future routes are artifact-only design entries until later review;
- any future compute remains bound to H200/pomplun policy, exactly-one
  allowlist enablement, and immediate allowlist disablement after submission.

Focused validation:

```text
uv run pytest tests/natural_evidence_v2/test_r4_positive_selectivity_pressure_pivot_package.py
4 passed
```

Py-compile:

```text
uv run python -m py_compile scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_pivot_package.py
PASS
```

## Current Blocker

`BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_ROUTE_SELECTION_NEXT`

## Next Allowed Action

Artifact-only route selection between:

1. teacher-forced protected-pressure / soft-controller scoring route;
2. metric-exact objective repair route;
3. explicit stop record.

No Slurm, generation, model scoring, training, Llama, same-family null,
sanitizer, FAR aggregation, payload-diversity work, or paper-facing positive
claim is unlocked by this validation.
