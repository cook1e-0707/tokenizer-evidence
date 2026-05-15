# Hermes Sync: R4 Selectivity Pressure-Pivot Package

phase:
`V2_R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_PACKAGE_STATIC_VALIDATION_PASS_NO_COMPUTE`

blocker:
`BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_ROUTE_SELECTION_NEXT`

summary:

```text
Codex recorded the post-859491 repair/pivot route and statically validated the
pressure/selectivity pivot package.

Key facts:
- 859491 remains a failed diagnostic, not a positive result.
- protected accepts were 0/32 under format_scrub=all and 0/32 with no scrub.
- raw/task-only/wrong-key/wrong-payload were also 0/32.
- support-window events were present but not protected-selective:
  protected mean events 9.875, raw 9.375, task-only 8.5625.
- raw max keyed score 23 exceeded protected max keyed score 16.
- unchanged 859491 resubmission is forbidden.
- 859491 transcript mining, threshold relaxation, key/payload remapping, and
  positive reclassification are forbidden.

New artifacts:
- docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_859491_REPAIR_PIVOT_ROUTE_20260515_0412.md
- results/natural_evidence_v2/status/r4_positive_selectivity_859491_repair_pivot_route_20260515_0412/
- configs/natural_evidence_v2/r4_positive_selectivity_pressure_pivot_package.yaml
- scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_pivot_package.py
- tests/natural_evidence_v2/test_r4_positive_selectivity_pressure_pivot_package.py
- docs/natural_evidence_v2/R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_PACKAGE_STATIC_VALIDATION_20260515_0418.md
- results/natural_evidence_v2/status/r4_positive_selectivity_pressure_pivot_package_validation_20260515_0418/

Validation:
- pressure-pivot package static validator: PASS
- focused pytest: 4 passed
- py_compile via uv: PASS
- local allowlist safety: PASS with zero enabled entries
```

next_allowed_action:

```text
Artifact-only route selection between:
1. teacher-forced protected-pressure / soft-controller scoring route;
2. metric-exact objective repair route;
3. explicit stop record.
```

not_unlocked_by_current_state:

```text
Slurm submission
free generation
model scoring
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
