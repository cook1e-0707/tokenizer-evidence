# Hermes sync: R4 surface-mass failure diagnosis complete

phase:
V2_R4_SURFACE_BANK_REPAIR_DIAGNOSIS_AFTER_853815_FAIL

summary:
Codex completed the allowed artifact-only diagnosis after the failed R4 teacher-forced surface-mass gate from job `853815`.

New artifacts:
- `scripts/natural_evidence_v2/audit_r4_surface_mass_failure_after_853815.py`
- `results/natural_evidence_v2/status/r4_surface_mass_failure_diagnosis_after_853815/surface_mass_failure_diagnosis_summary.json`
- `results/natural_evidence_v2/status/r4_surface_mass_failure_diagnosis_after_853815/surface_mass_failure_diagnosis_report.md`
- `results/natural_evidence_v2/status/r4_surface_mass_failure_diagnosis_after_853815/per_coordinate_surface_mass_lift.csv`
- `results/natural_evidence_v2/status/r4_surface_mass_failure_diagnosis_after_853815/per_target_surface_mass_lift.csv`
- `results/natural_evidence_v2/status/r4_surface_mass_failure_diagnosis_after_853815/per_prefix_shape_mass_lift.csv`

Key findings:
- scored rows: 24,576
- joined base/protected/task-only records: 8,192
- protected mean target mass: 0.0000438295
- protected-vs-base mean lift: -0.0000864096
- protected-vs-task-only mean lift: -0.0002997293
- protected rank-1 rate: 0.4375
- target/other first-token overlap rate: 0.0
- every coordinate has both binary sides in the candidate bank

Interpretation:
The active blocker is not Slurm/provider failure, not the prior one-sided binary-bank formal issue, and not a target/other first-token overlap bug. The selected phrase-level target cylinders are extremely low probability under the R4 prefixes, and the existing protected adapter does not increase their mass.

next_allowed_action:
Artifact-only R4 target-construction / surface-bank / prefix-shape repair design only. Do not submit Slurm, run generation, train, launch Llama, run FAR/sanitizer, or make paper-facing claims from this state.
