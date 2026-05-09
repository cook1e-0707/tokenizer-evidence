# Hermes/Codex progress report

## Event

Reviewed WP3 restricted Step-label density audit job `850434` and prepared the
next artifact-only repair plan.

## Slurm status

```text
job_id=850434
job_name=nat-ev-v2-wp3dens
state=COMPLETED
exit_code=0:0
runtime=00:09:46
node=chimera13
```

## Result

The job completed successfully at the Slurm level but failed the WP3 structural
density business gate.

```text
complete_step_label_response_rate=0.98828125
detected_slot_rows=4048
mean_detected_structural_slots_per_response=15.8125
raw_bank_surface_exact_hit_rate=0.19095849802371542
wp4_allowed=false
```

Codex repaired the forbidden-surface matcher because the original substring
matcher counted ordinary words such as `certified` and `ownership` as old-route
`CERT`/`OWNER` markers. Re-auditing the same model outputs with the repaired
matcher gives:

```text
forbidden_public_surface_rate=0.0
structural_density_gate_status=FAIL
```

The remaining blocker is strict density: `253/256` responses had all sixteen
Step labels, so the maximum-slot mean is `15.8125 < 16.0`. The restricted
two-bank action surface also covers only `773/4048` detected slots, so expanded
action-verb bank candidates are needed before a future target-mass gate.

## Artifacts

```text
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_DENSITY_AUDIT_850434_REVIEW.md
results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_850434/
results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_850434_reclassified_20260508_2134/
scripts/natural_evidence_v2/build_wp3_restricted_step_label_repair_plan.py
results/natural_evidence_v2/status/wp3_restricted_step_label_repair_plan_20260508_2134/
```

## Next allowed action

Review the artifact-only repair plan. If approved, prepare a Slurm-only
tokenizer/context-mass scoring plan for the expanded Step-label action-verb
candidate banks.

Still forbidden: WP4, training, Qwen E2E, Llama, same-family null, sanitizer,
FAR aggregation, and positive paper claims.
