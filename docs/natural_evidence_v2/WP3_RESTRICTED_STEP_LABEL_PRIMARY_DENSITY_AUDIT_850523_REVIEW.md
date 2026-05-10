# WP3 Restricted Step-Label Primary Density Audit 850523 Review

## Scope

Slurm job `850523` ran the selected primary-policy strict Step-label
model-output density audit under base Qwen. This generated base-model outputs
only for WP3 density auditing. It did not train, run E2E, decode payloads,
aggregate FAR, or make a paper-facing positive claim.

Artifacts:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_primary_density_audit_850523/
```

## Slurm Result

```text
job_id=850523
job_name=nat-ev-v2-wp3dens
partition=DGXA100
node=chimera13
state=COMPLETED
exit_code=0:0
runtime=00:09:59
```

The wrapper used the Chimera virtual environment:

```text
/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3
```

## Summary

```text
status=FAIL_RESTRICTED_STEP_LABEL_MODEL_OUTPUT_DENSITY_STRUCTURAL_GATE
structural_density_gate_status=FAIL
total_responses=256
complete_step_label_response_count=255
complete_step_label_response_rate=0.99609375
responses_with_at_least_16_structural_slots_count=255
responses_with_at_least_16_structural_slots_rate=0.99609375
mean_detected_structural_slots_per_response=15.94140625
median_detected_structural_slots_per_response=16.0
detected_slot_rows=4081
forbidden_public_surface_rate=0.0
raw_bank_surface_exact_hit_rate=0.07718696397941681
wp4_allowed=false
```

The selected primary bank produced balanced accidental surface observations:

```text
step_label_recombined_create_develop_vs_choose_make_v1::0 = 159
step_label_recombined_create_develop_vs_choose_make_v1::1 = 156
```

Surface counts:

```text
Create=122
Develop=37
Choose=127
Make=29
```

These counts are a report-only raw accidental-surface diagnostic. They are not
ownership evidence, payload recovery, or FAR.

## Failure Mode

The density gate failed because exactly one response was not detected as a
complete line-start `Step 1:` through `Step 16:` response:

```text
prompt_id=qwen_v2_wp3_repair_density_826ed8adbcf8eb241f0f
variant_id=strict_compact_step_label_lines
detected_structural_slots=1
missing_expected_step_labels=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
response_id=qwen_v2_wp3_density_response_078ff9d44fbf0a873c94
```

The generated response did include `Step 1:` through `Step 16:`, but placed
them inline in one paragraph rather than one label per line. Under the current
line-start detector this is a structural miss, so the predeclared gate remains
failed.

This is a much narrower blocker than the earlier 850434 result:

```text
850434 complete rate = 253/256 = 0.98828125
850523 complete rate = 255/256 = 0.99609375
```

The remaining failure is concentrated in the `strict_compact_step_label_lines`
prompt variant, not in the primary bank mass or tokenizer stability.

## Decision

WP3 overall remains blocked because the strict density structural gate did not
pass. WP4, training, Qwen E2E, Llama, same-family null, sanitizer, FAR
aggregation, and paper-facing positive claims remain forbidden.

Next allowed action:

```text
Artifact-only density repair: remove or rewrite the strict_compact_step_label_lines
variant, or explicitly decide whether sentence-start inline Step labels are
inside the detector contract. Do not submit another Slurm job without review and
explicit approval.
```

## Detector-Contract Decision

2026-05-09T02:57Z decision:

```text
sentence-start inline Step labels are outside the current strict detector gate
```

The 850523 result remains a strict structural density fail. The next repair
should remove or rewrite `strict_compact_step_label_lines` in a fresh
artifact-only density plan rather than reclassify the observed inline response.

Decision record:

```text
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_850523_DETECTOR_CONTRACT_DECISION.md
```
