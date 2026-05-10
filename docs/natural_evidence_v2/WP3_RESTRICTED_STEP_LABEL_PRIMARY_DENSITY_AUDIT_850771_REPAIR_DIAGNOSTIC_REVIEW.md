# WP3 Restricted Step-Label Repaired Density Diagnostic 850771 Review

## Scope

Slurm job `850771` ran the user-approved repaired `850523` strict Step-label
model-output density diagnostic under base Qwen. This job used the repaired
192-prompt seed that removed `strict_compact_step_label_lines`.

This review is not a full WP3-R1 gate decision. The job lacks the required
dev `>=512` outputs, separate eval `>=2048` outputs, and eval oracle
prompt-local frame-completion field. It did not start training, protected
transcript generation, Qwen E2E, Llama, same-family null, sanitizer, FAR
aggregation, or any paper-facing positive claim.

Artifacts:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_primary_density_audit_850523_repair_850771/
```

## Slurm Result

```text
job_id=850771
job_name=nat-ev-v2-wp3dens
partition=DGXA100
node=chimera13
state=COMPLETED
exit_code=0:0
runtime=00:06:41
```

The wrapper used the Chimera virtual environment:

```text
/hpcstor6/scratch01/g/guanjie.lin001/venvs/zkrfa_py312/bin/python3
```

## Density Summary

```text
status=PASS_RESTRICTED_STEP_LABEL_MODEL_OUTPUT_DENSITY_STRUCTURAL_GATE_NEEDS_MANUAL_NATURALNESS
structural_density_gate_status=PASS
total_responses=192
complete_step_label_response_count=192
complete_step_label_response_rate=1.0
responses_with_at_least_16_structural_slots_count=192
responses_with_at_least_16_structural_slots_rate=1.0
mean_detected_structural_slots_per_response=16.0
median_detected_structural_slots_per_response=16.0
detected_slot_rows=3072
forbidden_public_surface_rate=0.0
raw_bank_surface_exact_hit_rate=0.072265625
wp4_allowed=false
```

Variant-level structural check:

```text
strict_literal_16_step_lines: 64/64 complete, min=16, mean=16, max=16
strict_no_heading_16_step_lines: 64/64 complete, min=16, mean=16, max=16
strict_numbered_step_label_lines: 64/64 complete, min=16, mean=16, max=16
```

No response audit row had missing expected step labels, fewer than 16 detected
structural slots, or an incomplete strict line-start Step-label sequence.

The primary bank's raw accidental surface counts were balanced but sparse:

```text
step_label_recombined_create_develop_vs_choose_make_v1::0 = 110
step_label_recombined_create_develop_vs_choose_make_v1::1 = 112
```

Surface counts:

```text
Create=89
Develop=21
Choose=96
Make=16
```

These counts are a report-only raw accidental-surface diagnostic. They are not
ownership evidence, payload recovery, or FAR.

## Manual Naturalness Review

The exported manual naturalness file contains 32 examples. All 32 are from
`strict_literal_16_step_lines` because the exporter takes the first examples in
artifact order. The sampled responses read as ordinary procedural checklists
for everyday tasks. No forbidden public surface text was found by the audit.

Manual labels:

```text
PASS=31
BORDERLINE=1
FAIL_FORMAT_OR_TEMPLATE_ARTIFACT=0
FAIL_FORBIDDEN_SURFACE=0
FAIL_SEMANTIC_COHERENCE=0
PASS+BORDERLINE rate=1.0
```

The single `BORDERLINE` label is for a minor duplicated emergency-kit item in
one otherwise coherent checklist. Spot checks from the two other retained
variants also read as ordinary checklist prose, but a future full WP3-R1/R3
expansion should export a variant-balanced manual sample across dev and eval.

Manual naturalness diagnostic status:

```text
PASS_DIAGNOSTIC_SAMPLE_ONLY
```

## Decision

Job `850771` passes the strict line-start structural density diagnostic for the
192-prompt repaired seed, and the sampled manual naturalness review passes for
diagnostic purposes.

It does not unlock WP4 or training. WP3 remains blocked because:

- WP3-R1 still requires dev `>=512` and eval `>=2048` model-output density
  expansion with eval oracle prompt-local frame completion recorded;
- WP3-R2 still requires a high-mass 2-way bank search, and the current primary
  bank has `min_bucket_mass=0.0125512375`, below the pilot threshold `0.03`;
- WP3-R3 should be repeated as a formal variant-balanced manual naturalness
  review on the expanded dev/eval artifacts.

Next allowed action:

```text
Prepare an artifact-only WP3-R1 strict density expansion plan with dev>=512,
eval>=2048, the strict line-start detector, and eval oracle prompt-local frame
completion recorded. Do not submit Slurm without explicit approval.
```
