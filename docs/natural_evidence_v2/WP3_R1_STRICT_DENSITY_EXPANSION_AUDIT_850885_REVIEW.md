# WP3-R1 Strict Density Expansion Audit 850885 Review

## Scope

Slurm job `850885` ran the single user-approved WP3-R1 strict Step-label
density expansion audit under base Qwen. This review covers synced artifacts:

```text
results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_audit_850885/
```

This review did not start WP4, training, protected transcript generation, Qwen
E2E, Llama, same-family nulls, sanitizer benchmarks, FAR aggregation, or any
paper-facing positive claim.

## Slurm Result

```text
job_id=850885
job_name=nat-ev-v2-wp3dens
partition=DGXA100
node=chimera12
state=COMPLETED
exit_code=0:0
runtime=01:23:00
remote_output_dir=/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/wp3_r1_strict_density_expansion_audit_20260509_040136
```

The Slurm stdout reports:

```text
status=FAIL_RESTRICTED_STEP_LABEL_MODEL_OUTPUT_DENSITY_STRUCTURAL_GATE
structural_density_gate_status=FAIL
total_responses=2560
wp4_allowed=false
```

The top-level runner status fails because the legacy structural check requires
global mean detected structural slots to be at least `16.0`. One eval response
used Chinese action text after `Step 2:` through `Step 16:`, so the English
first-word slot detector counted only `1` structural slot for that response.

## Split-Level Density Review

Dev split:

```text
total_responses=512
complete_step_label_response_count=512
complete_step_label_response_rate=1.0
responses_with_at_least_16_structural_slots_count=512
responses_with_at_least_16_structural_slots_rate=1.0
oracle_prompt_local_frame_completion_count=512
oracle_prompt_local_frame_completion_rate=1.0
mean_detected_structural_slots_per_response=16.0
median_detected_structural_slots_per_response=16.0
forbidden_public_surface_rate=0.0
```

Eval split:

```text
total_responses=2048
complete_step_label_response_count=2048
complete_step_label_response_rate=1.0
responses_with_at_least_16_structural_slots_count=2047
responses_with_at_least_16_structural_slots_rate=0.99951171875
oracle_prompt_local_frame_completion_count=2047
oracle_prompt_local_frame_completion_rate=0.99951171875
mean_detected_structural_slots_per_response=15.99267578125
median_detected_structural_slots_per_response=16.0
forbidden_public_surface_rate=0.0
```

The split-level thresholds from
`docs/natural_evidence_v2/V2_EXECUTION_STANDARD_AFTER_850523_EXPERT_REVIEW.md`
are met for dev/eval volume, complete response rate, dev mean slots, eval oracle
prompt-local frame completion, and forbidden surface rate. This is a density
review only; it is not payload recovery, FAR, or an ownership claim.

The single eval anomaly is:

```text
response_id=qwen_v2_wp3_density_response_f4ebed8edd46023e3061
prompt_id=qwen_v2_wp3_r1_density_c72aac81a70a5bfc3a97
variant_id=strict_numbered_step_label_lines
detected_structural_slots=1
topic=helping a remote team make a morning handoff smoother while staying calm with emphasis on early planning
```

## Naturalness Examples

The exported naturalness file contains `32` examples. All `32` are from the
`wp3_r1_dev` split and the `strict_literal_16_step_lines` variant.

Manual labels for the exported examples:

```text
PASS=32
BORDERLINE=0
FAIL_FORMAT_OR_TEMPLATE_ARTIFACT=0
FAIL_FORBIDDEN_SURFACE=0
FAIL_SEMANTIC_COHERENCE=0
PASS+BORDERLINE rate=1.0
```

This exported sample passes as a limited sample only. It is not a formal
WP3-R3 gate because the expansion gate requires a variant-balanced manual
sample, and the exported examples do not cover eval or the other two prompt
variants.

## Decision

Job `850885` is completed, synced, and reviewed.

Review status:

```text
REVIEWED_SPLIT_R1_DENSITY_THRESHOLDS_PASS_LEGACY_GLOBAL_STRUCTURAL_STATUS_FAIL_WP4_BLOCKED
```

WP4 remains blocked because WP3-R2 high-mass 2-way bank search and formal
WP3-R3 variant-balanced manual naturalness review remain open. The current
primary bank is still below the pilot absolute mass threshold recorded in the
execution standard.

Next allowed action:

```text
Continue WP3 only: address R2 high-mass 2-way bank search and/or formal
variant-balanced WP3-R3 naturalness review. Do not start WP4, training, Qwen
E2E, Llama, same-family null, sanitizer, FAR aggregation, or paper-facing
positive claims.
```
