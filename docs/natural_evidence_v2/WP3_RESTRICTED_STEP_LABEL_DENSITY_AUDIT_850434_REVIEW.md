# WP3 Restricted Step-Label Density Audit 850434 Review

## Scope

Slurm job `850434` ran the base-Qwen restricted Step-label model-output density
audit on `256` planned prompts. This was a WP3 density diagnostic only:
no training, no E2E evaluation, no payload recovery, no FAR aggregation, and no
paper-facing positive claim.

Primary artifacts:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_850434/
results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_850434_reclassified_20260508_2134/
```

## Slurm Result

```text
job_id=850434
job_name=nat-ev-v2-wp3dens
partition=DGXA100
node=chimera13
state=COMPLETED
exit_code=0:0
runtime=00:09:46
```

The wrapper completed and wrote all expected artifacts. The business gate failed:

```text
status=FAIL_RESTRICTED_STEP_LABEL_MODEL_OUTPUT_DENSITY_STRUCTURAL_GATE
wp4_allowed=false
```

## Main Metrics

Original job summary:

```text
total_responses=256
complete_step_label_response_count=253
complete_step_label_response_rate=0.98828125
responses_with_at_least_16_structural_slots_count=253
responses_with_at_least_16_structural_slots_rate=0.98828125
detected_slot_rows=4048
mean_detected_structural_slots_per_response=15.8125
median_detected_structural_slots_per_response=16.0
forbidden_public_surface_response_count=21
forbidden_public_surface_rate=0.08203125
raw_bank_surface_exact_hit_count=773
raw_bank_surface_exact_hit_rate=0.19095849802371542
```

After re-auditing the same response artifact with the repaired forbidden matcher:

```text
forbidden_public_surface_response_count=0
forbidden_public_surface_rate=0.0
structural_density_gate_status=FAIL
mean_detected_structural_slots_per_response=15.8125
wp4_allowed=false
```

The reclassification shows that the original forbidden-surface failure was a
detector bug, not an actual old-route public marker recurrence. The old matcher
used case-insensitive substring matching and counted ordinary words such as
`certified`, `certificates`, and `ownership` as `CERT`/`OWNER`. The wrapper now
matches explicit old-route markers as whole words and keeps `fingerprint` as a
case-insensitive whole-word forbidden term.

## Interpretation

The Step-label structure is close but not yet gate-passing. Base Qwen produced
complete `Step 1:` through `Step 16:` responses for `253/256` prompts. The
three misses were unlabeled checklist-style prose, usually with only fifteen
sentences. Because the selected route has a maximum of sixteen structural slots
per response, any missing response pulls the mean below the current
`>=16.0` gate. WP4 therefore remains blocked.

The restricted two-bank action surface is also too narrow for the future
trainable bucket policy. Only `773/4048` detected slots exactly hit the current
bank surfaces. Most observed step openers are ordinary action verbs outside the
two passing banks, especially:

```text
Prepare, Plan, Ensure, Pack, Determine, Define, Bring, Gather, Identify,
Practice, Reflect, Clean, Select, Arrange, Consider, Develop, Schedule, Use,
Take, Send, Decide, Establish, Keep, Document, Confirm, Assign, Include
```

This does not by itself fail the density gate because raw bank hits are a
report-only accidental-surface diagnostic. It does mean the current two-bank
restricted policy is likely too small for the next target-mass gate unless
expanded candidates pass tokenizer and context-specific mass scoring.

## Decision

Do not start WP4, training, Qwen E2E, Llama, same-family nulls, sanitizer,
FAR aggregation, or paper-facing positive claims.

Next allowed action:

1. Build an artifact-only repaired density plan with stricter prompt wording:
   require exactly sixteen lines, each beginning with the literal labels
   `Step 1:` through `Step 16:`.
2. Remove or rewrite prompt variants that produced unlabeled checklist prose.
3. Keep the repaired forbidden-surface matcher and report both exact-marker
   hits and ordinary-word near misses when useful.
4. Build expanded Step-label action-verb candidate banks from observed base-Qwen
   openers.
5. Run tokenizer/context-mass scoring for the expanded banks through Chimera
   Slurm only after wrapper and allowlist review.

Stop condition before WP4 remains unchanged: WP3 must have a model-output
density pass, tokenizer-stable candidate surfaces, and mass-balanced two-way
banks. The 850434 artifacts are useful evidence that controlled Step-label
density is feasible, but they are not a WP3 pass.
