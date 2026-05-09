# Hermes progress notification

Codex synced and reviewed Slurm job `850523`.

850523 result:

```text
state=COMPLETED
exit_code=0:0
runtime=00:09:59
status=FAIL_RESTRICTED_STEP_LABEL_MODEL_OUTPUT_DENSITY_STRUCTURAL_GATE
total_responses=256
complete_step_label_response_count=255
complete_step_label_response_rate=0.99609375
mean_detected_structural_slots_per_response=15.94140625
median_detected_structural_slots_per_response=16.0
forbidden_public_surface_rate=0.0
wp4_allowed=false
```

Review:

```text
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_PRIMARY_DENSITY_AUDIT_850523_REVIEW.md
```

Failure mode:

```text
variant_id=strict_compact_step_label_lines
detected_structural_slots=1
```

The failed response contained `Step 1:` through `Step 16:` inline in one
paragraph, but the current detector only counts line-start anchors. This is a
close density structural fail, not a mass/tokenizer failure.

No training, WP4, Qwen E2E, Llama, same-family null, sanitizer, FAR
aggregation, or paper-facing positive claim was started.

Next allowed action:

```text
Artifact-only density repair: remove or rewrite strict_compact_step_label_lines,
or explicitly decide whether sentence-start inline Step labels are inside the
detector contract. Do not submit another Slurm job without review and explicit
approval.
```
