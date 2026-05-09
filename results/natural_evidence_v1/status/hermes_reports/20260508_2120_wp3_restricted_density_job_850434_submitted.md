# Hermes/Codex progress report: WP3 restricted density job submitted

## Status

User explicitly permitted the restricted Step-label model-output density audit
submission. Codex submitted exactly one Chimera Slurm job.

## Slurm Job

```text
job_id: 850434
name: nat-ev-v2-wp3dens
partition: DGXA100
gres: gpu:A100:1
state_at_submission_check: PENDING(Resources)
```

The job runs:

```text
scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch
```

It will generate base-Qwen outputs for the 256 restricted 16-step prompts and
audit structural Step-label density. It is not training, not E2E, not payload
recovery, not FAR, and not a positive paper claim.

## Duplicate-Submission Guard

After the job was submitted, Codex disabled the allowlist entry again:

```text
v2_wp3_restricted_step_label_density_audit
enabled=false
enable_condition=submitted_once_as_job_850434_pending_result_review
```

The disabled allowlist was synced back to Chimera.

## Next Allowed Action

Monitor job `850434`. After completion, sync and review:

```text
restricted_step_label_model_outputs.jsonl
restricted_step_label_response_audit.jsonl
restricted_step_label_detected_slots.jsonl
restricted_step_label_naturalness_examples.jsonl
restricted_step_label_density_audit_summary.json
```

WP4 and training remain blocked.
