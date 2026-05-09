# Hermes progress notification

Codex submitted one Chimera Slurm job for the `natural_evidence_v2` WP3
selected primary-policy strict Step-label density audit.

Submission:

```text
job_id=850523
job_name=nat-ev-v2-wp3dens
partition=DGXA100
initial_state=RUNNING
node=chimera13
```

Prompts:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_primary_policy_density_plan_20260509_0225/restricted_step_label_strict_density_audit_prompts.jsonl
```

Policy directory:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_primary_policy_density_plan_20260509_0225
```

Remote output directory:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/wp3_restricted_step_label_primary_density_audit_20260509_023224
```

The `v2_wp3_restricted_step_label_density_audit` allowlist entry was disabled
immediately after submission:

```text
submitted_once_as_job_850523_pending_primary_policy_strict_density_result_review
```

No training, WP4, Qwen E2E, Llama, same-family null, sanitizer, FAR
aggregation, or paper-facing positive claim was started.

Next allowed action:

```text
Monitor job 850523; after completion, sync and review its density artifacts
before any further step.
```
