# WP3 Restricted Step-Label Density Wrapper Review

timestamp_utc: `2026-05-09T01:12:02Z`

## Decision

Reviewed and approved the single prepared Chimera Slurm wrapper for the
restricted Step-label base-Qwen model-output density audit:

```text
scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch
```

The reviewed runner is:

```text
scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py
```

The wrapper targets the already planned `256` prompts in:

```text
results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_plan_20260508_2055/restricted_step_label_density_audit_prompts.jsonl
```

## Safety Outcome

No Slurm job was submitted. No model outputs were generated. The allowlist entry
remains present but disabled:

```text
v2_wp3_restricted_step_label_density_audit
```

Reason: the controlling Hermes tick permitted wrapper review, but this Codex
request also included a hard `no generation` constraint. Submitting the density
audit would generate base-Qwen outputs, so submission is deferred until a later
explicit generation-permitted tick.

## Validation

Passed no-model checks:

```text
python3 -m py_compile scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py
bash -n scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch
python3 scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py --validate-plan-only
```

Additional checks passed:

```text
allowlist_check=pass
prompt_plan_check=pass rows=256
```

## Guardrails

training_started: `false`
model_generation_started: `false`
model_scoring_started: `false`
slurm_job_submitted: `false`
qwen_e2e_started: `false`
llama_started: `false`
same_family_null_started: `false`
sanitizer_started: `false`
far_aggregation_started: `false`
paper_claim_allowed: `false`
wp4_allowed: `false`

## Next Allowed Action

When a later supervisor tick explicitly permits model-output generation, enable
exactly one allowlist entry and submit exactly one Chimera Slurm job for the
restricted Step-label density audit. Do not start WP4 or training.
