# Hermes/Codex progress report: WP3 restricted density wrapper ready

## Status

Codex prepared the execution path for the restricted Step-label model-output
density audit.

## Artifacts

```text
scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py
scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_DENSITY_WRAPPER_REVIEW.md
configs/natural_evidence_v2/run_allowlist.yaml
```

## Validation

Local no-model validation passed:

```text
python3 -m py_compile scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py
bash -n scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch
python3 scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py --validate-plan-only
```

The allowlist entry is recorded but disabled:

```text
v2_wp3_restricted_step_label_density_audit
enabled=false
```

## Safety

No Slurm job, model generation, model scoring, training, Qwen E2E, Llama,
same-family null, sanitizer, FAR aggregation, or positive paper claim was
started.

## Next allowed action

Review the wrapper. If approved, enable exactly one allowlist entry and submit
exactly one Chimera Slurm job for base-Qwen model-output density audit over the
256 restricted 16-step prompts. WP4 and training remain blocked.
