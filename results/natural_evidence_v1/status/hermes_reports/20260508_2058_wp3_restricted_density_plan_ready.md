# Hermes/Codex progress report: WP3 restricted density plan ready

## Status

Codex built the artifact-only model-output density audit plan for the restricted
natural_evidence_v2 `Step N:` route.

## Artifacts

```text
scripts/natural_evidence_v2/build_wp3_restricted_step_label_density_audit_plan.py
docs/natural_evidence_v2/WP3_RESTRICTED_STEP_LABEL_DENSITY_AUDIT_PLAN.md
results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_plan_20260508_2055/
```

The plan writes `256` prompts for the selected 16-step route and a gate spec for:

- `Step 1:` through `Step 16:` adherence;
- structural slot density;
- forbidden public surface rate;
- raw accidental bank-surface hit reporting;
- manual naturalness examples.

## Safety

No model generation, model scoring, Slurm job, training, Qwen E2E, Llama,
same-family null, sanitizer, FAR aggregation, or positive paper claim was
started.

## Next allowed action

Review this density audit plan. If approved, implement or review exactly one
Chimera Slurm wrapper for base-Qwen model-output density audit on the 256
planned prompts. WP4 and training remain blocked.
