# WP3 Restricted Step-Label Density Wrapper Review

## Scope

This review covers the execution wrapper for the restricted 16-step
model-output density audit. It is not training, not Qwen proof-of-life E2E, not
payload recovery, not FAR, and not a paper-facing positive claim.

## Added Entrypoints

```text
scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py
scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch
```

The Slurm wrapper is also recorded in:

```text
configs/natural_evidence_v2/run_allowlist.yaml
```

The allowlist entry is intentionally disabled:

```text
name: v2_wp3_restricted_step_label_density_audit
enabled: false
enable_condition: pending_review_of_restricted_step_label_density_wrapper
```

## What The Job Will Do

When explicitly reviewed and enabled, the wrapper will run on Chimera Slurm and:

1. read the 256 planned prompts from the restricted density plan;
2. load base `Qwen/Qwen2.5-7B-Instruct`;
3. generate one deterministic response per prompt;
4. detect `Step 1:` through `Step 16:` structural anchors;
5. report complete-step-label response rate, structural slot density,
   forbidden-surface rate, and raw accidental candidate-surface hits;
6. export 32 examples for manual naturalness review.

Raw candidate-surface hits are report-only null-risk diagnostics. They are not
ownership evidence.

## Output Contract

The future Slurm job writes:

```text
restricted_step_label_model_outputs.jsonl
restricted_step_label_response_audit.jsonl
restricted_step_label_detected_slots.jsonl
restricted_step_label_naturalness_examples.jsonl
restricted_step_label_density_audit_summary.json
```

The summary keeps:

```text
training_started=false
e2e_eval_started=false
wp4_allowed=false
paper_claim_allowed=false
not_payload_recovery=true
not_full_far=true
```

## Gate Interpretation

The script can pass or fail the structural density gate, but it cannot by itself
unlock WP4. Even a structural pass still requires manual naturalness review and
a separate prompt-local payload contract with decoder-oracle substitution.

## Validation

Local validation is limited to no-model checks:

```text
python3 -m py_compile scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py
bash -n scripts/natural_evidence_v2/slurm/wp3_restricted_step_label_density_audit.sbatch
python3 scripts/natural_evidence_v2/run_wp3_restricted_step_label_density_audit.py --validate-plan-only
```

Do not generate model outputs on the local machine or a Chimera login node.

## Next Allowed Action

Review this wrapper. If approved, enable exactly one allowlist entry and submit
exactly one Chimera Slurm job for the restricted Step-label model-output density
audit. Do not start WP4 or training.
