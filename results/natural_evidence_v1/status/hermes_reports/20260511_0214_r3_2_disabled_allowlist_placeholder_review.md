# R3.2 Disabled Allowlist Placeholder Review

Route phase:

```text
V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED
```

Action taken:

```text
Recorded a disabled R3.2 Qwen locked-scale allowlist placeholder only.
```

Allowlist entry:

```text
name = v2_r3_2_qwen_locked_scale_eval
command = sbatch scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
enabled = false
enable_condition = disabled_until_r3_2_wrapper_precommit_and_gate_review_recorded
```

Review finding:

The entry is intentionally not submittable. The R3.2 wrapper implementation,
precommit contract, local plan-only validation, and final gate review remain
missing. The referenced wrapper path is a reserved future path and does not yet
exist.

No Slurm job was submitted. No training, generation, Qwen E2E rerun, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing positive
claim was started.

Next allowed action:

```text
Implement or review the R3.2-specific Qwen locked-scale wrapper and precommit
package, then run local plan-only validation only. Do not submit Slurm until the
wrapper, allowlist, precommit, notification, and gate review are all recorded.
```
