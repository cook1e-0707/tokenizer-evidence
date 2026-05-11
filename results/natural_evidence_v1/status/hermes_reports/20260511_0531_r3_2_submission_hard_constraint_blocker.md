# R3.2 submission hard-constraint blocker

Timestamp UTC: `2026-05-11T05:31:04Z`

Controlling tick:
`results/natural_evidence_v1/status/hermes_reports/20260511_0530_scheduled_tick.md`

## Decision

`BLOCK_R3_2_SUBMISSION_HARD_CONSTRAINT_CONFLICT`

The 05:30 Hermes tick asks Codex to proceed through the approved R3.2 Qwen
locked-scale route and submit exactly one allowlisted Chimera Slurm job, but
the same tick also keeps these hard constraints active:

```text
no generation
no Qwen E2E rerun
```

The reviewed R3.2 wrapper is still plan-only:

```text
scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
```

It exits unless `VALIDATE_PLAN_ONLY=1`, and the recorded wrapper review says
the non-plan generation path is intentionally disabled in the current state.
Enabling the allowlist or submitting this wrapper as a full locked-scale eval
would therefore be unsafe and out of order.

## Status

No Slurm job was submitted. No training, generation, Qwen E2E rerun, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing
positive claim was started.

Next allowed action: resolve the R3.2 submission constraint conflict before
allowlist enablement or Slurm submission. A safe later tick must either
explicitly permit the reviewed generation/eval path, or restrict Codex to
artifact-only full-wrapper review without submission.
