# R3.2 Submission Blocker: 2026-05-11 05:00 UTC

## Decision

Block R3.2 allowlist enablement and Slurm submission for this tick.

The 05:00 Hermes tick asks Codex to proceed through the approved R3.2 Qwen
locked-scale route and submit exactly one allowlisted Chimera Slurm job, but
the same tick also hard-forbids generation and Qwen E2E rerun. The reviewed
R3.2 wrapper remains plan-only and exits unless `VALIDATE_PLAN_ONLY=1`.

Submitting the current wrapper as a full locked-scale generation/eval job would
therefore violate the tick constraints and the wrapper review.

## Inspected State

```text
controlling_tick = results/natural_evidence_v1/status/hermes_reports/20260511_0500_scheduled_tick.md
wrapper = scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
allowlist_entry = v2_r3_2_qwen_locked_scale_eval
allowlist_enabled = false
wrapper_full_eval_enabled = false
wrapper_plan_only = true
```

## Actions Not Taken

No Slurm job was submitted. No allowlist entry was enabled. No training,
generation, Qwen E2E rerun, Llama, same-family null, sanitizer benchmark, FAR
aggregation, or paper-facing positive claim was started.

## Next Safe Action

Resolve the R3.2 submission constraint conflict before allowlist enablement or
Slurm submission. A safe later tick must either explicitly permit the reviewed
generation/eval path, or restrict Codex to artifact-only full-wrapper review
without submission.
