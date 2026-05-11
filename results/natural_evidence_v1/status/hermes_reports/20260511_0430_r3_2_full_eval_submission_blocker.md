# R3.2 Qwen Locked-Scale Submission Blocker

timestamp_utc: 2026-05-11T04:30:37Z

## Decision

Do not update the allowlist and do not submit an R3.2 Chimera Slurm job from
this tick.

## Reason

The controlling R3.2 route asks Codex to proceed toward exactly one reviewed
locked-scale Slurm submission, but the tick-level hard constraints also forbid:

```text
generation
Qwen E2E rerun
```

The reviewed R3.2 wrapper currently recorded in
`docs/natural_evidence_v2/R3_2_QWEN_LOCKED_SCALE_WRAPPER_REVIEW_20260511.md`
is plan-only. Its Slurm wrapper exits unless `VALIDATE_PLAN_ONLY=1`, so it is
not yet a reviewed full locked-scale generation/eval wrapper.

Submitting it now would either submit a known plan-only wrapper as if it were a
full eval, or require enabling the disabled generation path despite this tick's
explicit no-generation constraint.

## State

No Slurm job was submitted. No allowlist entry was enabled. No training,
generation, Qwen E2E rerun, Llama, same-family null, sanitizer benchmark, FAR
aggregation, or paper-facing positive claim was started.

## Next Safe Action

Implement and review the R3.2 full locked-scale generation/eval wrapper as an
artifact/control-plane change only, without running generation locally or on the
Chimera login node. After that review exists, a later tick whose constraints
permit the reviewed full Slurm eval can notify channels, enable the allowlist
for one submission, submit exactly one Slurm job, and disable the allowlist.

