# Hermes/Codex conditional gate policy sync

## Current phase

`V2_R3_2B_QWEN_LOCKED_SCALE_SINGLE_JOB_SUBMISSION_READY`

## State update

The wording for later-stage work classes has been synchronized:

- training
- Llama
- same-family null
- sanitizer benchmark
- FAR aggregation
- paper-facing positive claims

These classes are not permanently forbidden. They are conditionally authorized
future actions, but each remains gate-locked until its prerequisite evidence is
recorded, the corresponding gate boolean is true, and the current
`next_allowed_action` explicitly names that class.

## Gate booleans

No gate boolean was unlocked in this sync. Current values remain:

- `training_allowed=false`
- `llama_allowed=false`
- `same_family_null_allowed=false`
- `sanitizer_allowed=false`
- `far_aggregation_allowed=false`
- `paper_claim_allowed=false`

## Actions executed

- Updated Hermes coordination wording from permanent/absolute forbidden wording
  to gate-controlled wording.
- Updated the compact current-state file timestamp and policy wording.
- Updated v1/v2 `gate_status.json` with
  `conditional_authorization_policy`.
- Kept the R3.2-B next allowed action unchanged.

## Slurm

No Slurm job was submitted. No allowlist entry was enabled.

## Next allowed action

R3.2-B submission preflight may proceed in a later notified tick: enable exactly
`v2_r3_2_qwen_locked_scale_eval`, verify it is the only enabled entry and the
same-contract `a55e` preflight/replay gates still pass, submit exactly one
Chimera Slurm job, then immediately disable the entry after `sbatch` returns.
