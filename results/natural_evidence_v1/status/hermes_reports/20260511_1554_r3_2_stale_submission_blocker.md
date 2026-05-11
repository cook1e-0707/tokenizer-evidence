# R3.2 Stale Submission Blocker: 2026-05-11T15:54Z

`BLOCK_R3_2_STALE_SUBMISSION_REQUEST_JOB_853070_ALREADY_SUBMITTED`

## Reason

The controlling Hermes report for this worker requested the R3.2-B single-job
submission action from `20260511_1553_scheduled_tick.md`.

The compact current state read at worker start is newer and records that R3.2-B
has already been submitted exactly once:

- job id: `853070`
- submitted entry: `v2_r3_2_qwen_locked_scale_eval`
- submitted command:
  `sbatch scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`
- submission record:
  `results/natural_evidence_v2/status/r3_2b_submission_record.json`
- allowlist state after submission: zero enabled entries

Submitting again from the stale instruction would violate the one-Slurm-job
limit and the compact state's next allowed action.

## Action Taken

Recorded this blocker only. No allowlist entry was enabled, no Slurm job was
submitted, no generation or Qwen E2E rerun was started, and no gate-controlled
training, Llama, same-family null, sanitizer, FAR aggregation, or paper-facing
claim work was started.

## Next Allowed Action

Monitor Slurm job `853070`. After completion, sync and review R3.2 locked-scale
artifacts. Do not submit another R3.2 job unless this job fails before producing
reviewable artifacts and a new recovery route is explicitly recorded.
