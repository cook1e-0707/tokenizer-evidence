# R3.2 submission blocker: 2026-05-11 14:04Z

Status: `BLOCK_R3_2_SUBMISSION_ALLOWLIST_UNSAFE_NO_SLURM`

The 2026-05-11T14:03 Hermes notification path succeeded:

- Telegram: sent
- Email: sent
- Notification JSON:
  `results/natural_evidence_v1/status/hermes_reports/20260511_1403_scheduled_tick_notification.json`

Submission is still not safe or unambiguous. The local allowlist has forbidden
`llama_v2_wp6_e2e_eval` enabled while Llama remains blocked by the current
route, and the reviewed R3.2 entry remains disabled:

- forbidden enabled entry: `llama_v2_wp6_e2e_eval`
- reviewed R3.2 entry: `v2_r3_2_qwen_locked_scale_eval`
- reviewed R3.2 command:
  `sbatch scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`

No allowlist entry was changed and no Slurm job was submitted. No training,
Llama, same-family null, sanitizer benchmark, FAR aggregation, paper-facing
positive claim, generation, Qwen E2E rerun, or Chimera login-node CPU/GPU work
was started.

Next safe action: repair the submission preflight without running CPU/GPU work
by disabling the forbidden local Llama allowlist entry and verifying no local or
remote forbidden Llama/sanitizer/FAR entries are enabled. A later notified tick
may then enable only `v2_r3_2_qwen_locked_scale_eval`, submit exactly one
reviewed Chimera Slurm job, and disable the entry after submission.
