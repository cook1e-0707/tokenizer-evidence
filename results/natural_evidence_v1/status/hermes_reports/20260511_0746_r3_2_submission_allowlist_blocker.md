# R3.2 Submission Blocker: Allowlist Not Submission-Safe

timestamp_utc:
2026-05-11T07:46:46Z

phase:
V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED

status:
BLOCK_R3_2_SUBMISSION_ALLOWLIST_UNSAFE_NO_SLURM

controlling_next_allowed_action:
Enable the existing R3.2 allowlist entry for exactly one reviewed Slurm command
only after the next required TG/email notification path is satisfied, then
submit exactly one allowlisted Chimera Slurm job.

notification_gate:
PASS. The Hermes 2026-05-11 07:46 notification artifact reports Telegram and
email both sent successfully:
`results/natural_evidence_v1/status/hermes_reports/20260511_0746_scheduled_tick_notification.json`.

blocker:
The notification gate is satisfied, but the submission path is not safe.
`configs/natural_evidence_v2/run_allowlist.yaml` currently has the forbidden
Llama entry `llama_v2_wp6_e2e_eval` enabled while the compact state and gate
status keep `llama_allowed=false`.

This conflicts with the requirement to enable exactly one reviewed R3.2 command
under the no-Llama constraint. The prior remote-checkout blocker has also not
been superseded by a recorded reconciliation artifact in the compact state.

state_changing_action:
No allowlist entry was enabled and no Slurm job was submitted. This blocker
report records why the controlling action was not safe to execute at this tick.

forbidden_actions_confirmed:
No training, Llama, same-family null, sanitizer benchmark, FAR aggregation,
paper-facing positive claim, generation, Qwen E2E rerun, or Chimera login-node
CPU/GPU work was started by this tick.

next_allowed_action:
Repair the Chimera submission preflight without running CPU/GPU work: reconcile
the remote checkout to the already reviewed R3.2 files and ensure the allowlist
has no enabled forbidden Llama/sanitizer/FAR entries. After that, a later
notified tick may enable only `v2_r3_2_qwen_locked_scale_eval`, submit exactly
one reviewed Slurm job, and disable the entry after submission.
