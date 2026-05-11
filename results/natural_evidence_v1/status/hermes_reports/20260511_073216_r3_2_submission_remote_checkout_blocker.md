# R3.2 Submission Blocker: Remote Checkout Still Not Submission-Safe

timestamp_utc:
2026-05-11T07:32:16Z

phase:
V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED

status:
BLOCK_R3_2_SUBMISSION_REMOTE_CHECKOUT_STILL_NOT_REVIEWED_NO_SLURM

controlling_next_allowed_action:
Enable the existing R3.2 allowlist entry for exactly one reviewed Slurm command
only after the next required TG/email notification path is satisfied, then
submit exactly one allowlisted Chimera Slurm job.

notification_gate:
PASS. The Hermes 2026-05-11 07:31 notification artifact reports Telegram and
email both sent successfully:
`results/natural_evidence_v1/status/hermes_reports/20260511_0731_scheduled_tick_notification.json`.

blocker:
The notification gate is satisfied, but the Chimera submission path remains not
safe or unambiguous. A fresh remote preflight reached `chimerahead.umb.edu` and
confirmed that `~/tokenizer-evidence` exists, but the remote checkout still does
not contain the reviewed R3.2 wrapper/config/current-state files required for
the exact reviewed command:

```text
scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch
scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py
scripts/natural_evidence_v2/aggregate_r3_2_locked_scale_shards.py
configs/natural_evidence_v2/r3_2_qwen_same_contract_locked_scale.yaml
docs/natural_evidence_v2/CURRENT_STATE.md
```

Additional safety issue: the visible local allowlist state still contains an
enabled Llama entry while the current compact state and gate status keep
`llama_allowed=false`. That conflicts with the requirement to enable exactly
one reviewed R3.2 Slurm command under the no-Llama constraint.

state_changing_action:
Recorded this blocker only. No allowlist entry was enabled and no Slurm job was
submitted.

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
