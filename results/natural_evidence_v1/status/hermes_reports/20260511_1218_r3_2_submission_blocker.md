# R3.2 Submission Blocker: Unsafe Local Allowlist

timestamp_utc: `2026-05-11T12:18:42Z`

phase: `V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED`

controlling_tick:
`results/natural_evidence_v1/status/hermes_reports/20260511_1217_scheduled_tick.md`

notification_json:
`results/natural_evidence_v1/status/hermes_reports/20260511_1217_scheduled_tick_notification.json`

notification_status: `SENT_ALL_REQUIRED_CHANNELS`

## Decision

No R3.2 allowlist entry was enabled and no Chimera Slurm job was submitted.

The 12:17 Hermes tick authorizes enabling exactly one reviewed R3.2 allowlist
entry and submitting exactly one reviewed Chimera Slurm job only after the
notification path is satisfied. The notification path is satisfied, but the
submission preflight is still unsafe because the local allowlist has an enabled
forbidden Llama GPU entry while `llama_allowed=false`.

## Observed Enabled Local Allowlist Entries

- `v2_wp3_fixed_artifact_audit`
- `llama_v2_wp6_e2e_eval`

## Blocking Condition

- `llama_v2_wp6_e2e_eval` is enabled locally.
- `llama_allowed=false`.
- R3.2 submission requires enabling only
  `v2_r3_2_qwen_locked_scale_eval` for exactly one reviewed command.

## State-Changing Action

Recorded this blocker only. No training, Llama work, same-family null,
sanitizer benchmark, FAR aggregation, paper-facing positive claim, generation,
Qwen E2E rerun, allowlist enablement, or Slurm submission was started.

## Next Allowed Action

Repair the submission preflight without running CPU/GPU work: disable the
forbidden local Llama allowlist entry and verify no local or remote forbidden
Llama/sanitizer/FAR entries are enabled. A later notified tick may then enable
only `v2_r3_2_qwen_locked_scale_eval`, submit exactly one reviewed Chimera
Slurm job, and disable the entry after submission.
