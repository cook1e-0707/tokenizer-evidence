# R3.2 submission blocker: 2026-05-11 13:49Z

## Decision

`BLOCK_R3_2_SUBMISSION_LOCAL_ALLOWLIST_UNSAFE_NO_SLURM`

The 2026-05-11 13:48Z Hermes TG/email notification path succeeded, but the
R3.2 Slurm submission is still not safe. The local allowlist still has a
forbidden Llama GPU entry enabled:

```text
name: llama_v2_wp6_e2e_eval
enabled: true
```

Llama remains forbidden in the current phase (`llama_allowed=false`). Enabling
`v2_r3_2_qwen_locked_scale_eval` now would leave more than one GPU route
enabled, including a forbidden route, so exactly-one reviewed-command submission
is not unambiguous.

## Evidence Checked

```text
docs/natural_evidence_v2/CURRENT_STATE.md
results/natural_evidence_v1/status/gate_status.json
results/natural_evidence_v2/status/gate_status.json
results/natural_evidence_v1/status/hermes_reports/20260511_1348_scheduled_tick.md
results/natural_evidence_v1/status/hermes_reports/20260511_1348_scheduled_tick_notification.json
configs/natural_evidence_v2/run_allowlist.yaml
```

Notification status:

```text
SENT_ALL_REQUIRED_CHANNELS
```

## Action Taken

No Slurm job was submitted. No allowlist entry was enabled. No training,
Llama, same-family null, sanitizer benchmark, FAR aggregation, paper-facing
positive claim, generation, Qwen E2E rerun, or login-node CPU/GPU work was
started.

## Next Allowed Action

Repair the submission preflight without running CPU/GPU work: disable the
forbidden local `llama_v2_wp6_e2e_eval` allowlist entry and verify no local or
remote forbidden Llama/sanitizer/FAR entries are enabled. A later notified tick
may then enable only `v2_r3_2_qwen_locked_scale_eval`, submit exactly one
reviewed Chimera Slurm job, and disable the entry after submission.
