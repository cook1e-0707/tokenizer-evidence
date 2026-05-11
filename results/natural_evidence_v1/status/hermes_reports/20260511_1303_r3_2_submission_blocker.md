# R3.2 Submission Blocker

timestamp_utc: 2026-05-11T13:04:06Z

phase:
`V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED`

requested_next_action:
Enable the existing R3.2 allowlist entry for exactly one reviewed Slurm command
after the required TG/email notification path is satisfied, then submit exactly
one allowlisted Chimera Slurm job.

notification_status:
`SENT_ALL_REQUIRED_CHANNELS`

blocker_status:
`BLOCK_R3_2_SUBMISSION_LOCAL_ALLOWLIST_UNSAFE_NO_SLURM`

blocking_reason:
The 2026-05-11T13:03 Hermes notification path passed, but the submission
preflight is still unsafe. The local allowlist has
`llama_v2_wp6_e2e_eval` enabled while `llama_allowed=false`, and the reviewed
R3.2 entry `v2_r3_2_qwen_locked_scale_eval` remains disabled. Enabling R3.2 and
submitting Slurm from this state would not satisfy the exactly-one reviewed
Qwen R3.2 allowlist route.

local_enabled_allowlist_entries:
- `v2_wp3_fixed_artifact_audit`
- `llama_v2_wp6_e2e_eval`

state_changing_action:
Recorded blocker only; no allowlist enablement, no Slurm submission, no
generation, no Qwen E2E rerun, no training, no Llama, no sanitizer benchmark,
no FAR aggregation, and no paper-facing positive claim.

next_allowed_action:
Repair the submission preflight without running CPU/GPU work: disable the
forbidden local Llama allowlist entry and verify no local or remote forbidden
Llama/sanitizer/FAR entries are enabled. A later notified tick may then enable
only `v2_r3_2_qwen_locked_scale_eval`, submit exactly one reviewed Chimera
Slurm job, and disable the entry after submission.
