# Hermes natural_evidence_v1 scheduler registration

phase:
POST_846699_BRANCH_AWARE_SCORE_INTERPRETATION_COMPLETE

active_jobs:
Hermes gateway is running. `hermes cron status` reports 2 active jobs.

completed_jobs:
No new Slurm or model job was submitted during scheduler registration.

artifact_changes:
- Added `scripts/natural_evidence_v1/hermes_supervision_tick.py`.
- Added `/Users/guanjie/.hermes/scripts/natural_evidence_v1_codex_tick.sh`.
- Registered Hermes cron job `6623971c79d9`
  (`natural-evidence-v1-codex`, every 15 minutes).

gate_changes:
Hermes supervision now has a real scheduler binding for this repository. The
cron script sends TG/email before launching a background Codex worker. If a
previous worker is still active, the next tick sends a blocked/monitor-only
notification and does not launch another worker.

red_flags:
The scheduler-chain test used `HERMES_NAT_EV_RUN_CODEX=0`, so it validated
notification and worker launch plumbing without executing a Codex project
action. The registered cron job will use default `HERMES_NAT_EV_RUN_CODEX=1`.

telegram_notification:
Scheduler-chain test start and completion notifications were sent successfully.

email_notification:
Scheduler-chain test start and completion notifications were sent successfully.

notification_json:
- `results/natural_evidence_v1/status/hermes_reports/20260508_0640_scheduled_tick_notification.json`
- `results/natural_evidence_v1/status/hermes_reports/20260508_0640_scheduled_tick_completion_notification.json`

state_changing_action:
Scheduler registration and state documentation only.

next_allowed_action:
At the next scheduled tick, after TG/email succeeds, the background Codex worker
should execute at most one small allowed project action: artifact-only repaired
teacher-forced target-mass probe design over the primary branch-aware
candidates.

forbidden_actions_confirmed:
Training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer, FAR
aggregation, and paper-facing positive claims remain forbidden.

next_scheduled_run:
2026-05-08T02:55:55.861412-04:00
