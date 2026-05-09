# Hermes natural_evidence_v1 scheduler reliability pass

phase:
POST_846699_BRANCH_AWARE_SCORE_INTERPRETATION_COMPLETE

active_jobs:
Hermes gateway is running. `natural-evidence-v1-codex` remains active with
schedule `every 15m`.

completed_jobs:
No new Slurm/model job was submitted in this reliability pass.

artifact_changes:
- Hardened `scripts/natural_evidence_v1/hermes_supervision_tick.py` with worker
  PID recording and stale-lock cleanup.
- Updated `/Users/guanjie/.hermes/scripts/natural_evidence_v1_codex_tick.sh`
  with an explicit PATH so Hermes cron can find Homebrew Python and Codex.
- Added a Codex binary fallback for the VS Code extension path.

gate_changes:
No research gate was opened. This only improves scheduler reliability.

red_flags:
The reliability test used `HERMES_NAT_EV_RUN_CODEX=0`; no Codex project action
was executed during the test.

telegram_notification:
Start and completion notifications were sent successfully.

email_notification:
Start and completion notifications were sent successfully.

notification_json:
- `results/natural_evidence_v1/status/hermes_reports/20260508_0644_scheduled_tick_notification.json`
- `results/natural_evidence_v1/status/hermes_reports/20260508_0644_scheduled_tick_completion_notification.json`

state_changing_action:
Scheduler reliability hardening only.

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
