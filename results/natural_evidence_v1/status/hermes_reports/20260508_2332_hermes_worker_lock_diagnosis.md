# Hermes worker lock diagnosis

phase:
V2_WP3_CONTEXT_SPECIFIC_MASS_PLAN_READY_NEEDS_SLURM_SCORER

question:
Why did Hermes appear not to continue even though the next step was known?

finding:
Hermes did continue. The `20260508_2324` scheduler tick sent Telegram/email and
launched a Codex worker. The worker started `codex exec`, but that child process
buffered stdout and produced no visible transcript/report while running. Because
Hermes uses a single-worker lock, the silent worker held the lock and would have
blocked subsequent ticks.

action taken:
- terminated the silent `codex exec` child;
- confirmed the Hermes worker exited and released the lock;
- validated the artifact it had already written;
- reduced Hermes default Codex timeout from `7200` seconds to `900` seconds;
- reduced stale-lock slack from `+1800` seconds to `+300` seconds;
- updated gate status to record the completed context-specific mass plan.

artifact recovered:
- `scripts/natural_evidence_v2/build_wp3_context_mass_plan.py`
- `results/natural_evidence_v2/status/wp3_context_mass_plan_20260508_2324/`

plan summary:
- `eligible_detection_rows=7744`
- `score_plan_rows=230`
- `lowercase rows=115`
- `sentence_case rows=115`
- no model scoring, generation, training, E2E, FAR, or paper claim.

next_allowed_action:
Review the context-specific mass plan and prepare a plan-consuming Chimera Slurm
scorer. Do not run CPU/GPU scoring on the Chimera login node.

forbidden_actions_confirmed:
No training, generation, Qwen E2E, Llama, same-family null, sanitizer, FAR, or
positive paper claim was started.

notification:
Telegram and email notification were sent successfully via
`scripts/natural_evidence_v1/hermes_notify.py`; delivery summary is stored at
`results/natural_evidence_v1/status/hermes_reports/20260508_2332_hermes_worker_lock_diagnosis_notification.json`.
