# Hermes / Codex sync: R4 surface-mass scorer wrapper prepared

timestamp_utc: 2026-05-13T01:28:25Z

phase:
`V2_R4_TEACHER_FORCED_SURFACE_MASS_SCORER_WRAPPER_PREPARATION_NO_SUBMIT`

route_decision:
`docs/natural_evidence_v2/R4_TEACHER_FORCED_SURFACE_MASS_SCORER_ROUTE_DECISION_20260513.md`

summary:

- User explicitly authorized R4 teacher-forced surface-mass scorer wrapper preparation.
- This supersedes the latest no-Slurm hold only for wrapper preparation.
- Added disabled allowlist entry: `v2_r4_teacher_forced_surface_mass_score_h200`.
- Added wrapper:
  `scripts/natural_evidence_v2/slurm/r4_teacher_forced_surface_mass_score_h200.sbatch`.
- Ran local plan-only smoke with `VALIDATE_PLAN_ONLY=1`.
- Smoke result: `DRY_RUN_VALIDATED_INPUTS`, `8192` score rows, conditions `base`, `protected`, `task_only`.
- Post-wrapper allowlist safety: `PASS` with no enabled entries.

actions_not_started:

- allowlist enablement
- Slurm submission
- model scoring
- free generation
- training
- Llama
- same-family null
- sanitizer benchmark
- FAR aggregation
- payload-diversity claim
- paper-facing positive claim

key_artifacts:

- `results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_scorer_route_decision_20260513.json`
- `results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_wrapper_plan_smoke_20260513/r4_teacher_forced_surface_mass_summary.json`
- `results/natural_evidence_v2/status/r4_teacher_forced_surface_mass_score_allowlist_safety_zero_after_state_20260513.json`
- `docs/natural_evidence_v2/CURRENT_STATE.md`

next_allowed_action:

Review wrapper and plan-only smoke. A separate single-submission route decision
is required before enabling the allowlist entry or submitting exactly one Slurm
scoring job.
