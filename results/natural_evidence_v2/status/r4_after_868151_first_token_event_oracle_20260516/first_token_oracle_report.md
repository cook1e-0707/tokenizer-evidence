# R4 After-868151 First-Token Event Oracle

Status: `FIRST_TOKEN_EVENT_ORACLE_RECORDED_ARTIFACT_ONLY_NOT_PRECOMMITTED_NOT_POSITIVE`

This is a posthoc artifact-only diagnostic on failed `868151` transcripts. It is not precommitted and cannot make `868151` a positive result.

## Aggregate
- `protected`: accepts ignoring forbidden `12` / `12`, complete-pair sum `96`
- `raw`: accepts ignoring forbidden `0` / `12`, complete-pair sum `45`
- `task_only`: accepts ignoring forbidden `0` / `12`, complete-pair sum `39`
- `wrong_key`: accepts ignoring forbidden `0` / `12`, complete-pair sum `96`
- `wrong_payload`: accepts ignoring forbidden `0` / `12`, complete-pair sum `96`

## Interpretation
first-token event oracle tests whether the teacher-forced first-token controller signal survives as row-local first-word events even when full phrase surfaces are absent
