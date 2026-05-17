# R4 After-868260 Quality-Repair Confirmation Full-Mode Wrapper Review

Status: `PASS_R4_AFTER_868260_QUALITY_REPAIR_CONFIRMATION_FULL_MODE_WRAPPER_REVIEW_NO_SUBMIT`

This review advances the route from a plan-only/fail-closed wrapper to a reviewed full-mode wrapper package, but it does not submit Slurm and does not reclassify `868260`.

## What Changed

- `generate_r4_after_868016_controller_outputs.py` now supports the precommitted duplicate-safe sampling policy, HMAC-derived public sampling seeds, attempt logging, and trace-binding fields.
- `r4_after_868016_controller_generation_h200.sbatch` now passes duplicate-safe policy args into generation and verifies first-token event trace binding before decoding.
- `r4_after_868260_quality_repair_confirmation_h200.sbatch` now delegates non-plan mode to the reviewed generation/decode wrapper while preserving H200 `pomplun / cs_yinxin.wan` controls.
- The route validator now requires the reviewed full-mode wrapper fragments and keeps Slurm disabled until a separate single-submission preflight.

## Verification

- `uv run python -m py_compile ...`: pass.
- Route tests and helper tests: `16 passed`.
- `PLAN_ONLY=1` wrapper smoke: pass, no generation.
- `PLAN_ONLY=0 VALIDATE_PLAN_ONLY=1` delegate smoke: pass, no generation, toy decode protected accept=1 and wrong controls=0.

## Still Not Done

No Slurm job was submitted. The next route step is a fresh local/remote hash preflight for this updated full-mode wrapper package, followed by exactly-one allowlist submission preflight if hashes and allowlist safety pass.
