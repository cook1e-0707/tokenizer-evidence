# Hermes Pre-Submit: R4 coverage-scale floor-dominant route

Phase:
`V2_R4_METRIC_EXACT_COVERAGE_SCALE_SINGLE_SUBMISSION_READY`

Summary:

- Source failure: reviewed job `864705`.
- New route: coverage-scale floor-dominant metric-exact micro-overfit.
- Local static validation: PASS.
- Local wrapper plan-only smoke: PASS.
- Remote hash preflight: PASS.
- Remote route validation: PASS.
- Remote wrapper plan-only smoke: PASS.
- Local and remote allowlist safety: PASS with zero enabled entries.
- Chimera active jobs: none.

Submission scope:

- Exactly one allowlist entry may be enabled:
  `v2_r4_candidate_v3_coverage_scale_micro_overfit_h200`
- Exactly one H200/pomplun Slurm job may be submitted.
- The allowlist entry must be disabled immediately after `sbatch` returns.
- No generation, Qwen E2E, Llama, same-family null, sanitizer, FAR,
  payload-diversity claim, or paper-facing positive claim is unlocked.

Command:

```text
sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus,TASK_CE_WEIGHT=0.0,TARGET_MASS_FLOOR=0.25,TARGET_MASS_FLOOR_LAMBDA=75.0,TARGET_MASS_CEILING=0.50,TARGET_MASS_CEILING_LAMBDA=5.0,MARGIN_LAMBDA=1.0,MAX_TRAIN_ROWS=8192,MAX_SCORE_ROWS=8192,MAX_STEPS=4096,BATCH_SIZE=2,GRADIENT_ACCUMULATION_STEPS=8,LEARNING_RATE=1e-4 scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```
