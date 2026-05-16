# R4 Floor-Dominant Micro-Overfit Pre-Submit Sync

phase: V2_R4_METRIC_EXACT_FLOOR_DOMINANT_REMOTE_PREFLIGHT_PASS_SUBMISSION_READY

status:
- Source failure `864332` was reviewed and failed teacher-forced target-mass gates.
- New floor-dominant repair route was recorded.
- Local static validation passed.
- Local wrapper plan-only smoke passed.
- Remote Chimera route validation passed.
- Remote wrapper plan-only smoke passed.
- Local/remote route file hashes matched.
- Local and remote allowlist checks passed with zero enabled entries.
- Chimera active jobs: none.

single allowed submission:
- allowlist entry: `v2_r4_candidate_v3_floor_dominant_micro_overfit_h200`
- command: `sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus,TASK_CE_WEIGHT=0.0,TARGET_MASS_FLOOR=0.20,TARGET_MASS_FLOOR_LAMBDA=50.0,TARGET_MASS_CEILING=0.45,TARGET_MASS_CEILING_LAMBDA=5.0,MARGIN_LAMBDA=1.0,MAX_STEPS=128,LEARNING_RATE=1e-4 scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`

not unlocked:
- generation
- Qwen E2E rerun
- Llama
- same-family null
- sanitizer
- FAR aggregation
- payload-diversity claim
- paper-facing positive claim
