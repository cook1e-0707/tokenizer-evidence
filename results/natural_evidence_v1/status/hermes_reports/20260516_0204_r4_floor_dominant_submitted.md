# R4 Floor-Dominant Micro-Overfit Submitted

phase: V2_R4_METRIC_EXACT_FLOOR_DOMINANT_H200_JOB_864705_RUNNING

submission:
- job_id: `864705`
- job_name: `nat-ev-v2-r4mof`
- partition: `pomplun`
- node seen after submit: `chimera21`
- allowlist entry used: `v2_r4_candidate_v3_floor_dominant_micro_overfit_h200`

command:

```text
sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus,TASK_CE_WEIGHT=0.0,TARGET_MASS_FLOOR=0.20,TARGET_MASS_FLOOR_LAMBDA=50.0,TARGET_MASS_CEILING=0.45,TARGET_MASS_CEILING_LAMBDA=5.0,MARGIN_LAMBDA=1.0,MAX_STEPS=128,LEARNING_RATE=1e-4 scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```

control plane:
- Pre-submit local/remote allowlist checks passed with exactly the reviewed entry enabled.
- After `sbatch`, the allowlist entry was disabled locally and on Chimera.
- Post-submit local/remote allowlist checks passed with zero enabled entries.

next_allowed_action:
- Monitor Slurm job `864705`.
- After completion, sync artifacts and review teacher-forced surface-mass gates.

not unlocked:
- generation
- Qwen E2E rerun
- Llama
- same-family null
- sanitizer
- FAR aggregation
- payload-diversity claim
- paper-facing positive claim
