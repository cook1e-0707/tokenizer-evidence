# R4 Metric-Exact Micro-Overfit Submitted

phase: V2_R4_METRIC_EXACT_MICRO_OVERFIT_H200_JOB_864332_RUNNING

submission:
- job_id: `864332`
- job_name: `nat-ev-v2-r4mof`
- partition: `pomplun`
- node seen after submit: `chimera21`
- allowlist entry used: `v2_r4_candidate_v3_micro_overfit_h200`
- command: `sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`

control plane:
- Pre-submit local and remote allowlist checks passed with exactly the reviewed entry enabled.
- After `sbatch`, the allowlist entry was disabled locally and on Chimera.
- Post-submit local and remote allowlist checks passed with zero enabled entries.
- `CURRENT_STATE.md` and v1/v2 `gate_status.json` have been updated and synced to Chimera.

next allowed action:
- Monitor Slurm job `864332`.
- After completion, sync artifacts and review metric-exact micro-overfit train/heldout teacher-forced gates.

not unlocked by this submission:
- generation
- Qwen E2E rerun
- Llama
- same-family null
- sanitizer
- FAR aggregation
- payload-diversity claim
- paper-facing positive claim
