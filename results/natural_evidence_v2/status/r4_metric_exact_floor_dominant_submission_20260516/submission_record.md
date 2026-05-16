# R4 Floor-Dominant Micro-Overfit Submission

status: SUBMITTED_R4_METRIC_EXACT_FLOOR_DOMINANT_H200_JOB_RUNNING

job_id: 864705
job_name: nat-ev-v2-r4mof
partition: pomplun
node_seen_after_submit: chimera21
allowlist_entry: v2_r4_candidate_v3_floor_dominant_micro_overfit_h200
command: `sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus,TASK_CE_WEIGHT=0.0,TARGET_MASS_FLOOR=0.20,TARGET_MASS_FLOOR_LAMBDA=50.0,TARGET_MASS_CEILING=0.45,TARGET_MASS_CEILING_LAMBDA=5.0,MARGIN_LAMBDA=1.0,MAX_STEPS=128,LEARNING_RATE=1e-4 scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch`

Post-submit control plane:
- Local post-submit allowlist safety: PASS with zero enabled entries.
- Remote post-submit allowlist safety: PASS with zero enabled entries.
- No generation, Qwen E2E, Llama, same-family null, sanitizer, FAR, payload-diversity claim, or paper-facing claim is unlocked by this submission.

Next allowed action: monitor job `864705`, sync completed artifacts when the job exits, and review teacher-forced gates before any further compute route.
