# R4 Metric-Exact Micro-Overfit Pre-Submit Sync

phase: V2_R4_METRIC_EXACT_MICRO_OVERFIT_REMOTE_PREFLIGHT_PASS_SUBMISSION_READY

status:
- Local JSON state validation passed.
- Local allowlist safety passed with zero enabled entries.
- Remote allowlist safety passed with zero enabled entries.
- Remote plan-only H200 wrapper smoke passed with `SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus`.
- Local/remote hashes matched for the reviewed route files.
- Chimera active jobs check showed no active jobs.

next action:
- Enable exactly one allowlist entry: `v2_r4_candidate_v3_micro_overfit_h200`.
- Submit exactly one H200/pomplun Slurm job using the reviewed wrapper.
- Immediately disable the allowlist entry after `sbatch` returns.
- Record post-submit allowlist safety locally and remotely.

scope:
- This route may run the reviewed metric-exact micro-overfit training/scoring job.
- It does not unlock generation, Llama, same-family null, sanitizer, FAR aggregation, payload-diversity claims, or paper-facing positive claims.
