# R4 Floor-Dominant Remote Preflight

status: PASS_R4_METRIC_EXACT_FLOOR_DOMINANT_REMOTE_PREFLIGHT_NO_SUBMIT

Checks:
- Local/remote file hashes match: `True`
- Remote route validation: `PASS_R4_METRIC_EXACT_FLOOR_DOMINANT_ROUTE_STATIC_VALIDATION_NO_COMPUTE`
- Remote allowlist safety: `PASS_ZERO_ENABLED`
- Remote wrapper plan-only smoke: `PASS`
- Active jobs: none
- Slurm submitted: `False`

Next allowed action: Send Hermes TG/email notification, enable exactly one allowlist entry v2_r4_candidate_v3_floor_dominant_micro_overfit_h200, submit one H200 Slurm job, immediately disable allowlist, and record post-submit safety.
