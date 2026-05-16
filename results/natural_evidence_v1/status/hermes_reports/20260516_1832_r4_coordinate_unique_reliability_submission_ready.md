# Hermes sync: R4 coordinate-unique reliability dev generation submission ready

phase:
V2_R4_AFTER_864832_COORDINATE_UNIQUE_RELIABILITY_DEV_GENERATION_REMOTE_PREFLIGHT_PASSED_SUBMISSION_READY

summary:
- The reliability-codebook generation route surface ambiguity blocker has been repaired artifact-only with a coordinate-unique surface bank.
- Local route validation passed.
- Local wrapper plan-only smoke passed: toy protected accepts 1/1, wrong-key 0/1, wrong-payload 0/1.
- Remote hash preflight passed with empty diff.
- Remote route validation passed.
- Remote zero-enabled allowlist safety passed.
- Remote wrapper plan-only smoke passed.
- Active Chimera jobs before submission: 0.

next controlled action:
Submit exactly one H200/pomplun Slurm array job for `v2_r4_after_864832_reliability_dev_generation_h200`, then immediately disable the allowlist entry and record post-submit zero-enabled allowlist safety.

scope:
- Qwen only
- same contract `a55e`
- 32 dev blocks, 64 prompts per block, 4 shards
- generated arms: protected/raw/task_only
- decode controls: protected/raw/task_only/wrong_key/wrong_payload
- primary decode: reliability-codebook `format_scrub=all`

not unlocked by this route:
training, Llama, same-family null, sanitizer, FAR, payload diversity claim, paper-facing positive claim.
