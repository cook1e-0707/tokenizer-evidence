# R4 First-Token Event Duplicate-Safe Generation Policy V2 Validation

Status: `PASS_R4_FIRST_TOKEN_EVENT_DUPLICATE_SAFE_GENERATION_POLICY_V2`

The config is artifact-only and does not permit Slurm or generation.

- retry is exact-hash only
- retry is blind to decode success and payload match
- same policy applies to all arms
- duplicate-exhausted rows remain failed
