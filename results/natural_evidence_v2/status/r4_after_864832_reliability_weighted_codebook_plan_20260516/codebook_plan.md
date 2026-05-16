# R4 After 864832 Reliability-Weighted Codebook Plan

Status: `PASS_RELIABILITY_WEIGHTED_CODEBOOK_PLAN_8_PAIRS_AVAILABLE_NO_COMPUTE`

This is artifact-only planning from reviewed `866147` dev scoring artifacts.
It does not run Slurm, generation, training, Llama, FAR, sanitizer, or claims.

## Selection

- min lift vs base: `0.03`
- min rank1: `0.8`
- min median margin: `>0.0`
- selected pairs: `8`
- selected coordinates: `[6, 22, 10, 26, 1, 17, 3, 19, 15, 31, 8, 24, 4, 20, 7, 23]`

## Candidate Contract

- 4 payload bits
- 4 checksum bits
- 2 coordinates per bit
- pair-majority then checksum decoder
- primary reporting must remain `format_scrub=all`

## Next

The candidate codebook is not precommitted and cannot be used for compute
until a separate reviewed route decision freezes it and passes control-plane
preflight.
