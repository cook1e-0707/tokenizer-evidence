# R4 After-870987 Same-Family Raw Null Route Decision - 2026-05-22

Status: `ROUTE_DECISION_R4_AFTER_870987_SAME_FAMILY_RAW_NULL_PLANNING_NO_SUBMIT`

Recorded at: `2026-05-22T13:30:48Z`

## Why This Is Next

The R4 Qwen first-token event route now has:

- locked-scale protected evidence passed;
- standard controls at `0/256` for raw/task-only/wrong-key/wrong-payload;
- organic raw null at `0/256`;
- no duplicate or trace-binding issue in the organic null package.

The next scientific risk is whether the verifier is accidentally accepting raw outputs from unprotected models in the same Qwen family. This is not Llama migration and not a full FAR claim. It is a same-family raw null route.

## Planned Models

- `Qwen/Qwen2.5-3B-Instruct`: same_family_smaller_raw, 64 raw blocks
- `Qwen/Qwen2.5-7B-Instruct`: same_family_reference_raw, 64 raw blocks
- `Qwen/Qwen2.5-14B-Instruct`: same_family_larger_raw, 64 raw blocks

If `Qwen/Qwen2.5-14B-Instruct` is unavailable on Chimera/HF cache or is too large for the reviewed H200 memory policy, the route must record a replacement before submission. Do not silently substitute another model.

## Minimum Gate

```text
per model raw accepts: 0/64
per model raw accepts ignoring quality: 0/64
technical forbidden public surface: 0
trace binding: 100% valid for generated rows when trace records are present
global duplicate extra rows: 0, or fail with attribution before scale-up
```

## Allowed Now

```text
artifact-only route/config/wrapper planning
model availability/hash preflight planning
allowlist entry disabled-by-default if wrapper exists
```

## Not Allowed Now

```text
Slurm submission
generation
training
Llama migration
sanitizer benchmark
FAR aggregation
payload diversity
paper-facing positive claim
```

## Next Allowed Action

Implement plan-only same-family raw-null route validation and wrapper preflight. Do not submit Slurm until route validation, local/remote hash preflight, zero-enabled allowlist safety, and exactly-one submission preflight pass.
