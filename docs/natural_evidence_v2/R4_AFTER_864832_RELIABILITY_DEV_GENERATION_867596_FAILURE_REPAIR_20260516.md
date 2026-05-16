# R4 Reliability Dev Generation 867596 Failure Repair

Date: 2026-05-16

## Failure

Job `867596` was submitted for the reviewed coordinate-unique reliability
dev-generation route. All four array tasks failed in about one second:

```text
job: 867596_[0-3]
state: FAILED
exit code: 1:0
elapsed: 00:00:01
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
```

The failure was not a model, generation, or decoder result. The Slurm stdout
shows that the internal route validator failed before generation:

```text
errors:
  - allowlist entry must be disabled
```

The cause is a submission-window race. The reviewed submission protocol enables
exactly one allowlist entry, calls `sbatch`, then immediately disables it. The
array tasks started before the disabled allowlist copy reached the job process,
so the full-mode wrapper saw its own reviewed entry still enabled and failed the
plan-only allowlist invariant.

## Repair

The route validator now has a full-mode-only option:

```text
--allow-submission-enabled-entry
```

When this flag is absent, the old plan-only behavior remains unchanged: the
allowlist entry must be disabled.

When this flag is present, the validator permits only these states:

```text
enabled entries: []
enabled entries: [v2_r4_after_864832_reliability_dev_generation_h200]
```

Any other enabled entry remains a hard failure.

The H200 wrapper passes this flag only when `VALIDATE_PLAN_ONLY != 1`; plan-only
validation still requires zero enabled entries.

## Validation

```text
local route validation:
  PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT

local wrapper plan-only:
  PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_WRAPPER_PLAN_ONLY

local enabled-window validation:
  PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT
  allow_submission_enabled_entry: true
```

## Next Step

Run a fresh local/remote preflight with the repaired validator/wrapper, notify
Hermes, then submit exactly one replacement H200/pomplun array job. The
allowlist entry must still be disabled immediately after `sbatch` returns.

This repair does not unlock training, Llama, same-family null, sanitizer, FAR,
payload diversity, or paper-facing claims.
