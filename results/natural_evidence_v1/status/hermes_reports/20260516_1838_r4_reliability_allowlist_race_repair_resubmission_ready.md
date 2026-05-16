# Hermes sync: R4 reliability dev generation replacement submission ready

phase:
V2_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867596_ALLOWLIST_RACE_REPAIR_PREFLIGHT_PASSED_RESUBMISSION_READY

summary:
- Job 867596 failed before generation due to a control-plane validator allowlist race.
- No model generation, decode result, training, Llama, sanitizer, FAR, or claim artifact was produced by 867596.
- Validator/wrapper repaired: plan-only still requires zero enabled entries; full-mode startup permits only zero enabled entries or exactly the reviewed `v2_r4_after_864832_reliability_dev_generation_h200` entry.
- Fresh remote hash preflight passed.
- Fresh remote route validation passed.
- Fresh remote zero-enabled allowlist safety passed.
- Fresh remote wrapper plan-only smoke passed.
- Active Chimera jobs before replacement submission: 0.

next controlled action:
Submit exactly one replacement H200/pomplun Slurm array job with the repaired wrapper, then immediately disable the allowlist entry and record post-submit zero-enabled allowlist safety.
