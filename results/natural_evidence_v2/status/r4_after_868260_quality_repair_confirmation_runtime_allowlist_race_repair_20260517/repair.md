# R4 After-868260 Runtime Allowlist Race Repair

Status: `PASS_R4_AFTER_868260_RUNTIME_ALLOWLIST_RACE_REPAIR_LOCAL_VALIDATED_NO_SUBMIT`

Job `868291` failed before generation because the runtime wrapper still required the allowlist entry to be enabled. That conflicted with the required immediate post-`sbatch` disablement.

Repair: the delegated generation wrapper no longer passes `--allow-submission-enabled-entry` at job runtime. The exactly-one allowlist check remains a pre-submit validation only.

Verification: `16 passed`; `PLAN_ONLY=0 VALIDATE_PLAN_ONLY=1` delegate smoke passed with no generation.
