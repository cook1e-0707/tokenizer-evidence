# R4 After-868260 Job 868291 Failure Review

Status: `FAILED_R4_AFTER_868260_JOB_868291_ALLOWLIST_RUNTIME_VALIDATION_RACE_NO_GENERATION`

Job `868291` failed immediately on all four array shards before generation/model forward. This was a control-plane race, not a model or method result.

Root cause: the delegated generation wrapper used `--allow-submission-enabled-entry` during job runtime. The route requires immediate allowlist disablement after `sbatch`, so by the time the array started, runtime validation saw `enabled_entries=[]` and failed.

Repair: keep the exactly-one allowlist check in the pre-submit path only; runtime job validation must accept the post-submit zero-enabled allowlist.
