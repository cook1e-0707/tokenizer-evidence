# R4 reliability dev generation submission 867596 review

Status: FAIL_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867596_ALLOWLIST_RACE_VALIDATOR_BLOCKED

Job `867596` was submitted as the reviewed coordinate-unique reliability dev-generation route. All four array tasks failed in about one second before generation. The Slurm stdout shows the internal route validator failed because it saw `allowlist entry must be disabled`. This is a control-plane race: the submitted wrapper started while the reviewed allowlist entry was still briefly enabled for the submission window.

No model generation, decode result, training, Llama, sanitizer, FAR, or paper-claim action was produced by this failed job.

Repair: validator/wrapper now permit exactly the reviewed submission entry during full-mode startup via `--allow-submission-enabled-entry`; plan-only validation still requires the entry disabled.
