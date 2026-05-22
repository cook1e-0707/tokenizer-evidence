# R4 Same-Family Tokenizer Preflight 874775 Failure Review

Status: `FAIL_WRAPPER_DEFAULT_PLAN_ONLY_ALLOWLIST_STATE_MISMATCH_NO_TOKENIZER_STARTED`

The array failed before tokenizer execution. The wrapper defaulted to `PLAN_ONLY=1`, so Slurm tasks ran route validation while the submission allowlist entry was still enabled. No model forward, scoring, generation, training, or tokenizer boundary preflight was started. The repair changes actual Slurm default to `PLAN_ONLY=0` while preserving explicit plan-only smoke mode.
