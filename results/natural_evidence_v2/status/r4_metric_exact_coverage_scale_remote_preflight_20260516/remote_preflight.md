# R4 Coverage-Scale Remote Preflight

Status: `PASS_R4_METRIC_EXACT_COVERAGE_SCALE_REMOTE_PREFLIGHT_NO_SUBMIT`

Checks:

- Local and Chimera hashes match for route config, allowlist, state, route doc,
  validator, wrapper, trainer, test, and v2 gate status.
- Remote route validator passed.
- Remote allowlist safety passed with zero enabled entries.
- Remote wrapper plan-only smoke passed and exited before model/tokenizer
  loading, CUDA initialization, adapter loading, training, scoring, or Slurm
  submission.
- Active Chimera jobs: none.

No allowlist entry was enabled and no Slurm job was submitted during this
preflight.
