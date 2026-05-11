Recorded a fresh R3.2 blocker instead of enabling the allowlist or submitting Slurm.

Changed:
- [20260511_0445_r3_2_submission_hard_constraint_blocker.md](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_0445_r3_2_submission_hard_constraint_blocker.md)
- [20260511_0445_r3_2_submission_hard_constraint_blocker.json](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v1/status/hermes_reports/20260511_0445_r3_2_submission_hard_constraint_blocker.json)
- Added matching entries to v1/v2 `gate_status.json`

Reason: the controlling tick requested one R3.2 Slurm submission, but the same hard constraints still forbid generation and Qwen E2E rerun. The reviewed R3.2 wrapper is still plan-only, so submission would be unsafe.

Validation passed:
- `jq empty results/natural_evidence_v1/status/gate_status.json`
- `jq empty results/natural_evidence_v2/status/gate_status.json`
- `jq empty results/natural_evidence_v1/status/hermes_reports/20260511_0445_r3_2_submission_hard_constraint_blocker.json`

No training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer, FAR aggregation, paper claim, Slurm submission, or Chimera login-node CPU/GPU work was started.