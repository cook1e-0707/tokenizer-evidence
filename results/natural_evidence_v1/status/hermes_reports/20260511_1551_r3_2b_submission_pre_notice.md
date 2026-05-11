# Hermes/Codex R3.2-B submission pre-notice

## Current phase

`V2_R3_2B_QWEN_LOCKED_SCALE_SINGLE_JOB_SUBMISSION_READY`

## Requested action

Submit exactly one Chimera Slurm job for the reviewed Qwen v2 R3.2
same-contract `a55e` locked-scale evaluation.

## Preflight status

- allowlist enabled entries before submission: `[]`
- R3.2-A allowlist decontamination: `PASS`
- R3.2 same-contract plan-only preflight: `PASS`
- R3.2 `852426` replay: `PASS`
- R3.2 full wrapper review: `PASS`
- wrapper syntax check: `PASS`
- contract id: `a55e`
- payload diversity tested: `false`
- intended allowlist entry: `v2_r3_2_qwen_locked_scale_eval`
- intended command: `sbatch scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`

## Constraints

- enable exactly one allowlist entry for the submission;
- submit exactly one Slurm job;
- immediately disable the entry after `sbatch` returns;
- do not start training, Llama, same-family null, sanitizer, FAR aggregation,
  or paper-facing positive claims;
- do not run CPU/GPU work on the Chimera login node.

## Next state update

After `sbatch` returns, Codex will write
`results/natural_evidence_v2/status/r3_2b_submission_record.json`, update
v1/v2 `gate_status.json`, sync local/remote state, and send a completion
notification.
