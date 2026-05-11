# Hermes/Codex R3.2-B job 853070 failed before generation

## Current phase

`V2_R3_2C_JOB_853070_FAILED_PROMPT_SPLIT_MISMATCH_NO_RESUBMIT`

## Job status

- job id: `853070`
- job name: `nat-ev-v2-r32qwen`
- Slurm state: `FAILED`
- elapsed: `00:00:00`
- exit code: `1:0`
- node: `chimera12`

## Failure

The job failed before any model generation. It wrote R3.2 precommit artifacts,
then failed in the first shard precommit decode call:

```text
ValueError: split 'wp3_r1_eval' has only 0 prompts; need 512
```

Root cause: the wrapper used file rows `0..511` for shard 0 while
`decode_wp6_r1_scale_blocks.py` filters for `split='wp3_r1_eval'`. In the
configured prompt file, rows `0..511` are `wp3_r1_dev`, so the selected eval
window was empty.

## Synced artifacts

- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_eval_853070/slurm/nat-ev-v2-r32qwen-853070.out`
- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_eval_853070/slurm/nat-ev-v2-r32qwen-853070.err`
- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_eval_853070/precommit/`
- `results/natural_evidence_v2/status/r3_2_qwen_locked_scale_eval_853070/r3_2_job_853070_failure_review.md`

## Claim control

No training, Llama, same-family null, sanitizer, FAR aggregation, payload
diversity claim, or paper-facing positive claim was unlocked. No second Slurm
job was submitted.

## Next allowed action

Repair the R3.2 prompt allocation and wrapper split contract artifact-only.
Do not submit another R3.2 Slurm job until the repaired allocation is recorded,
plan-only preflight is rerun, `852426` replay compatibility is re-reviewed or
superseded, allowlist safety is rechecked, and a new single-job submission route
is recorded.
