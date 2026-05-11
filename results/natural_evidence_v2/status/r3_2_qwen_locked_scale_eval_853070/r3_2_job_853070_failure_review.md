# R3.2 job 853070 immediate failure review

## Job

- job id: `853070`
- job name: `nat-ev-v2-r32qwen`
- Slurm state: `FAILED`
- elapsed: `00:00:00`
- exit code: `1:0`
- node: `chimera12`

## What ran

The wrapper started successfully, wrote the R3.2 precommit artifacts, and then
failed before any model generation.

Artifacts produced:

- `precommit/r3_2_qwen_locked_scale_contract.json`
- `precommit/r3_2_selected_prompt_manifest.json`

## Failure

The failure happened in the first shard precommit decode call:

```text
ValueError: split 'wp3_r1_eval' has only 0 prompts; need 512
```

This is not a model failure and not a protected/null result. The job did not
reach generation, decoding, aggregate review, or any R3.2 accept/reject result.

## Root cause

The R3.2 wrapper passed `expected_file_row_start=0` and
`expected_file_row_end=511` to `decode_wp6_r1_scale_blocks.py` while that script
first filters the prompt file by `split='wp3_r1_eval'`. In the configured prompt
file, the first 512 rows are `wp3_r1_dev`, so the intersection of
`wp3_r1_eval` with file rows `0..511` is empty.

The R3.2 prompt allocation/precommit is therefore inconsistent with the
decoder/generator split filtering contract. This is a control-plane/prompt
allocation bug, not evidence against the v2 method.

## Claim control

- no training was started
- no Llama job was started
- no same-family null was started
- no sanitizer benchmark was started
- no FAR aggregation was started
- no paper-facing claim is allowed
- no payload diversity claim is allowed

## Next allowed action

Repair the R3.2 prompt allocation and wrapper split contract artifact-only.
Do not submit another R3.2 Slurm job until the repaired allocation is recorded,
plan-only preflight is rerun, `852426` replay compatibility is re-reviewed or
superseded, allowlist safety is rechecked, and a new single-job submission route
is recorded.
