# R3.2 Prompt Split Contract Repair: 2026-05-11 17:47Z

## Decision

Record the repaired R3.2 prompt allocation and wrapper split contract after job
`853070` failed before generation.

This is artifact-only. It does not submit Slurm, start generation, rerun Qwen
E2E, train, start Llama, start same-family nulls, run sanitizer benchmarks,
aggregate FAR, or make paper-facing positive claims.

Machine-readable status:

```text
results/natural_evidence_v2/status/r3_2_prompt_split_contract_repair_20260511_1747.json
```

## Root Cause Being Repaired

The failed wrapper used file-row windows `0..511`, `512..1023`, and so on, while
the decode/generation scripts filter the prompt file by split before enforcing
the explicit file-row range. The configured prompt file has:

```text
wp3_r1_dev rows  = 0..511
wp3_r1_eval rows = 512..2559
```

Therefore shard `00` selected `0..511` under `split=wp3_r1_eval`, which is an
empty intersection.

## Repaired Contract

R3.2 must explicitly use the eval split everywhere the wrapper calls
generation or decode:

```text
selected_split = wp3_r1_eval
prompt_source_rows = 2560
prompt_source_sha256 = 20154c7b14851ce2116041176ab92acc727f1c49c343826eac9ecfc9430fc179
eval_prompt_file_rows = 512..2559
eval_prompt_count = 2048
```

The repaired locked-scale allocation uses the four available 512-row eval
windows, reused circularly across the 12 replicate groups:

| eval window index | file rows | sha256 |
|---:|---:|---|
| 0 | 512..1023 | `c470e445dcf25e356b70e0089827ec3434c66ccab1f5a77ed60c86e192183c26` |
| 1 | 1024..1535 | `52487748415b39a675e0c0860927cfc04047af0bec647a7c2fecb5812e044630` |
| 2 | 1536..2047 | `5e151bd80b2befe1b77658f98976c13d8901b6dfb357b5339a66fc4da6fab72a` |
| 3 | 2048..2559 | `d0acb1f20ed5b54b08a90facc852905764c341605606801f6a48eb10721b063c` |

Shard assignment:

```text
eval_window_index = replicate_group_index % 4
shard_00, shard_04, shard_08 -> rows 512..1023
shard_01, shard_05, shard_09 -> rows 1024..1535
shard_02, shard_06, shard_10 -> rows 1536..2047
shard_03, shard_07, shard_11 -> rows 2048..2559
```

All shard precommit decode calls and all generation calls must pass:

```text
--split wp3_r1_eval
--expected-file-row-start / --expected-file-row-end or
--prompt-file-row-start / --prompt-file-row-end matching the shard window
```

The previous `deterministic_5_window_circular_reuse_by_replicate_group_index`
policy is superseded for R3.2 because it includes the dev split. The repaired
policy is:

```text
deterministic_4_eval_window_circular_reuse_by_replicate_group_index
```

## Still Blocked

This record does not make R3.2 ready for resubmission. Before any new R3.2
Slurm job, the repaired allocation must be implemented in the precommit builder
and wrapper, plan-only preflight must be rerun, `852426` replay compatibility
must be re-reviewed or explicitly superseded, allowlist safety must be
rechecked, and a new single-job submission route must be recorded.

## Status

```text
RECORDED_R3_2_PROMPT_SPLIT_CONTRACT_REPAIR_NO_SLURM_NO_GENERATION
```
