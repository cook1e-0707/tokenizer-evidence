# R4 After-868348 Global-Unique Row Bank Plan

Date: 2026-05-17

## Status

`PASS_R4_AFTER_868348_GLOBAL_UNIQUE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT`

This artifact-only plan repairs the immediate row-bank capacity blocker found
after the `868348` dev diagnostic. It builds a 32-shard, 32,768-row bank using
the reviewed cover-natural dev prompts, the repaired first-token event codebook,
and a 16-prefix rotated allocation policy.

It does not reclassify `868348`, does not generate outputs, does not score a
model, and does not submit Slurm.

## Key Counts

- selected prompts: `2048`
- selected coordinates: `16`
- total rows: `32768`
- rows per shard: `1024`
- unique content prompt/prefix pairs: `32768`
- duplicate content prompt/prefix extra rows: `0`
- max prefix template fraction: `0.0625`

## Next Allowed Action

Artifact-only validation and actual Qwen tokenizer/controller preflight planning
for this row bank. No generation or Slurm submission is allowed until those
checks pass and a reviewed route is recorded.
