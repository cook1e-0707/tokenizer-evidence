# R4 After-869348 Global-Unique Locked-Scale Row Bank Plan

Date: 2026-05-18

## Status

`PASS_R4_AFTER_869348_GLOBAL_UNIQUE_LOCKED_SCALE_ROW_BANK_BUILT_ARTIFACT_ONLY_NO_SUBMIT`

This artifact-only plan builds the held-out locked-scale row bank after the
`869348` Qwen first-token-event 32-block dev diagnostic passed. It uses the
locked split from the reviewed R4 cover-natural prompt bank, keeps the same
`a55e` contract and first-token event controller surface/codebook, and allocates
96 globally unique prompt/prefix shards.

It does not tokenize, score, generate outputs, train, enable an allowlist entry,
submit Slurm, or create a paper-facing claim.

## Key Counts

- selected prompts: `6144`
- selected coordinates: `16`
- total row cylinders: `98304`
- rows per shard: `1024`
- unique content prompt/prefix pairs: `98304`
- duplicate content prompt/prefix extra rows: `0`
- max prefix template fraction: `0.0625`

## Next Allowed Action

Artifact-only route validation and tokenizer boundary preflight for this locked
row bank. No generation or locked-scale Slurm submission is allowed until those
checks pass and a reviewed route is recorded.
