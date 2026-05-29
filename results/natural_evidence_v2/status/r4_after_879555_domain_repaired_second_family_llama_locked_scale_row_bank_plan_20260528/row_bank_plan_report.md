# R4 After-879406 Second-Family Llama Locked-Scale Row-Bank Plan

Status: `PASS_R4_AFTER_879406_SECOND_FAMILY_LLAMA_LOCKED_SCALE_ROW_BANK_PLAN_ARTIFACT_ONLY_TOKENIZER_PENDING`

This is artifact-only. It copies the reviewed tokenizer-neutral 96-shard Qwen
locked-scale row bank into a Llama locked-scale candidate package after the
`879406` 32-block Llama dev diagnostic passed.

It does not run a tokenizer, load model weights, submit Slurm, generate, train,
or create a paper-facing or locked-scale transfer claim.

## Counts

- rows: `98304`
- prompts: `6144`
- shards: `96`
- rows per shard: `1024`
- tokenizer-specific token-id fields detected: `[]`

## Next Allowed Action

Artifact-only Llama tokenizer-boundary preflight route validation for this
96-shard row bank. No Llama locked-scale generation may be submitted until that
preflight and a reviewed H200 route pass.
