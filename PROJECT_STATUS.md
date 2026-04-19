# Project Status

## Current Batch Status

- `batch3_preflight_failed`
- Reason:
  - clean generated-text baseline was not accepted
  - downstream attack runs were all `accepted_before=false -> accepted_after=false`
  - these runs are archived locally under `batch3_preflight_failed/` and must not be treated as formal robustness evidence

## Current Priority

1. `batch2.8`: migrate method-facing experiments off GPT-2.
2. Re-freeze tokenizer-specific catalogs for the selected method models.
3. Re-establish one clean generated-text acceptance result before reopening Batch 3.

## Model Policy

- `gpt2` is smoke-only from this point onward:
  - parser/verifier unit tests
  - plumbing checks
  - local smoke validation
- `gpt2` must not be used for paper-facing generated-text acceptance, Batch 2.8, Batch 3, or later comparison.
- Batch 2.8 model plan:
  - bridge: `Qwen/Qwen2.5-3B-Instruct`
  - main: `Qwen/Qwen2.5-7B-Instruct`
  - replication: `meta-llama/Meta-Llama-3.1-8B-Instruct`
