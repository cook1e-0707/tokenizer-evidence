# Project Status

## Current Batch Status

- `batch3_preflight_failed`
- Reason:
  - clean generated-text baseline was not accepted
  - downstream attack runs were all `accepted_before=false -> accepted_after=false`
  - these runs are archived locally under `batch3_preflight_failed/` and must not be treated as formal robustness evidence

## Current Priority

1. `compiled-c3`: extend the passing Qwen 7B compiled path from `block_count=1` to `block_count=2`.
2. Keep Batch 3, baselines, and new model families frozen until compiled multi-block acceptance passes.
3. Preserve the compile-then-train path as the only active main-path implementation.

## Compiled Milestones

- `compiled-c0`: minimal Qwen 7B compiled path passed.
- `compiled-c1`: asymmetric single-block compiled path passed.
- `compiled-c2`: full single-block compiled path passed.
- Next target: `compiled-c3` double-block compiled path on the same Qwen 7B codebook.

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
