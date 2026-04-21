# Project Status

## Current Batch Status

- `batch3_preflight_failed`
- Reason:
  - clean generated-text baseline was not accepted
  - downstream attack runs were all `accepted_before=false -> accepted_after=false`
  - these runs are archived locally under `batch3_preflight_failed/` and must not be treated as formal robustness evidence

## Current Priority

1. `batch3a`: run a minimal robustness grid on accepted Qwen 7B compiled-c3 baselines.
2. Keep baselines and new model families frozen until small-scope robustness is established on the compiled path.
3. Preserve the compile-then-train path as the only active main-path implementation.

## Compiled Milestones

- `compiled-c0`: minimal Qwen 7B compiled path passed.
- `compiled-c1`: asymmetric single-block compiled path passed.
- `compiled-c2`: full single-block compiled path passed.
- `compiled-c3`: double-block compiled path passed on the same Qwen 7B codebook.
- `compiled-c3-r1`: representative multi-payload validation passed on `U00`, `U03`, `U12`, and `U15`.
- `compiled-c3-r2`: multi-seed validation passed on `U00` and `U15` with seeds `23` and `29`.
- `batch3-preflight-reopen`: attack harness restored on accepted compiled-c3 baselines.
- Next target: `batch3a` small robustness grid on the same compiled-c3 path.

## 2026-04-20

### Milestone: Full Single-Block Compiled Path Passed

Qwen/Qwen2.5-7B-Instruct passed the full single-block compiled path under the compile-then-train framework.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- training path: `compile-then-train`
- objective: `field-conditioned masked bucket objective`
- decoding: deterministic one-token-per-slot constrained decoding
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `1`

Passing result:
- stage: `compiled-c2`
- accepted = `true`
- verifier_success = `true`
- decoded_payload correct
- no NaN / non-finite training failure
- compiled train contract emitted successfully

Interpretation:
- the full single-block compiled path is standing
- the primary blocker is no longer contract compilation, contextual alignment, or single-block bucket control

### Milestone: Double-Block Compiled Path Passed

Qwen/Qwen2.5-7B-Instruct then passed the double-block compiled path on the same compile-then-train framework and compiled codebook.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- training path: `compile-then-train`
- objective: `field-conditioned masked bucket objective`
- decoding: deterministic one-token-per-slot constrained decoding
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`

Passing result:
- stage: `compiled-c3`
- accepted = `true`
- verifier_success = `true`
- decoded_payload correct
- no NaN / non-finite training failure
- compiled train and eval contracts emitted successfully
- deterministic rendered canonical blocks verified successfully

Interpretation:
- the compiled multi-block path is now standing for the current Qwen 7B codebook
- the next gate is not a larger codebook or a new model family
- the next gate is representative multi-payload validation under the unchanged compiled-c3 framework

### Milestone: Representative Multi-Payload Double-Block Validation Passed

Qwen/Qwen2.5-7B-Instruct passed `compiled-c3-r1` on representative double-block payload targets without changing the compiled contract, codebook, prompt contract, or objective.

Verified scope:
- framework: unchanged `compile-then-train`
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`
- representative payload targets: `U00`, `U03`, `U12`, `U15`

Passing result:
- all four representative payload runs produced `accepted = true`
- all four representative payload runs produced `verifier_success = true`
- all four representative payload runs decoded the correct payload
- all four representative payload runs remained numerically healthy

Interpretation:
- the compiled-c3 path is no longer only a single-target success
- the next gate is seed robustness under the unchanged compiled-c3 setup

### Milestone: Multi-Seed Double-Block Validation Passed

Qwen/Qwen2.5-7B-Instruct passed `compiled-c3-r2` on additional seeds while keeping the compiled contract, codebook, prompt contract, and objective fixed.

Verified scope:
- framework: unchanged `compile-then-train`
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`
- payload targets: `U00`, `U15`
- seeds: `23`, `29`

Passing result:
- all four effective runs produced `accepted = true`
- all four effective runs produced `verifier_success = true`
- all four effective runs decoded the correct payload
- all four effective runs remained numerically healthy

Interpretation:
- the compiled-c3 path is no longer only a single-seed success
- the next gate is reopening Batch 3 preflight on accepted compiled baselines

### Milestone: Batch 3 Preflight Reopened

Qwen/Qwen2.5-7B-Instruct successfully reopened Batch 3 preflight on accepted compiled-c3 clean baselines.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r2` runs
- attack path: canonical attack over deterministic rerendered compiled slot values
- preflight attacks:
  - `U00 @ seed 23` with `whitespace_scrub`
  - `U15 @ seed 29` with `truncate_tail`

Passing result:
- both attack runs completed successfully
- both attack runs started from `accepted_before = true`
- one benign attack preserved acceptance
- one stronger truncation attack caused acceptance failure

Interpretation:
- the attack harness is now aligned with the compiled canonical path
- the next gate is a small-scope `Batch 3A` robustness grid, not a broad robustness sweep
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
