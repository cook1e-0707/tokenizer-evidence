# Project Status

## Current Standing Status

- Accepted clean compiled path:
  - `compiled-c3-r4` passed on `Qwen/Qwen2.5-7B-Instruct`
  - representative payloads `U00`, `U03`, `U12`, and `U15` are accepted at seed `17`
- Standing robustness path:
  - `batch3-preflight-reopen` passed
  - `batch3a` passed
  - `batch3b` passed
  - `batch3c` passed
  - `batch3d` passed
- Active execution scope:
  - manuscript-facing consolidation only
  - theorem-package reporting and repair only
  - no additional Batch 3 expansion
  - no new baselines
  - no new model families
- Theorem-package standing:
  - `T1 contextual_exact` produced an accepted Chimera run under `theorem1_qwen7b/contextual_exact`
  - `T1 sequence_proxy` is not yet a completed control-arm result because its eval package still points `compiled_gate` at `scaffolded_slot_values`
  - initial `T2` objective package executed cleanly, but remained non-discriminative because the single-token-per-bucket compiled catalog collapses all three objectives to the same effective supervision problem
  - repaired `T2-r1` package now produced a discriminative result on the strict-passed multi-member-bucket Qwen catalog at `U15`
  - `fixed_representative` passes the exact-slot gate on the canonical representatives
  - `bucket_mass` remains bucket-correct but misses the canonical representative on `TOPIC=climate`
  - `uniform_bucket` remains bucket-correct but fails the exact-slot gate by selecting non-canonical members `SECTION=review` and `TOPIC=climate`

## Current Priority

1. Consolidate the accepted `compiled-c3-r4`, `batch3c`, and `batch3d` results into manuscript-facing tables and paper-facing summary artifacts.
2. Land the `T2-r1` theorem result as a paper-facing objective table plus a bucket-level supplementary table.
3. Keep `T1` truthful: `contextual_exact` stands, `sequence_proxy` remains blocked on eval/artifact compatibility until repaired.
4. Add explicit statistical aggregation, compute accounting, and run inclusion lists for the standing Qwen 7B claims.

## Archived Failures

- `batch3_preflight_failed`
  - clean generated-text baseline was not accepted
  - downstream attack runs were all `accepted_before=false -> accepted_after=false`
  - archived locally under `batch3_preflight_failed/`
  - not formal robustness evidence
- Pre-compiled Qwen 7B main-path failures
  - sequence-continuation era failures before compile-then-train
  - contract-coverage failures before the compiled contract compiler closed the train/eval gap
  - historical only; not current standing evidence
- Partial `batch3b` launch before `U03/U12` clean baselines existed
  - operational/configuration gap, later repaired by `compiled-c3-r3`
  - superseded by the accepted final `batch3b` result

## Compiled Milestones

- `compiled-c0`: minimal Qwen 7B compiled path passed.
- `compiled-c1`: asymmetric single-block compiled path passed.
- `compiled-c2`: full single-block compiled path passed.
- `compiled-c3`: double-block compiled path passed on the same Qwen 7B codebook.
- `compiled-c3-r1`: representative multi-payload validation passed on `U00`, `U03`, `U12`, and `U15`.
- `compiled-c3-r2`: multi-seed validation passed on `U00` and `U15` with seeds `23` and `29`.
- `compiled-c3-r3`: supplemental clean baselines passed on `U03` and `U12` with seeds `23` and `29`.
- `compiled-c3-r4`: supplemental clean baselines passed on `U00`, `U03`, `U12`, and `U15` with seed `17`.
- `batch3-preflight-reopen`: attack harness restored on accepted compiled-c3 baselines.
- `batch3a`: small robustness grid passed on `U00` and `U15` with seeds `23` and `29` across `whitespace_scrub` and `truncate_tail`.
- `batch3b`: payload-expansion robustness grid passed on `U00`, `U03`, `U12`, and `U15` with seeds `23` and `29`.
- `batch3c`: seed-expansion robustness grid passed on `U00`, `U03`, `U12`, and `U15` with seed `17`.
- `batch3d`: single-family delimiter-attack expansion passed on `U00`, `U03`, `U12`, and `U15` with seed `17`.
- Next target: manuscript appendix tables and main-result consolidation.

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

### Milestone: Batch 3A Small Robustness Grid Passed

Qwen/Qwen2.5-7B-Instruct passed `Batch 3A` on accepted compiled-c3 clean baselines without changing the model, codebook, runtime envelope, or attack harness.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r2` runs
- payloads: `U00`, `U15`
- seeds: `23`, `29`
- attack families:
  - `whitespace_scrub`
  - `truncate_tail`

Passing result:
- all eight attack runs completed successfully
- all eight attack runs started from `accepted_before = true`
- all four `whitespace_scrub` runs preserved acceptance
- all four `truncate_tail` runs caused acceptance failure

Interpretation:
- the attack harness is no longer only preflight-valid
- small-scope robustness behavior is now stable on the compiled-c3 Qwen 7B path
- the next gate is `Batch 3B`, which should expand payload coverage while keeping the same seeds, attack families, and runtime constraints

### Archived Failure: Batch 3B Payload Expansion Partially Executed

`Batch 3B` was launched to expand payload coverage from `U00/U15` to `U00/U03/U12/U15` under the unchanged compiled-c3 Qwen 7B attack path.

Observed result:
- `U00` and `U15` attack runs completed successfully and preserved the `Batch 3A` behavior pattern
- `U03` and `U12` attack runs did not produce valid attack outputs

Root cause:
- the missing half was not an attack-harness failure
- `U03_s23`, `U03_s29`, `U12_s23`, and `U12_s29` clean compiled-c3 baselines had never been materialized
- as a result, `attack.clean_eval_summary_path` expanded to an empty value for those cases

Required repair:
- supplement accepted clean baselines for `U03/U12 @ seed 23/29` on the same compiled-c3 path
- rerun only the missing `Batch 3B` attacks after those clean baselines exist

Guard added:
- attack execution now fails immediately with a clear error if `attack.clean_eval_summary_path` is empty or does not point to a real eval summary file

### Milestone: Compiled-C3-R3 Supplemental Clean Baselines Passed

Qwen/Qwen2.5-7B-Instruct passed the supplemental compiled-c3 clean-baseline stage for the previously missing payload and seed combinations.

Verified scope:
- framework: unchanged `compile-then-train`
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`
- payload targets: `U03`, `U12`
- seeds: `23`, `29`

Passing result:
- all four supplemental clean-baseline runs produced `accepted = true`
- all four supplemental clean-baseline runs produced `verifier_success = true`
- all four supplemental clean-baseline runs decoded the correct payload
- all four supplemental clean-baseline runs remained numerically healthy

Interpretation:
- the clean-baseline coverage needed for the full `Batch 3B` payload grid is now complete
- the remaining `Batch 3B` gap was operational rather than methodological

### Milestone: Batch 3B Payload-Expansion Grid Passed

Qwen/Qwen2.5-7B-Instruct passed `Batch 3B` after supplementing the missing clean baselines and rerunning only the blocked attack cases.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r2` and `compiled-c3-r3` runs
- payloads: `U00`, `U03`, `U12`, `U15`
- seeds: `23`, `29`
- attack families:
  - `whitespace_scrub`
  - `truncate_tail`

Passing result:
- all sixteen attack runs completed successfully
- all sixteen attack runs started from `accepted_before = true`
- all eight `whitespace_scrub` runs preserved acceptance
- all eight `truncate_tail` runs caused acceptance failure

Interpretation:
- payload expansion on the compiled-c3 Qwen 7B robustness path is now standing
- the next gate can move to a larger `Batch 3C`, but still without reopening baselines or new model families

### Milestone: Compiled-C3-R4 Seed-17 Clean Baselines Passed

Qwen/Qwen2.5-7B-Instruct passed the supplemental compiled-c3 clean-baseline stage for the seed-17 robustness extension.

Verified scope:
- framework: unchanged `compile-then-train`
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`
- payload targets: `U00`, `U03`, `U12`, `U15`
- seed: `17`

Passing result:
- all four clean-baseline runs produced `accepted = true`
- all four clean-baseline runs produced `verifier_success = true`
- all four clean-baseline runs decoded the correct payload
- all four clean-baseline runs remained numerically healthy

Interpretation:
- the clean-baseline coverage now includes the full representative payload grid at seed `17`
- the project can test a seed-expansion robustness stage without changing model family or payload scope

### Milestone: Batch 3C Seed-Expansion Grid Passed

Qwen/Qwen2.5-7B-Instruct passed `Batch 3C` by adding seed `17` across the established `Batch 3B` payload grid while keeping the attack families fixed.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r4` runs
- payloads: `U00`, `U03`, `U12`, `U15`
- seed: `17`
- attack families:
  - `whitespace_scrub`
  - `truncate_tail`

Passing result:
- all eight attack runs completed successfully
- all eight attack runs started from `accepted_before = true`
- all four `whitespace_scrub` runs preserved acceptance
- all four `truncate_tail` runs caused acceptance failure

Interpretation:
- the robustness grid now covers the representative payload set across an additional seed
- the next minimal axis of expansion is an additional attack family, not a broader model or payload sweep

### Milestone: Batch 3D Additional Attack-Family Grid Passed

Qwen/Qwen2.5-7B-Instruct passed `Batch 3D` by adding a delimiter-destruction attack family on top of the already accepted `Batch 3C` payload and seed grid.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r4` runs
- payloads: `U00`, `U03`, `U12`, `U15`
- seed: `17`
- new attack family:
  - `delimiter_scrub`

Passing result:
- all four attack runs completed successfully
- all four attack runs started from `accepted_before = true`
- all four `delimiter_scrub` runs caused acceptance failure

Interpretation:
- the compiled-c3 Qwen 7B path now has stable robustness evidence across payload expansion, seed expansion, and an additional structure-breaking attack family
- the next step should be documentation and paper-facing result consolidation rather than continued grid growth
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
