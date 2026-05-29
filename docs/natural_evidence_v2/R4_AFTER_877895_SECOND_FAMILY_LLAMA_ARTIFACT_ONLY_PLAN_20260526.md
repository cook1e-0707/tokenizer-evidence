# R4 After-877895 Second-Family Llama Artifact-Only Plan - 2026-05-26

Status: `ROUTE_PLAN_R4_AFTER_877895_SECOND_FAMILY_LLAMA_ARTIFACT_ONLY_NO_SUBMIT`

## Purpose

Plan a canonical second-family migration for the current R4 provider-side
first-token event evidence route. This is not the old Step-label Llama route.
The goal is to test whether the trace-bound first-token event channel can be
made tokenizer-native for a second model family.

## Starting Facts

- Qwen R4 locked-scale positive package has passed.
- Qwen pre-FAR standard-control and organic-null package has passed.
- Qwen same-family raw-null package `877895` has passed across
  Qwen2.5-3B/7B/14B.
- Llama-related artifact inventory passed as artifact-only and found one
  reusable R4 candidate reference: `configs/model/llama3_1_8b_instruct.yaml`.
- Old Llama WP5/WP6 scripts and `LLAMA_V2_MIGRATION_PLAN_20260510.md` predate
  R4 and are historical/debug hints only.

## Candidate Second Family

```text
model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
model_config_reference: configs/model/llama3_1_8b_instruct.yaml
route_kind: provider-side first-token event, trace-bound, tokenizer-native
contract: same-contract a55e initially
payload_diversity_tested: false
```

## Required Route Stages

### L0 Inventory

Already completed:

```text
results/natural_evidence_v2/status/r4_after_877895_llama_migration_inventory_20260526/llama_inventory_summary.json
```

### L1 Artifact-Only Row-Bank Planning

Build a second-family row-bank plan that is independent of Qwen token ids.
The row bank may reuse task domains and prompt allocation principles, but it
must not reuse Qwen tokenizer boundaries or Qwen target token-id sets.

Hard requirements:

```text
no model forward
no generation
no training
no Llama scoring
no allowlist enablement
no old WP5/WP6 Step-label wrapper submission
```

### L2 Tokenizer-Only Boundary Preflight

Prepare a tokenizer-only Slurm route for the Llama tokenizer. The preflight must
check:

```text
prefix stability
non-empty target first-token id set
non-empty other first-token id set
target/other first-token id disjointness
zero boundary exceptions
```

The preflight must not load model weights or run a forward pass.

### L3 Small Second-Family Diagnostic Route

Only after L2 passes and is reviewed, prepare a small second-family generation
diagnostic. This route is still not a paper claim. It must include raw,
task-only or no-controller equivalents as applicable, wrong-key, and
wrong-payload replay controls.

### L4 Review Before Scale

Only if the small diagnostic passes should a locked-scale second-family route
be planned.

## Not Allowed by This Plan

```text
Slurm submission
allowlist enablement
Llama tokenizer preflight submission
Llama generation
Llama model scoring
training
sanitizer benchmark
FAR aggregation
payload diversity
paper-facing positive claim
text-only phrase-decoder success claim
```

## Next Allowed Action

Implement artifact-only L1 row-bank planning and route validation for the
second-family tokenizer-native first-token event route. Do not submit Slurm.
