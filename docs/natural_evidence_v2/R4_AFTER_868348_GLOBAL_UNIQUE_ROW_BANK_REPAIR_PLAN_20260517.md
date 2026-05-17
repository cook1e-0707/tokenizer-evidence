# R4 After-868348 Global-Unique Row Bank Repair Plan

Date: 2026-05-17

## Decision

The active repair path is Option A: build a larger reviewed row bank before any
new generation rerun. The `868348` dev diagnostic is not reclassified.

## Why This Was Needed

`868348` had strong first-token event signal:

```text
protected strict accepts: 32/32
protected accepts ignoring quality: 32/32
raw/task-only/wrong-key/wrong-payload accepts: 0/32 each
trace binding invalid rows: 0
```

It failed the strict quality gate because the global exact duplicate count was
nonzero. Artifact-only attribution localized the duplicates to `task_only`, but
the precommitted dev gate still required global exact duplicates to be zero.

The old 32-block row allocation used cyclic reuse of a reviewed four-block
allocation. That was acceptable only as a dev diagnostic caveat; it is not safe
to rerun the same route after the duplicate failure.

## Artifact-Only Repair

Codex built:

```text
results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_plan_20260517/
```

The row bank uses:

```text
source prompts:
  results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl
source surface bank:
  results/natural_evidence_v2/precommit/r4_after_864832_coordinate_unique_surface_bank_20260516/surface_bank.json
source codebook:
  results/natural_evidence_v2/precommit/r4_after_868212_repaired_first_token_event_precommit_20260516/codebook.json
contract:
  a55e
```

Key counts:

```text
rows: 32768
shards: 32
rows per shard: 1024
selected coordinates: 16
prompts per shard: 64
prefix templates: 16
unique content prompt/prefix pairs: 32768
duplicate content prompt/prefix extra rows: 0
min unique content prompt/prefix pairs per selected coordinate: 2048
```

The 16 prefix templates are rotated by prompt and coordinate so a single
coordinate is not permanently tied to a single visible prefix family.

## Current Status

This is a pre-generation artifact only:

```text
generation_started: false
model_scoring_started: false
training_started: false
slurm_submitted: false
paper_claim_allowed: false
```

It repairs row-bank capacity only. It does not prove tokenizer compatibility,
controller compatibility, output uniqueness, trace recovery, or naturalness.

## Next Allowed Action

Artifact-only validation and actual Qwen tokenizer/controller preflight planning
for the global-unique row bank.

No Slurm generation may be submitted until:

```text
route validation passes
actual Qwen tokenizer boundary preflight passes
controller/decode preflight passes
local/remote hash preflight passes
zero-enabled allowlist safety passes
exactly one reviewed H200 route is recorded
```

Forbidden as current claims:

```text
paper-facing positive
text-only phrase decoder success
FAR
Llama transfer
payload diversity
sanitizer robustness
cross-family generality
```
