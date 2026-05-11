# R3.2 Prompt Allocation Decision: 2026-05-11

Superseded scope note: the original prompt-window reuse math is retained for
provenance, but the payload-major `P00/P01/P02/P03` semantics are superseded by
`docs/natural_evidence_v2/R3_2_PAYLOAD_SEMANTICS_DECISION.md`. Canonical R3.2
now uses same-contract `a55e` replicate groups `shard_00..shard_11`, not
payload IDs. The prompt-window reuse rule is now interpreted as
`replicate_group_index mod 5`.

## Decision

Route R3.2 will use a deterministic prompt-window reuse policy before wrapper
implementation.

No Slurm job was submitted. No training, generation, Qwen E2E rerun, Llama,
same-family null, sanitizer benchmark, FAR aggregation, or paper-facing
positive claim was started.

Machine-readable status:

```text
results/natural_evidence_v2/status/r3_2_prompt_allocation_decision_20260511_0244.json
```

## Controlling Inputs

```text
docs/natural_evidence_v1/AUTOMATION_STATE.md
docs/natural_evidence_v1/next_step_codex_plan.md
results/natural_evidence_v1/status/gate_status.json
docs/natural_evidence_v2/PROTOCOL_CONTRACT.md
docs/natural_evidence_v2/CLAIM_GUARDRAILS.md
results/natural_evidence_v2/status/gate_status.json
docs/natural_evidence_v2/R3_2_QWEN_LOCKED_SCALE_PACKAGE_REVIEW_20260511.md
results/natural_evidence_v1/status/hermes_reports/20260511_0231_r3_2_wrapper_prompt_allocation_blocker.md
```

## Prompt Source

The selected reviewed prompt source is:

```text
local_path = results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/restricted_step_label_strict_density_audit_prompts.jsonl
remote_expected_path = /home/guanjie.lin001/tokenizer-evidence/results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/restricted_step_label_strict_density_audit_prompts.jsonl
prompt_source_rows = 2560
prompt_source_sha256 = 20154c7b14851ce2116041176ab92acc727f1c49c343826eac9ecfc9430fc179
```

The prompt source has fewer rows than a fully disjoint R3.2 allocation would
need:

```text
fully_disjoint_required_rows = 12 cells * 8 blocks/cell * 64 rows/block = 6144
available_source_rows = 2560
```

Therefore R3.2 must not claim cell-disjoint prompt allocation. The wrapper must
record the reuse rule in the precommit material before any transcript
generation.

## Prompt-Window Policy

Use five contiguous 512-row windows over the 2560-row source file:

| window | file rows | window sha256 |
|---|---:|---|
| W0 | 0..511 | 98b78920c07a9753f5ad5dcc7af198b5a4374f95d57b2d00d05f9ae327330038 |
| W1 | 512..1023 | c470e445dcf25e356b70e0089827ec3434c66ccab1f5a77ed60c86e192183c26 |
| W2 | 1024..1535 | 52487748415b39a675e0c0860927cfc04047af0bec647a7c2fecb5812e044630 |
| W3 | 1536..2047 | 5e151bd80b2befe1b77658f98976c13d8901b6dfb357b5339a66fc4da6fab72a |
| W4 | 2048..2559 | d0acb1f20ed5b54b08a90facc852905764c341605606801f6a48eb10721b063c |

Cell ordering is payload-major, then seed:

```text
cell_index = payload_index * 3 + seed_index
payload_order = P00, P01, P02, P03
seed_order = 17, 23, 29
prompt_window_index = cell_index mod 5
```

Per-cell allocation:

| cell | payload | seed | window | file rows |
|---:|---|---:|---:|---:|
| 0 | P00 | 17 | W0 | 0..511 |
| 1 | P00 | 23 | W1 | 512..1023 |
| 2 | P00 | 29 | W2 | 1024..1535 |
| 3 | P01 | 17 | W3 | 1536..2047 |
| 4 | P01 | 23 | W4 | 2048..2559 |
| 5 | P01 | 29 | W0 | 0..511 |
| 6 | P02 | 17 | W1 | 512..1023 |
| 7 | P02 | 23 | W2 | 1024..1535 |
| 8 | P02 | 29 | W3 | 1536..2047 |
| 9 | P03 | 17 | W4 | 2048..2559 |
| 10 | P03 | 23 | W0 | 0..511 |
| 11 | P03 | 29 | W1 | 512..1023 |

Each cell uses 8 consecutive 64-row blocks inside its assigned 512-row window.
Within a cell:

```text
block_j rows = window_start + 64*j .. window_start + 64*j + 63
for j in 0..7
```

## Selected Prompt Manifest Hash Policy

The R3.2 wrapper must write a selected prompt manifest before generation. The
manifest must include:

```text
schema_name = natural_evidence_v2_r3_2_prompt_allocation_manifest_v1
package_id = qwen_v2_r3_2_locked_scale_package_v1
prompt_source_path
prompt_source_rows
prompt_source_sha256
prompt_window_policy
payload_order
seed_order
cell_id, payload_id, seed, window index, file rows for every cell
block_id, block index, file rows, and row-jsonl sha256 for every block
```

The selected prompt manifest hash is the SHA-256 of canonical JSON with sorted
keys and compact separators, excluding any self-referential hash field:

```text
selected_prompt_manifest_sha256 = 4d49ae100b272f184a8b2563e5b64f768e6db01425a2384f1457a4eb10eedb67
```

The future R3.2 precommit contract must include this hash and must refuse to run
if the recomputed hash differs.

## Overwrite Refusal Surfaces

The R3.2 wrapper must refuse to start, including in plan-only mode, if the
target output directory already contains any of:

```text
precommit/r3_2_qwen_locked_scale_contract.json
precommit/r3_2_selected_prompt_manifest.json
r3_2_generation_summary.json
r3_2_generated_outputs.jsonl
r3_2_slot_observations.jsonl
r3_2_decode_decisions.jsonl
r3_2_coordinate_majority_decode_rows.jsonl
r3_2_coordinate_majority_summary.json
r3_2_support_by_block_budget.csv
r3_2_gate_review.json
```

It must also refuse if the output directory contains any legacy WP6 R1/R2
generation, decode, or precommit artifact name, because R3.2 must not be mixed
with previous run artifacts.

## Status

```text
R3_2_PROMPT_ALLOCATION_DECISION_RECORDED_NO_WRAPPER_NO_SLURM
```

Next allowed action: implement or review an R3.2-specific Qwen locked-scale
wrapper and disabled allowlist entry, then run local plan-only validation only.
Do not submit Slurm.
