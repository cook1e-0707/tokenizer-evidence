# R4 After-868348 Candidate Row Source Audit

Date: 2026-05-17

## Status

`PASS_R4_AFTER_868348_EXISTING_ROW_SOURCES_HAVE_NECESSARY_GLOBAL_UNIQUE_CAPACITY_NO_RERUN`

This is an artifact-only audit. It does not reclassify `868348`, does not build
a new allocation, does not generate outputs, does not score a model, and does
not submit Slurm.

## Gate Context

The reviewed `868348` dev diagnostic had strong first-token event signal but
failed the strict global exact duplicate gate:

- protected strict accepts: `32/32`
- controls: `0/32` for raw/task-only/wrong-key/wrong-payload
- trace-binding invalid rows: `0`
- global exact duplicate extra rows: `2`
- duplicate attribution: task-only only

The immediate blocker is whether an existing reviewed row source can support a
32-block rerun without cyclic prompt/prefix reuse.

## Aggregate Inventory

- scanned row files: `1`
- compatible source files: `1`
- compatible rows: `32768`
- unique content prompt/prefix pairs: `32768`
- required rows for 32 blocks: `32768`
- required globally unique content prompt/prefix pairs: `32768`

## Top Compatible Sources

- `results/natural_evidence_v2/status/r4_after_868348_global_unique_row_bank_plan_20260517/row_allocation_rows.jsonl`: compatible_rows=32768, unique_content_pairs=32768, coordinates=16

## Interpretation

Existing compatible row sources meet the necessary aggregate count checks for a globally unique 32-block allocation. This is not yet a route approval; a deduplicating allocation builder, tokenizer/controller preflight, and reviewed Slurm route are still required.

## Next Allowed Action

Implement artifact-only deduplicating allocation construction and validation; do not submit Slurm until that allocation and the route preflight pass.
