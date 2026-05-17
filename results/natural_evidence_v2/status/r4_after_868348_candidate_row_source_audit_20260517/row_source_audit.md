# R4 After-868348 Candidate Row Source Audit

Date: 2026-05-17

## Status

`FAIL_R4_AFTER_868348_EXISTING_ROW_SOURCES_INSUFFICIENT_FOR_GLOBAL_UNIQUE_32_BLOCK_ALLOCATION_NO_RERUN`

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

- scanned row files: `369`
- compatible source files: `6`
- compatible rows: `55296`
- unique content prompt/prefix pairs: `4096`
- required rows for 32 blocks: `32768`
- required globally unique content prompt/prefix pairs: `32768`

## Top Compatible Sources

- `results/natural_evidence_v2/status/r4_after_868299_first_token_event_dev_diagnostic_plan_20260517/row_allocation_rows.jsonl`: compatible_rows=32768, unique_content_pairs=2048, coordinates=16
- `results/natural_evidence_v2/status/r4_after_864832_two_sided_cover_bank_rows_20260516/cover_bank_aligned_target_only_rows.jsonl`: compatible_rows=8192, unique_content_pairs=2048, coordinates=32
- `results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl`: compatible_rows=4096, unique_content_pairs=2048, coordinates=16
- `results/natural_evidence_v2/status/r4_after_868212_full16_quality_repair_plan_20260516/row_allocation_rows.jsonl`: compatible_rows=4096, unique_content_pairs=2048, coordinates=16
- `results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_rows_20260516/reliability_surface_mass_rows.jsonl`: compatible_rows=3072, unique_content_pairs=2048, coordinates=12
- `results/natural_evidence_v2/status/r4_after_868151_first_token_event_quality_repair_plan_20260516/row_allocation_rows.jsonl`: compatible_rows=3072, unique_content_pairs=2048, coordinates=12
- `results/natural_evidence_v2/status/r3_2_h200_853430_review/remote_artifacts/shards/shard_00/coordinate_majority_r3_2_shard/r3_2_shard_decode_rows.jsonl`: compatible_rows=0, unique_content_pairs=0, coordinates=0
- `results/natural_evidence_v2/status/r3_2_h200_853430_review/remote_artifacts/shards/shard_01/coordinate_majority_r3_2_shard/r3_2_shard_decode_rows.jsonl`: compatible_rows=0, unique_content_pairs=0, coordinates=0
- `results/natural_evidence_v2/status/r3_2_h200_853430_review/remote_artifacts/shards/shard_02/coordinate_majority_r3_2_shard/r3_2_shard_decode_rows.jsonl`: compatible_rows=0, unique_content_pairs=0, coordinates=0
- `results/natural_evidence_v2/status/r3_2_h200_853430_review/remote_artifacts/shards/shard_03/coordinate_majority_r3_2_shard/r3_2_shard_decode_rows.jsonl`: compatible_rows=0, unique_content_pairs=0, coordinates=0
- `results/natural_evidence_v2/status/r3_2_h200_853430_review/remote_artifacts/shards/shard_04/coordinate_majority_r3_2_shard/r3_2_shard_decode_rows.jsonl`: compatible_rows=0, unique_content_pairs=0, coordinates=0
- `results/natural_evidence_v2/status/r3_2_h200_853430_review/remote_artifacts/shards/shard_05/coordinate_majority_r3_2_shard/r3_2_shard_decode_rows.jsonl`: compatible_rows=0, unique_content_pairs=0, coordinates=0

## Interpretation

The existing compatible row sources do not meet the necessary global-unique capacity checks for a strict 32-block rerun. The project should not rerun the 868348 route from the current row bank. The next repair must either build a larger reviewed prompt/row bank with tokenizer/controller preflight, or record a separate precommitted duplicate-gate semantics decision for future runs. Neither option can retroactively rescue 868348.

## Next Allowed Action

Artifact-only route planning for a larger reviewed row bank or a future-only duplicate-gate semantics package; no generation or Slurm submission from the current row bank.
