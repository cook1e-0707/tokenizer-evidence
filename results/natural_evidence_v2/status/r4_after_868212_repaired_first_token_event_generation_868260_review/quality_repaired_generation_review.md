# R4 After-868151 Quality-Repaired Generation Review

Status: `FAIL_R4_AFTER_868151_QUALITY_REPAIRED_FIRST_TOKEN_EVENT_BLOCK_DIAGNOSTIC_GATE_GLOBAL_DUPLICATE_CAVEAT`

- source job id: `868260`
- shards seen: `4` / `4`
- first-token protected accepts: `2` / `4`
- first-token control accepts: `{'raw': 0, 'task_only': 0, 'wrong_key': 0, 'wrong_payload': 0}`
- first-token forbidden public surface count: `3`
- first-token duplicate response hash count, per block: `13`
- token-id trace rows: `12288`
- event status counts: `{'erasure': 11574, 'other': 10, 'target': 704}`
- global duplicate response hash count: `7612`
- full-phrase protected accepts, format_scrub=all: `0`

## Interpretation

The quality-repaired first-token event diagnostic passes the small block-level gate: protected recovers 3/4 blocks, all controls reject, and the contextual literal / per-block duplicate gates are clean.

This is not a locked positive result. One protected shard failed because bit index 1, coordinate 26 had zero support, and generated-output hashes still have global cross-shard/condition duplicates. The full-phrase decoder remains failed, as expected.

## Failed Protected Blocks

- `shard_00_block_00` decoded `10100101` vs expected `10100101`; missing bit indices ``; min support `8`
- `shard_01_block_00` decoded `10100101` vs expected `10100101`; missing bit indices ``; min support `5`
