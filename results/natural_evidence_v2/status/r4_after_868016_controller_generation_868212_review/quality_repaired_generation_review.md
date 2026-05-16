# R4 After-868151 Quality-Repaired Generation Review

Status: `PASS_R4_AFTER_868151_QUALITY_REPAIRED_FIRST_TOKEN_EVENT_BLOCK_DIAGNOSTIC_GATE_NOT_LOCKED_POSITIVE_GLOBAL_DUPLICATE_CAVEAT`

- source job id: `868212`
- shards seen: `4` / `4`
- first-token protected accepts: `3` / `4`
- first-token control accepts: `{'raw': 0, 'task_only': 0, 'wrong_key': 0, 'wrong_payload': 0}`
- first-token forbidden public surface count: `0`
- first-token duplicate response hash count, per block: `0`
- token-id trace rows: `9216`
- event status counts: `{'erasure': 8293, 'other': 84, 'target': 839}`
- global duplicate response hash count: `4424`
- full-phrase protected accepts, format_scrub=all: `0`

## Interpretation

The quality-repaired first-token event diagnostic passes the small block-level gate: protected recovers 3/4 blocks, all controls reject, and the contextual literal / per-block duplicate gates are clean.

This is not a locked positive result. One protected shard failed because bit index 1, coordinate 26 had zero support, and generated-output hashes still have global cross-shard/condition duplicates. The full-phrase decoder remains failed, as expected.

## Failed Protected Blocks

- `shard_03_block_00` decoded `1-100101` vs expected `10100101`; missing bit indices `1`; min support `0`
