# Hermes/Codex Sync: 868212 repair route recorded

phase:
`V2_R4_AFTER_868212_FIRST_TOKEN_EVENT_RELIABILITY_DUPLICATE_REPAIR_ROUTE_RECORDED_ARTIFACT_ONLY_NEXT`

route:
`docs/natural_evidence_v2/R4_AFTER_868212_FIRST_TOKEN_EVENT_RELIABILITY_DUPLICATE_REPAIR_ROUTE_20260516.md`

summary:
```text
Codex recorded the next reviewed route after the 868212 diagnostic.

Route type:
- artifact-only repair implementation
- no Slurm generation/scoring/training authorized

Required repairs:
- reject singleton-bit codebooks
- require active_coordinate_count >= 2 for each committed bit
- coordinate 26 cannot be sole coordinate for any bit
- report within-arm, within-shard, cross-shard, cross-arm, and per-block duplicates

Source facts:
- protected first-token accepts: 3/4
- controls: 0/4 each
- failed block: shard_03_block_00
- missing coordinate: 26
- coordinate-26 protected erasures in shard_03: 64/64
- global duplicate response hash count: 4424
```

next_allowed_action:
Codex/Hermes may implement artifact-only reliability/duplicate repair preflights
and tests. Do not submit another Slurm job until those preflights pass and a
new single-submission route is recorded.
