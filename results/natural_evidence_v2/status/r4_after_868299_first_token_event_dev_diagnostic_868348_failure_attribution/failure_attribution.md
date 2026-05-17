# R4 after-868299 dev diagnostic 868348 failure attribution

Status: `RECORDED_R4_AFTER_868299_DEV_DIAGNOSTIC_868348_FAILURE_ATTRIBUTION_DUPLICATES_TASK_ONLY_ONLY`

## Key Facts

- protected strict accepts: `32/32`
- protected ignoring-quality accepts: `32/32`
- control accepts: `{'raw': 0, 'task_only': 0, 'wrong_key': 0, 'wrong_payload': 0}`
- trace binding invalid rows: `0`
- protected forbidden public surface count: `0`
- protected duplicate response hash count: `0`
- global exact duplicate extra rows: `2`
- duplicate rows by arm: `{'task_only': 4}`

## Interpretation

868348 passes the first-token event signal and null-separation checks, but fails the precommitted global exact duplicate quality gate. The exact duplicates are confined to task_only rows and arise from reused prompt/prefix pairs across cyclic dev shards, not protected accepted outputs.

This artifact does not reclassify `868348` as a pass and does not unlock paper-facing claims.
