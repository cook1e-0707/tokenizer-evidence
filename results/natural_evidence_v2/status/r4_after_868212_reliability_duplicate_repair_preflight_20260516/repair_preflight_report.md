# R4 After-868212 Reliability/Duplicate Repair Preflight

Status: `FAIL_R4_AFTER_868212_RELIABILITY_DUPLICATE_REPAIR_PREFLIGHT_NO_SUBMIT`

## Codebook Reliability

- min active coordinates per bit: `2`
- singleton/codebook failures: `5`

## Duplicate Taxonomy

- generated rows: `9216`
- duplicate groups: `2908`
- duplicate extra rows: `4424`
- cross-arm duplicate groups: `1621`
- cross-shard duplicate groups: `2141`

## Route Implication

This preflight must pass before any next generation/scoring/training Slurm route is recorded.

## Singleton Failures

- bit `1` active=[26] source=[10, 26] reason=active_coordinate_count_below_minimum
- bit `1` active=[26] source=[10, 26] reason=coordinate_26_is_sole_active_coordinate
- bit `3` active=[19] source=[3, 19] reason=active_coordinate_count_below_minimum
- bit `5` active=[8] source=[8, 24] reason=active_coordinate_count_below_minimum
- bit `6` active=[4] source=[4, 20] reason=active_coordinate_count_below_minimum
