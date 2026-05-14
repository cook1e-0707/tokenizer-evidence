# R4 Positive Support-Window Coverage Dry-Run

## Verdict

`FAIL_SUPPORT_WINDOW_DRY_RUN_NULL_OR_CONTROL_ACCEPTS_NO_COMPUTE`

Support-window contract is not selective enough in this dry-run because one or more null/control arms accept.

This is a diagnostic-only dry-run over existing `859277` outputs. It does not
reclassify `859277` as positive and does not unlock compute or claims.

## Coverage By Condition

| condition | rows | rows with events | support rate | total events | mean events/row |
| --- | ---: | ---: | ---: | ---: | ---: |
| protected | 2048 | 1917 | 0.936 | 6312 | 3.082 |
| raw | 2048 | 1725 | 0.842 | 3889 | 1.899 |
| task_only | 2048 | 1726 | 0.843 | 3843 | 1.876 |

## Dry-Run Decoder Summary

| arm | blocks | dry-run accepts | mean events | mean coords | mean keyed score | mean margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| protected | 32 | 22 | 197.25 | 18.81 | 139.81 | 274.81 |
| raw | 32 | 12 | 121.53 | 16.97 | 61.09 | 114.12 |
| task_only | 32 | 14 | 120.09 | 17.47 | 61.22 | 115.56 |
| wrong_key | 32 | 0 | 197.25 | 18.53 | -149.06 | -303.94 |
| wrong_payload | 32 | 0 | 197.25 | 17.97 | -137.12 | -292.00 |

## Artifacts

- `coverage_summary.json`
- `condition_coverage.csv`
- `per_block_dry_run_decode.csv`
- `event_family_counts.csv`
- `support_event_sample.jsonl`
