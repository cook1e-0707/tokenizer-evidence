# R4 Positive Support-Window Coverage Dry-Run

## Verdict

`FAIL_SUPPORT_WINDOW_DRY_RUN_NO_PROTECTED_ACCEPTS_NO_COMPUTE`

Support-window contract increases support diagnostics but does not recover protected accept-like blocks in this dry-run.

This is a diagnostic-only dry-run over existing `859277` outputs. It does not
reclassify `859277` as positive and does not unlock compute or claims.

## Coverage By Condition

| condition | rows | rows with events | support rate | total events | mean events/row |
| --- | ---: | ---: | ---: | ---: | ---: |
| protected | 2048 | 117 | 0.057 | 129 | 0.063 |
| raw | 2048 | 107 | 0.052 | 111 | 0.054 |
| task_only | 2048 | 86 | 0.042 | 88 | 0.043 |

## Dry-Run Decoder Summary

| arm | blocks | dry-run accepts | mean events | mean coords | mean keyed score | mean margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| protected | 29 | 0 | 4.45 | 1.79 | 0.72 | 3.66 |
| raw | 31 | 0 | 3.58 | 1.77 | 2.10 | 2.13 |
| task_only | 26 | 0 | 3.38 | 1.73 | 1.38 | 1.54 |
| wrong_key | 29 | 0 | 4.45 | 1.79 | -4.17 | -7.79 |
| wrong_payload | 29 | 0 | 4.45 | 1.76 | -3.07 | -6.69 |

## Artifacts

- `coverage_summary.json`
- `condition_coverage.csv`
- `per_block_dry_run_decode.csv`
- `event_family_counts.csv`
- `support_event_sample.jsonl`
