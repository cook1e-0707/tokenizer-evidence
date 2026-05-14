# R4 Positive Support-Window Selectivity Analysis

## Verdict

`FAIL_SELECTIVITY_ANALYSIS_COMMON_SUPPORT_NO_COMPUTE`

Support-window events are broad task-language features, not a selective protected channel. Raw/task-only accepted blocks are driven by common positive-polarity support events under the same key.

This is artifact-only analysis over the already failed `859277` outputs. It does not
reclassify that run and does not permit Slurm, generation, model scoring, training, or claims.

## Dry-Run Arm Summary

| arm | blocks | accepts | mean events | mean positive keyed events | mean keyed score | min margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| protected | 32 | 22 | 197.25 | 168.53 | 139.81 | 222.00 |
| raw | 32 | 12 | 121.53 | 91.31 | 61.09 | 12.00 |
| task_only | 32 | 14 | 120.09 | 90.66 | 61.22 | 26.00 |
| wrong_key | 32 | 0 | 197.25 | 168.53 | -149.06 | -434.00 |
| wrong_payload | 32 | 0 | 197.25 | 168.53 | -137.12 | -422.00 |

## Selectivity Diagnostics

- accepted null/control blocks: `26`
- diagnostic selective surfaces under 859277: `0`
- raw plan-family event fraction: `0.725`
- task-only plan-family event fraction: `0.727`

Top protected families:

| family | protected events | raw events | task-only events | protected/max-null rate |
| --- | ---: | ---: | ---: | ---: |
| plan | 5063 | 2821 | 2792 | 1.794754 |
| maintenance | 652 | 487 | 507 | 1.285996 |
| clarify | 200 | 111 | 114 | 1.754386 |
| safety | 167 | 165 | 149 | 1.012121 |
| learning | 129 | 197 | 173 | 0.654822 |

## Artifacts

- `selectivity_summary.json`
- `surface_selectivity.csv`
- `family_selectivity.csv`
- `coordinate_selectivity.csv`
- `accepted_null_block_attribution.csv`
