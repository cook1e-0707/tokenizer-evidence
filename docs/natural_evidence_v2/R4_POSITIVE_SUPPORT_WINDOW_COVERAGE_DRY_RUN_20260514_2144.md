# R4 Positive Support-Window Coverage Dry-Run Review

Timestamp: `2026-05-14T21:44:01Z`

## Verdict

```text
FAIL_SUPPORT_WINDOW_DRY_RUN_NULL_OR_CONTROL_ACCEPTS_NO_COMPUTE
```

The support-window repair package fixes the zero-support symptom but is not
selective enough. It produces many support events in protected outputs, but it
also produces enough support in raw and task-only outputs to create dry-run
accept-like blocks.

This is diagnostic-only over existing `859277` outputs. It does not reclassify
`859277` as positive and does not unlock Slurm, generation, model scoring,
training, Llama, FAR, sanitizer, payload diversity, or paper-facing claims.

## Artifacts

```text
results/natural_evidence_v2/status/r4_positive_support_window_coverage_dry_run_20260514_2144/
```

Key files:

```text
coverage_summary.json
coverage_report.md
condition_coverage.csv
per_block_dry_run_decode.csv
event_family_counts.csv
support_event_sample.jsonl
```

## Coverage

| condition | rows | rows with events | support rate | total events | mean events/row |
| --- | ---: | ---: | ---: | ---: | ---: |
| protected | 2048 | 1917 | 0.936 | 6312 | 3.082 |
| raw | 2048 | 1725 | 0.842 | 3889 | 1.899 |
| task_only | 2048 | 1726 | 0.843 | 3843 | 1.876 |

The support-window extractor is no longer starving the decoder. However, the
support is common in null arms, not protected-specific.

## Dry-Run Decoder Summary

| arm | blocks | dry-run accepts | mean events | mean coords | mean keyed score | mean margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| protected | 32 | 22 | 197.25 | 18.81 | 139.81 | 274.81 |
| raw | 32 | 12 | 121.53 | 16.97 | 61.09 | 114.12 |
| task_only | 32 | 14 | 120.09 | 17.47 | 61.22 | 115.56 |
| wrong_key | 32 | 0 | 197.25 | 18.53 | -149.06 | -303.94 |
| wrong_payload | 32 | 0 | 197.25 | 17.97 | -137.12 | -292.00 |

Wrong-key and wrong-payload controls reject, but raw/task-only accepts are fatal
for this support contract. The support-window bank is too broad and captures
ordinary task language in unprotected outputs.

## Interpretation

The repair moved the failure from:

```text
zero support
```

to:

```text
support is too common across arms
```

This is progress diagnostically, but not a usable evidence channel. The next
repair must improve selectivity before any compute route is eligible.

## Current Blocker

```text
BLOCK_R4_POSITIVE_SUPPORT_WINDOW_COMMON_ACROSS_ARMS_REPAIR_NEXT
```

## Next Allowed Action

```text
artifact-only selectivity repair planning and static validation
```

The repair must focus on null separation, not simply increasing support.

