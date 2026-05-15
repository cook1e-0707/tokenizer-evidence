# R4 Positive Selectivity Dev Diagnostic 859491 Review

Date: 2026-05-15T04:05:00Z

## Decision

Status:

```text
FAIL_DEV_GATE_NO_POSITIVE_CLAIM
```

Slurm and wrapper execution were clean, but the positive dev gate failed. This
run must not be reclassified as positive and must not unlock paper-facing
claims, Llama, sanitizer, FAR, payload diversity, or another unchanged
generation run.

## Slurm

- job id: `859491`
- partition/QoS/account/GPU: `pomplun` / `pomplun` / `cs_yinxin.wan` / H200
- terminal state: all array tasks `COMPLETED`
- exit code: `0:0`
- elapsed:
  - `859491_0`: `00:40:50`
  - `859491_1`: `00:40:41`
  - `859491_2`: `00:42:27`
  - `859491_3`: `00:33:39`

## Primary Gate

Primary decode mode:

```text
format_scrub=all
```

Results:

| Arm | Accepts | Blocks |
| --- | ---: | ---: |
| protected | 0 | 32 |
| raw | 0 | 32 |
| task_only | 0 | 32 |
| wrong_key | 0 | 32 |
| wrong_payload | 0 | 32 |

The required protected gate was `>=26/32`, so this is a hard positive failure.
Null/control arms are clean.

## Support And Score

Primary support summary:

| Arm | Mean Events | Mean Distinct Coords | Max Keyed Score | Max Margin |
| --- | ---: | ---: | ---: | ---: |
| protected | 9.875 | 5.000 | 16.0 | 20.0 |
| raw | 9.375 | 4.625 | 23.0 | 20.0 |
| task_only | 8.5625 | 4.15625 | 18.0 | 18.0 |

This is not a zero-event failure like the previous phrase-only route. The
selectivity prompt policy elicited support-window events, but those events are
still ordinary task-language events and are not protected-selective. Raw has a
higher max keyed score than protected.

## Output Checks

- generated outputs: `6144`
- per-condition outputs: `2048` each for protected/raw/task_only
- duplicate response hashes: `0`
- duplicate condition/prompt rows: `0`
- technical literal hits: `114`
  - `coordinate`: `104`
  - `bucket`: `10`

The technical literal hits are dominated by ordinary task-domain language such
as volunteer coordination. This remains a matcher-policy issue, but it does not
rescue the positive failure because protected recovery is `0/32`.

## Control Decision

Current blocker:

```text
BLOCK_R4_POSITIVE_SELECTIVITY_859491_REPAIR_PIVOT_ROUTE_DECISION_NEXT
```

Next allowed action:

```text
Artifact-only repair / pivot route decision and failure analysis only. Do not
resubmit this route unchanged.
```

