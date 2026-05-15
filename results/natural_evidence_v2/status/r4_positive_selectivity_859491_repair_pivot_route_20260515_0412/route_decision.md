# R4 Positive Selectivity 859491 Repair / Pivot Route

Status: `PASS_REPAIR_PIVOT_ROUTE_RECORDED_NO_COMPUTE`

## Decision

`859491` is frozen as a failed positive diagnostic. The H200/pomplun wrapper
completed cleanly, but the positive dev gate failed: protected accepts were
`0/32` under `format_scrub=all` and `0/32` under no-scrub. Raw, task-only,
wrong-key, and wrong-payload controls were also `0/32`.

The failure is not a Slurm or wrapper issue. It is also not a zero-support
issue: protected/raw/task-only all produced ordinary support-window events.
The problem is lack of protected-selective keyed signal.

## Pivot

Do not continue by resubmitting the same selectivity prompt-policy route. The
next project-advancing step is artifact-only pressure/selectivity pivot package
design and static validation.

The package must decide whether a future teacher-forced pressure/controller
route, metric-exact objective route, or stop record is justified. No compute is
unlocked by this route decision.

## Current Blocker

`BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_PIVOT_ARTIFACT_PACKAGE_NEXT`

## Next Allowed Action

Artifact-only pressure/selectivity pivot package design and static validation
only.

## Not Unlocked

No Slurm, generation, model scoring, training, Llama, same-family null,
sanitizer, FAR aggregation, payload diversity, or paper-facing positive claim.
