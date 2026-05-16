# R4 After-868151 First-Token Event Quality Audit

Status: `QUALITY_AUDIT_RECORDED_ARTIFACT_ONLY_R4_AFTER_868151_FIRST_TOKEN_EVENT`

## Facts

- generated rows: `9216`
- `coordinate` literal hits: `14`
- likely domain-sense `coordinate` hits: `10`
- duplicate response hash count across conditions: `2803`

## Duplicate Summary

- `protected`: rows `3072`, unique hashes `2317`, duplicate count `755`, max group `2`
- `raw`: rows `3072`, unique hashes `2048`, duplicate count `1024`, max group `2`
- `task_only`: rows `3072`, unique hashes `2048`, duplicate count `1024`, max group `2`

## Interpretation

- forbidden literal coordinate is entangled with ordinary volunteer-coordination task semantics in the audited examples
- deterministic greedy row-cylinder generation creates large duplicate response groups across repeated prompt/prefix patterns

## Route Implication

repair route must either avoid coordination-domain prompts or use a reviewed contextual technical-literal matcher, and must add diversity/window controls before another generation Slurm route
