# Hermes/Codex Sync: R4 Selectivity Repair Route Recorded

Timestamp UTC: 2026-05-14T21:54:00Z

## Phase

`V2_R4_POSITIVE_SUPPORT_WINDOW_SELECTIVITY_REPAIR_ROUTE_RECORDED_NO_COMPUTE`

## Decision

Codex recorded an artifact-only selectivity repair route after the support-window
selectivity analysis failed.

The threshold sensitivity diagnostic shows the current support-window failure
is not threshold-only:

- threshold `6`: protected `22/32`, raw `12/32`, task-only `14/32`
- threshold `75`: protected `22/32`, raw `2/32`, task-only `0/32`
- threshold `100`: protected `22/32`, raw `1/32`, task-only `0/32`
- threshold `125`: protected `14/32`, raw `0/32`, task-only `0/32`

Clearing raw/task-only controls drops protected below the dev target. The
current support-window bank therefore needs an independent selectivity repair
package, not an unchanged resubmission or post-hoc threshold change.

## Artifacts

- `docs/natural_evidence_v2/R4_POSITIVE_SUPPORT_WINDOW_SELECTIVITY_REPAIR_ROUTE_20260514_2154.md`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_repair_route_20260514_2154/route_summary.json`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_repair_route_20260514_2154/threshold_sensitivity.csv`

## Next Allowed Action

Artifact-only implementation and static validation of a selectivity repair
package.

No Slurm submission, generation, model scoring, training, Llama, same-family
null, sanitizer, FAR aggregation, payload-diversity work, or paper-facing claim
is unlocked.
