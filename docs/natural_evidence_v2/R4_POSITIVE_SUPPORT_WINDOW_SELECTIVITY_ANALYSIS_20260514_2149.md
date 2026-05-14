# R4 Positive Support-Window Selectivity Analysis

## Decision

Status: `FAIL_SELECTIVITY_ANALYSIS_COMMON_SUPPORT_NO_COMPUTE`.

The support-window repair fixed the zero-event failure from job `859277`, but
it is not selective enough to define a protected evidence channel. The broad
support-window events are common in ordinary raw and task-only task-language
outputs, so the current contract cannot be used for another generation or
Slurm route.

This is artifact-only analysis over existing `859277` outputs. It does not
reclassify `859277`, does not permit post-hoc surface-bank construction from
those transcripts, and does not unlock compute or claims.

## Evidence

- Generated rows analyzed: `6144`.
- Protected dry-run accepts: `22/32`.
- Raw dry-run accepts: `12/32`.
- Task-only dry-run accepts: `14/32`.
- Wrong-key dry-run accepts: `0/32`.
- Wrong-payload dry-run accepts: `0/32`.
- Accepted null/control blocks: `26`.
- Diagnostic selective surfaces under `859277`: `0`.
- Raw plan-family event fraction: `0.725`.
- Task-only plan-family event fraction: `0.727`.

The largest contributor is the `plan` family:

| family | protected events | raw events | task-only events | protected/max-null rate |
| --- | ---: | ---: | ---: | ---: |
| plan | 5063 | 2821 | 2792 | 1.794754 |
| maintenance | 652 | 487 | 507 | 1.285996 |
| clarify | 200 | 111 | 114 | 1.754386 |
| safety | 167 | 165 | 149 | 1.012121 |
| learning | 129 | 197 | 173 | 0.654822 |

The high positive-keyed event count in raw/task-only accepted blocks shows the
failure is not only support volume. The keyed mapping plus broad common events
gives enough positive score for unprotected outputs.

## Artifacts

- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/selectivity_summary.json`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/selectivity_report.md`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/surface_selectivity.csv`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/family_selectivity.csv`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/coordinate_selectivity.csv`
- `results/natural_evidence_v2/status/r4_positive_support_window_selectivity_analysis_20260514_2149/accepted_null_block_attribution.csv`

## Next Allowed Action

Artifact-only selectivity repair route design and static validation only.

No Slurm submission, generation, model scoring, training, Llama,
same-family null, sanitizer, FAR aggregation, payload-diversity claim, or
paper-facing positive claim is unlocked.
