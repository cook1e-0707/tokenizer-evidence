# R4 Positive Support-Window Selectivity Repair Route

## Decision

Current phase:
`V2_R4_POSITIVE_SUPPORT_WINDOW_SELECTIVITY_REPAIR_ROUTE_RECORDED_NO_COMPUTE`.

Current blocker:
`BLOCK_R4_POSITIVE_SELECTIVITY_REPAIR_PACKAGE_ARTIFACT_ONLY_NEXT`.

The support-window repair line must not be resubmitted unchanged. The
selectivity analysis shows the current broad event-window bank captures normal
task language in raw and task-only outputs. This is a channel-design failure,
not a Slurm, H200, wrapper, or threshold-only failure.

## Why Threshold Tuning Is Not Enough

Post-hoc threshold tuning is not allowed to reclassify `859277`. As a
diagnostic, however, the score distribution also shows why simply raising the
keyed-score threshold is not the right next route:

| keyed-score threshold | protected blocks | raw blocks | task-only blocks |
| ---: | ---: | ---: | ---: |
| 6 | 22 | 12 | 14 |
| 50 | 22 | 8 | 8 |
| 75 | 22 | 2 | 0 |
| 100 | 22 | 1 | 0 |
| 125 | 14 | 0 | 0 |
| 150 | 7 | 0 | 0 |

The first threshold that clears raw/task-only controls drops protected blocks
to `14/32`, far below the dev diagnostic target. The current event definition
therefore lacks enough protected/null separation.

## Repair Direction

Implement a new artifact-only selectivity repair package before any compute:

1. Remove broad self-cue windows such as `plan <within_8_tokens> plan`.
2. Remove support families whose diagnostic null rate is comparable to
   protected (`safety`, `learning`, broad `maintenance`).
3. Require event evidence to be independently sourced, not mined from `859277`
   transcripts.
4. Add a static selectivity preflight that fails closed if the package relies
   on a single repeated family, a self-cue pair, or broad ordinary task words.
5. Add a static fixture with:
   - protected-like positive fixture accepting;
   - wrong-key and wrong-payload rejecting;
   - raw/task-like generic planning fixture rejecting.
6. Add a future dev-null route only after the artifact-only package passes
   static validation.

## Required Artifacts For Next Step

- `configs/natural_evidence_v2/r4_positive_selectivity_repair_package.yaml`
- `scripts/natural_evidence_v2/build_r4_positive_selectivity_repair_package.py`
- `scripts/natural_evidence_v2/validate_r4_positive_selectivity_repair_package.py`
- `tests/natural_evidence_v2/test_r4_positive_selectivity_repair_package.py`
- `results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_<timestamp>/`

## Static Gates

- source policy: `independent_static_taxonomy_not_859277_transcripts`
- self-cue event rows: `0`
- technical literals: `0`
- structural markers: `0`
- max family fraction: `<= 0.25`
- generic raw/task fixture accept: `false`
- wrong-key fixture accept: `false`
- wrong-payload fixture accept: `false`
- toy protected fixture accept: `true`
- unchanged `859277` reclassification: `false`

## Not Allowed

- no Slurm submission;
- no generation;
- no model scoring;
- no training;
- no Llama;
- no same-family null;
- no sanitizer;
- no FAR aggregation;
- no payload-diversity claim;
- no paper-facing positive claim;
- no post-hoc surface-bank construction from `859277`.

This route only permits artifact-only implementation and static validation of
the selectivity repair package.
