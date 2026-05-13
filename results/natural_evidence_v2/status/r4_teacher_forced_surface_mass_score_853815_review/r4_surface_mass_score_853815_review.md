# R4 teacher-forced surface-mass score review

Job `853815` completed cleanly on Chimera:

- job name: `nat-ev-v2-r4tfm`
- state: `COMPLETED`
- elapsed: `00:04:39`
- exit code: `0:0`
- node: `chimera21`

This is not a Slurm/provider failure. It is a method gate failure.

## Gate Result

Teacher-forced surface gate: `FAIL`

Rows:

- score rows: `8192`
- scored rows: `24576`
- conditions: `base`, `protected`, `task_only`

## Key Numbers

| Condition | Mean target mass | Mean target margin | Median margin | Rank-1 rate |
|---|---:|---:|---:|---:|
| base | 0.0001302390 | -0.0000483820 | -0.0000096248 | 0.4375 |
| protected | 0.0000438295 | -0.0000133995 | -0.0000096318 | 0.4375 |
| task_only | 0.0003435588 | -0.0001943248 | -0.0000025933 | 0.4375 |

Observed lifts:

- protected vs base: `-0.0000864096` (required `>= +0.15`)
- protected vs task-only: `-0.0002997293` (required `>= +0.10`)
- task-only vs base: `+0.0002133198`

## Interpretation

The binary repair candidate fixed the formal two-sided surface-bank issue, but
it did not create a trainable surface channel under the existing protected
adapter. The protected adapter does not increase the target phrase-surface
mass; target masses are near zero across all arms.

This means the next step cannot be generation or locked-scale recovery. The
next step is artifact-only diagnosis of the surface bank, prefix shapes, and
target construction.

## Still Locked

- no free generation
- no training
- no Llama
- no same-family null
- no sanitizer
- no FAR aggregation
- no payload-diversity claim
- no paper-facing positive claim
