# R4 surface-mass failure diagnosis after 853815

This is an artifact-only diagnosis. It reads existing 853815 score rows,
the binary surface-bank repair candidate, and the frozen teacher-forced
probe rows. It does not train, generate, score models, submit Slurm, run
Llama, aggregate FAR, or make paper claims.

## Gate Result

- status: `FAIL_SURFACE_MASS_GATE_ARTIFACT_DIAGNOSED`
- scored rows: `24576`
- joined base/protected/task-only records: `8192`
- protected mean target mass: `0.0000438295`
- protected-vs-base mean lift: `-0.0000864096`
- protected-vs-task-only mean lift: `-0.0002997293`
- protected rank-1 rate: `0.4375`

## Diagnostic Interpretation

The binary candidate bank has both bit sides for every coordinate, so the
previous one-sided-bank formal problem is not the active blocker. The
first-token target/other overlap rate is not the primary failure either.
The active blocker is that the selected phrase-level target cylinders are
extremely low probability under the relevant prefixes, and the existing
protected adapter does not increase their mass.

## Worst Coordinates By Protected-vs-Base Lift

- coordinate `3`: lift `-0.0002907717`, protected mass `0.0001146455`
- coordinate `15`: lift `-0.0002907717`, protected mass `0.0001146455`
- coordinate `19`: lift `-0.0002907717`, protected mass `0.0001146455`
- coordinate `31`: lift `-0.0002907717`, protected mass `0.0001146455`
- coordinate `1`: lift `-0.0002806840`, protected mass `0.0001084136`

## Best Coordinates By Protected-vs-Base Lift

- coordinate `4`: lift `0.0000054594`, protected mass `0.0000226545`
- coordinate `8`: lift `0.0000054594`, protected mass `0.0000226545`
- coordinate `20`: lift `0.0000054594`, protected mass `0.0000226545`
- coordinate `24`: lift `0.0000054594`, protected mass `0.0000226545`
- coordinate `6`: lift `0.0000037794`, protected mass `0.0000245015`

## Highest Protected-Mass Surfaces

- `explain the reason`: protected mass `0.0001590915`, lift vs base `-0.0003031427`
- `record the choice`: protected mass `0.0001466426`, lift vs base `-0.0002028735`
- `track progress`: protected mass `0.0001211614`, lift vs base `-0.0003660669`
- `organize the details`: protected mass `0.0000970632`, lift vs base `-0.0002124570`
- `keep notes`: protected mass `0.0000854056`, lift vs base `-0.0003682390`
- `share the update`: protected mass `0.0000508176`, lift vs base `-0.0001179486`
- `prepare the materials`: protected mass `0.0000316106`, lift vs base `-0.0000003223`
- `clarify the next move`: protected mass `0.0000289504`, lift vs base `-0.0001127227`

## Next Allowed Action

Artifact-only repair design for R4 target construction, surface-bank
selection, and prefix shapes. Do not submit another scoring job or run
generation/training/Llama/FAR/sanitizer/paper-claim actions from this state.
