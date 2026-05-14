# Hermes/Codex Sync: R4 Selectivity Package Coverage Dry-Run

Timestamp UTC: 2026-05-14T22:02:00Z

## Phase

`V2_R4_POSITIVE_SELECTIVITY_PACKAGE_COVERAGE_DRY_RUN_FAIL_NO_COMPUTE`

## Result

Codex built and statically validated the artifact-only selectivity repair
package, then ran the allowed artifact-only coverage dry-run on existing failed
`859277` outputs.

Static package:

- contract id: `r4_positive_selectivity_repair_v1`
- event-window rows: `96`
- max family fraction: `0.167`
- self-cue rows: `0`
- toy positive accept: `true`
- generic raw/task accept: `false`
- wrong-key/wrong-payload accept: `false`

Coverage dry-run:

- protected accepts: `0/29`
- raw accepts: `0/31`
- task-only accepts: `0/26`
- wrong-key accepts: `0/29`
- wrong-payload accepts: `0/29`
- protected row support rate: `0.057`
- protected mean events/block: `4.45`
- protected mean distinct coordinates/block: `1.79`

Interpretation: the package is cleaner and no longer causes null accepts, but
it is too sparse under the old `859277` prompt/output policy. The next blocker
is prompt-policy elicitation, not threshold tuning.

## Next Allowed Action

Artifact-only prompt-policy elicitation route design for the selectivity
package.

No Slurm submission, generation, model scoring, training, Llama, same-family
null, sanitizer, FAR aggregation, payload-diversity work, or paper-facing claim
is unlocked.
