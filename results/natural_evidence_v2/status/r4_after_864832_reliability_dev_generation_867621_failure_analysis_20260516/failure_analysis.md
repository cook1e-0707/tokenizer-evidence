# R4 Reliability Dev Generation 867621 Failure Analysis

Status: `FAILURE_ANALYSIS_RECORDED_NO_RERUN`

## Root Cause

The reviewed H200 job completed cleanly, but the protected free-generation channel did not emit the
precommitted coordinate-unique reliability surface bank. This is a free-generation transfer failure,
not a Slurm completion failure, tokenizer-boundary failure, decoder oracle failure, or null-control failure.

## Primary Evidence

- protected accepts, `format_scrub=all`: `0/32`
- protected gate: `>=26/32`
- null/control gate pass: `True`
- protected surface matches against coordinate-unique bank: `0`
- protected rows with any coordinate-unique bank surface: `0`
- protected duplicate response hash rows: `508`
- protected max duplicate response hash count: `27`
- protected rows with repeated sentence/clause units: `2001`

## Visible Pattern

The protected adapter still strongly biases old candidate-v3 continuation language such as
`Create a plan` / `Prepare a schedule`, but the reliability decoder surfaces are longer
coordinate-identifiable phrases such as ordinary review/check/confirm/update continuations.
The old visible phrase pressure therefore does not transfer into the frozen reliability bank.

## Control Decision

Do not rerun this route unchanged. Do not lower gates or add 867621-observed phrases to the bank.
A new reviewed repair/pivot route is required before any further Slurm, generation, training,
Llama, sanitizer, FAR, payload-diversity, or paper-facing claim work.

## Artifacts

- `condition_summary.csv`
- `candidate_v3_visible_phrase_counts.csv`
- `surface_head_coverage.csv`
- `top_ngrams_by_condition.csv`
- `degenerate_examples.csv`
- `failure_analysis_summary.json`
