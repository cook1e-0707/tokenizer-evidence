# R4 After-864832 Reliability Dev Generation 867621 Review

Status: `FAIL_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867621_POSITIVE_GATE`

Job `867621` completed cleanly on Chimera H200/pomplun. All four array shards
completed with exit code `0:0`, wrote generated outputs, and produced both
`format_scrub=all` and no-scrub decode rows. The result is not a positive
evidence result.

## Gate Result

```text
generated rows: 6144
protected rows: 2048
raw rows: 2048
task_only rows: 2048
protected accepts, format_scrub=all: 0/32
protected accepts, no scrub: 0/32
raw accepts, format_scrub=all: 0/32
task-only accepts, format_scrub=all: 0/32
wrong-key accepts, format_scrub=all: 0/32
wrong-payload accepts, format_scrub=all: 0/32
protected forbidden public surface count: 0
```

The null controls remain clean, but the protected positive gate fails by a wide
margin: the dev gate required `>=26/32` protected accepts under
`format_scrub=all`, and the run produced `0/32`.

## Failure Analysis

Artifact-only failure analysis is recorded at:

```text
results/natural_evidence_v2/status/r4_after_864832_reliability_dev_generation_867621_failure_analysis_20260516/
```

The key diagnosis is:

```text
root_cause: free_generation_transfer_failure_surface_absent
protected coordinate-unique bank surface matches: 0
protected rows with any coordinate-unique bank surface: 0
protected duplicate response hash rows: 508
protected max duplicate response hash count: 27
protected rows with repeated sentence/clause units: 2001/2048
```

The protected model still emitted old candidate-v3 visible continuation
language at high volume:

```text
Create a plan: 44500 protected occurrences, 1938/2048 protected rows
Prepare a schedule: 4234 occurrences, 478/2048 rows
Prepare a budget: 11709 occurrences, 947/2048 rows
Prepare a plan: 11691 occurrences, 539/2048 rows
```

This is not a decoder oracle failure and not a Slurm failure. The
coordinate-unique reliability decoder works on the precommitted oracle, but the
free generator does not enter the frozen coordinate-unique surface bank. The
protected adapter instead collapses toward old `Create/Prepare` phrases.

## Control Decision

Do not rerun this route unchanged. Do not lower gates, add `867621`-observed
phrases to the bank, or treat the old candidate-v3 visible phrases as evidence.

The next allowed action is artifact-only repair or pivot planning. Any new
Slurm, generation, training, Llama, sanitizer, FAR, payload-diversity, or
paper-facing claim requires a new reviewed route decision and fresh preflight.
