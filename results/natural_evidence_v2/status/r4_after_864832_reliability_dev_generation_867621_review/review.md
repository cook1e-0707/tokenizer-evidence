# R4 reliability dev generation 867621 review

Status: `FAIL_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867621_POSITIVE_GATE`

Job `867621` completed cleanly on H200/pomplun. All four shards wrote generated outputs and both `format_scrub=all` and `format_scrub=none` decode summaries.

## Aggregate

```text
generated rows: 6144
protected rows: 2048
raw rows: 2048
task_only rows: 2048
format_scrub=all decode rows: 160
format_scrub=none decode rows: 160
```

## Gate Results

```text
protected accepts, format_scrub=all: 0/32
protected accepts, no scrub: 0/32
raw accepts, format_scrub=all: 0/32
task-only accepts, format_scrub=all: 0/32
wrong-key accepts, format_scrub=all: 0/32
wrong-payload accepts, format_scrub=all: 0/32
protected forbidden public surface count, format_scrub=all: 0
```

The null controls remain clean, but the positive protected recovery gate fails: protected recovery is `0/32`, far below the planned `>=26/32` dev gate.

## Channel Evidence

```text
protected selected surface count, format_scrub=all: sum=0, blocks_with_any=0/32, max=0
protected complete pairs, format_scrub=all: sum=0, max=0
```

Interpretation: the coordinate-unique reliability decoder works on the oracle and toy plan-only path, but the protected generator does not emit the coordinate-unique surface bank in free generation. This is a free-generation transfer failure, not a Slurm or null-control failure.

## Duplicate / Degeneracy Signal

```text
protected unique response hashes: 1690/2048
protected duplicate response hash rows: 508
protected max duplicate response hash count: 27
```

## Next Allowed Action

Artifact-only failure analysis only. Do not rerun, train, launch Llama, run sanitizer/FAR, or make a paper-facing claim until a new reviewed route decision is recorded.
