# R4 After-868151 First-Token Event Decoder Replay

Status: `FIRST_TOKEN_EVENT_DECODE_RECORDED_ARTIFACT_ONLY_NOT_POSITIVE`

This replay uses the implemented first-token event decoder on the already failed
`868151` transcripts. It is artifact-only and cannot reclassify `868151` as a
positive result.

## Results

```text
event rows extracted: 9216
event source: text_fallback_old_transcript only
event statuses: target=845, other=82, erasure=8289
protected accepts with quality gates: 0/4
protected accepts ignoring quality: 4/4
raw accepts ignoring quality: 0/4
task-only accepts ignoring quality: 0/4
wrong-key accepts ignoring quality: 0/4
wrong-payload accepts ignoring quality: 0/4
```

Quality blockers:

```text
protected forbidden public surface count: 6
protected duplicate response hash count: 755
raw forbidden public surface count: 8
raw duplicate response hash count: 1024
task-only duplicate response hash count: 1024
```

Interpretation: the first-token event signal is recoverable in the failed
transcripts when quality gates are ignored, but this replay is still not a
positive result. It uses text fallback because `868151` did not store token-id
event traces. A future route must store token-id event traces and must repair
forbidden technical literals and deterministic duplicate outputs before another
Slurm generation submission.
