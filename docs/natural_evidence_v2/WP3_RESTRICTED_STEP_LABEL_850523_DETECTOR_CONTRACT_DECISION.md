# WP3 Restricted Step-Label 850523 Detector Contract Decision

## Scope

This artifact-only decision follows Slurm job `850523`, which close-failed the
selected primary-policy strict density audit because one
`strict_compact_step_label_lines` response placed `Step 1:` through `Step 16:`
inline in one paragraph.

This decision did not train, generate transcripts, submit Slurm, run Qwen E2E,
run Llama, run same-family nulls, run sanitizer benchmarks, aggregate FAR, or
make a paper-facing positive claim.

## Decision

Sentence-start inline `Step N:` labels are outside the current strict
Step-label density detector contract for the primary WP3 route.

The strict detector gate remains line-start only for planned structural slots:
each accepted label must begin a response line after optional whitespace,
optional markdown bullet marker, and optional markdown emphasis. A `Step N:`
label embedded later in the same paragraph is not counted as a complete
structural slot for this gate.

## Rationale

The 850523 policy artifact inherited overbroad wording that said
`line-start or sentence-start`, but the reviewed audit implementation and gate
review counted only line-start anchors. Reclassifying the one inline-paragraph
response after observing the transcript would convert a formatting failure into
a pass under a detector interpretation that was not actually used by the
predeclared scoring script.

The least post-hoc repair is therefore prompt-side: remove or rewrite the
`strict_compact_step_label_lines` prompt variant in a fresh artifact-only repair
plan. The existing 850523 result remains:

```text
structural_density_gate_status=FAIL
wp4_allowed=false
```

## Next Allowed Action

Prepare an artifact-only prompt repair that removes or rewrites
`strict_compact_step_label_lines` and records the strict line-start detector
clarification. Do not submit another Slurm job without review and explicit
approval.

Still forbidden:

- no WP4
- no training
- no Qwen E2E
- no Llama
- no same-family null
- no sanitizer benchmark
- no FAR aggregation
- no paper-facing positive claim
