# R4 Positive Selectivity Pressure-Controller Route Selection

Timestamp UTC: `2026-05-15T04:24:00Z`

## Decision

Current phase:
`V2_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_SELECTED_ARTIFACT_ONLY_NO_COMPUTE`.

Current blocker:
`BLOCK_R4_POSITIVE_SELECTIVITY_PRESSURE_CONTROLLER_ROUTE_PLAN_NEXT`.

After the pressure/selectivity pivot package passed static validation, the next
route is selected as:

```text
teacher-forced protected-pressure / soft-controller scoring route
```

This route selection does not unlock Slurm. It authorizes only artifact-only
route planning, config preparation, wrapper review, and local static tests for a
future teacher-forced scoring-only job.

## Why This Route First

The latest free-generation diagnostic `859491` shows that lexical
support-window events are present but not protected-selective:

```text
protected mean events/block = 9.875
raw mean events/block       = 9.375
task-only mean events/block = 8.5625
protected max keyed score   = 16
raw max keyed score         = 23
```

Continuing to tweak the same prompt-policy / lexical event-window contract is
therefore not the shortest useful experiment. The next scientific question is
whether protected-specific distributional pressure can make the existing
natural continuation surfaces key/payload-selective before another generation
route is attempted.

Teacher-forced pressure/controller scoring is the safest next route because it:

- does not generate text;
- does not train or modify adapters;
- can preserve wrong-key and wrong-payload controls;
- directly measures target-mass lift, rank, margin, and collapse risk;
- can fail quickly before any additional free-generation compute.

## Routes Not Selected Yet

`metric_exact_objective_repair` is not selected as the immediate next route.
Training should wait until the scoring-only pressure/controller route shows
that a bounded pressure intervention can satisfy the teacher-forced gate without
collapse.

`explicit_stop_record` is also not selected yet because existing candidate-v3
teacher-forced artifacts showed nonzero positive direction before free
generation failed. A scoring-only pressure route is still a useful diagnostic.

## Future Route Scope

The future route to plan must be:

```text
Qwen only
same-contract a55e
teacher-forced scoring only
no generation
no training
no Llama
no FAR/sanitizer/same-family null
H200/pomplun only if model scoring is later submitted
```

It should score a bounded controller/gain grid and report at minimum:

- target mass lift vs base;
- target mass lift vs task-only;
- rank-1 rate;
- median target margin;
- wrong-key and wrong-payload separation;
- per-prefix/per-surface/per-coordinate weak strata;
- max single-surface mass or equivalent collapse metric;
- no target/other token overlap;
- no tokenizer/scorer boundary failures.

## Future Gate Before Generation

A later generation route can be reviewed only if a teacher-forced route records
all of:

```text
protected lift vs base >= +0.15
protected lift vs task-only >= +0.10
protected rank1 >= 0.75
protected median margin > 0
wrong-key simulated target mapping does not pass
wrong-payload simulated target mapping does not pass
no extreme collapse to one surface/template
```

The exact route config and wrapper must be reviewed before any Slurm
submission.

## Next Allowed Action

Artifact-only implementation of the teacher-forced pressure-controller route
plan:

- config;
- route validator;
- focused tests;
- wrapper plan-only review if needed;
- zero-enabled allowlist safety.

No Slurm submission, model scoring, generation, training, Llama, sanitizer,
FAR, same-family null, payload-diversity work, or paper-facing positive claim is
unlocked by this route selection.
