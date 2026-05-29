# R4 After 864832 Two-Sided Controller 866147 Failure Pivot Decision

Status: `PIVOT_RECORDED_NO_COMPUTE`

## Reviewed Result

Safety-bound controller job `866147` completed cleanly and failed the
selective teacher-forced gate:

```text
summaries present: 24/24
controlled basic gate pass: 0/24
overall selective gate pass: 0/24
wrong-key basic gate pass: 0/24
wrong-payload basic gate pass: 0/24
best grid: 23
bonus: 2.0
penalty: 0.5
max_target_mass: 0.5
max_kl_budget: 0.2
controlled lift vs base: +0.059981
controlled lift vs task-only: +0.068912
controlled rank1: 0.736084
controlled median margin: +0.017426
```

This was a clean teacher-forced scoring result: no generation, no training, no
Llama, no FAR/sanitizer, and no paper-claim work started. The null controls did
not pass their basic gates, so the failure is insufficient positive strength,
not a false-accept safety failure.

## Failure Attribution

Best-grid attribution shows that the controller is at the reviewed safety
boundary and remains cap-limited:

```text
max_kl_budget cap rows: 1344
max_target_mass cap rows: 314
weakest coordinate: 2, lift +0.006593, rank1 0.222656
strongest coordinate: 19, lift +0.137760, rank1 0.996094
```

Reliability is highly uneven:

```text
coordinates with lift >= +0.08: 10/32
coordinates with lift >= +0.10: 8/32
coordinates with rank1 >= 0.75: 18/32
prefixes with rank1 >= 0.75: 3/8
surfaces with rank1 >= 0.75: 12/21
```

The aggregate channel fails because weak coordinates/prefixes/surfaces dilute
the strong strata. Further global controller gain would exceed the reviewed
max-target-mass and KL safety bounds.

## Pivot

Do not run another global controller sweep on the same bank. The next route is
artifact-only reliability-codebook planning:

```text
selected next phase:
V2_R4_AFTER_864832_RELIABILITY_WEIGHTED_CODEBOOK_ARTIFACT_ONLY

allowed next work:
- select high-reliability coordinates/surfaces from 866147 dev artifacts only
- simulate a reliability-weighted or erasure-aware ECC codebook
- preserve wrong-key and wrong-payload controls
- record a new precommit if a smaller reliable channel is viable

not allowed from 866147:
- generation
- training
- Llama
- same-family null
- sanitizer
- FAR
- payload diversity
- paper-facing positive claim
```

## Rationale

The current global target-mass gate is not met even at the bounded controller
edge. However, multiple strata are strong enough to motivate a reliability-
weighted channel. The next scientific question is whether a precommitted
smaller/redundant codebook can use only the reliable natural continuation
surfaces while preserving null separation and format-scrub requirements.
