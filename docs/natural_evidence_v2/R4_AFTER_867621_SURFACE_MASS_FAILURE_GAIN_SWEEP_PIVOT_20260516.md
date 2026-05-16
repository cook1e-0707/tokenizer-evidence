# R4 After 867621 Surface-Mass Failure Gain-Sweep Pivot

Date: 2026-05-16

## Decision

Job `867849` is a clean teacher-forced scoring failure, not a tokenizer-boundary
failure and not a generation result.

The next canonical action is a protected-adapter gain sweep on the same
after-867621 coordinate-unique reliability rows:

```text
route: R4 after-867621 reliability protected-adapter gain sweep
mode: teacher-forced surface-mass scoring only
rows: results/natural_evidence_v2/status/r4_after_867621_reliability_surface_mass_rows_20260516/reliability_surface_mass_rows.jsonl
row_count: 4096
contract_id: a55e
generation_allowed: false
training_allowed: false
```

## Evidence From 867849

The H200 scoring job completed cleanly and had a valid actual-Qwen tokenizer
boundary preflight, but the protected adapter pressure was too weak:

```text
teacher_forced_surface_gate_status: FAIL
protected lift vs base: +0.006302
protected lift vs task_only: +0.011221
protected rank1 rate: 0.482666
protected median target margin: -0.000099876
task_only lift vs base: -0.004919
```

This says the task-only adapter is not the source of the target signal, while
the protected adapter does not yet put enough logit pressure on the selected
coordinate-unique natural continuations.

Per-coordinate analysis also shows localized signal rather than complete
absence of signal. Coordinate `7` had protected mean target mass `0.1467`, lift
vs base `0.1167`, lift vs task-only `0.1268`, and rank1 `1.0`. Several other
coordinates were weak or negative. The direct question is therefore whether
protected-adapter gain can turn this weak, uneven pressure into a passing
teacher-forced gate without changing the surface bank or starting generation.

## Scope

Allowed after this route is reviewed and local/remote preflight passes:

```text
exactly one H200/pomplun Slurm teacher-forced adapter-gain scoring job
base arm
task_only arm
protected adapter gains: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0
```

Not allowed by this pivot:

```text
free generation
training
Qwen E2E rerun
Llama
same-family null
sanitizer
FAR aggregation
payload diversity claim
paper-facing positive claim
```

## Future Gate

The gain sweep is only a teacher-forced calibration route. It unlocks a
generation route only if some protected gain satisfies all of:

```text
protected lift vs base >= +0.15
protected lift vs task_only >= +0.10
protected rank1 rate >= 0.75
protected median target margin > 0
target/other token-id overlap rate == 0
scorer boundary failures == 0
no task-only lift anomaly
no extreme collapse to a single surface
```

Passing this route would show that the failure mode is insufficient protected
adapter pressure. It would still not be a natural-output positive result.
