# R4 Controller-Only Safety-Bound Pressure Route

## Status

Artifact-only route package recorded. No Slurm submission is unlocked by this
document.

## Motivation

Job `863274` repaired wrong-control contamination but failed positive pressure:

```text
controlled-base basic gate passes: 0/72
overall selective gate passes: 0/72
wrong-key basic gate passes: 0/72
wrong-payload basic gate passes: 0/72
best controlled lift vs base: +0.0154036601
```

The selected row-level cap probe showed the best controlled-base grid was
mostly uncapped, so the failure is not mainly due to KL or target-mass clipping.
The next scoring route must therefore be a new controller-pressure package, not
a rerun of `863274`.

## Route Package

Config:

```text
configs/natural_evidence_v2/r4_controller_only_safety_bound_pressure_route.yaml
```

Wrapper:

```text
scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

The wrapper now derives controller grid values from the route config through:

```text
scripts/natural_evidence_v2/emit_r4_pressure_controller_grid.py
```

This prevents a reviewed route config from silently diverging from the grid
that the sbatch script actually executes.

## Grid

The safety-bound route keeps the previous controller-only conditions and expands
only within the existing validator safety envelope:

```text
bonus_nats: [1.50, 1.75, 2.00]
penalty_nats: [0.25, 0.50]
max_target_mass: [0.35, 0.50]
max_kl_budget: [0.10, 0.20]
grid_size: 24
```

The future H200 command must override the array to exactly:

```text
--array=0-23%4
```

The route remains teacher-forced scoring only:

```text
base
task_only
controlled_base
wrong_key_controlled_base
wrong_payload_controlled_base
```

## Gate

Future route pass requires:

```text
controlled lift vs base >= +0.15
controlled lift vs task-only >= +0.10
controlled rank1 >= 0.75
controlled median margin > 0
wrong-key basic gate passes = 0
wrong-payload basic gate passes = 0
```

Passing this route would only unlock review of a small generation diagnostic
route. It would not by itself unlock Llama, same-family null, sanitizer, FAR,
payload diversity, or paper-facing claims.

## Control Plane

The allowlist entry is present but disabled:

```text
v2_r4_controller_only_safety_bound_pressure_score_h200
```

Before any future submission:

```text
zero-enabled allowlist preflight
local/remote hash preflight
wrapper plan-only smoke
Hermes TG/email notification
enable exactly one allowlist entry
disable immediately after sbatch returns
```
