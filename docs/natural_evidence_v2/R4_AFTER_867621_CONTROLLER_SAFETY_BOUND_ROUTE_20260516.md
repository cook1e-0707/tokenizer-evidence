# R4 After 867621 Controller Safety-Bound Route

Date: 2026-05-16

## Decision

The after-867621 protected-adapter gain sweep job `867897` failed: the best
gain was `protected_gain_0_5`, with only `+0.008797` lift vs base and no passing
gain. This rules out a simple adapter-scale rescue for the coordinate-unique
reliability surfaces.

The next controlled compute route is a controller-only teacher-forced scoring
sweep on the same 4096 rows. It does not load the protected adapter for
controller arms. It tests whether a bounded, provider-side soft logit
controller can produce the target-mass separation that the learned adapter did
not.

## Scope

```text
mode: teacher-forced scoring only
model family: Qwen only
contract_id: a55e
rows: after-867621 coordinate-unique reliability rows
row_count: 4096
controller_condition_set: controller_only_controls
conditions: base, task_only, controlled_base, wrong_key_controlled_base, wrong_payload_controlled_base
generation_allowed: false
training_allowed: false
```

## Controller Grid

```text
bonus_nats: [1.50, 1.75, 2.00]
penalty_nats: [0.25, 0.50]
max_target_mass: [0.45, 0.50]
max_kl_budget: [0.10, 0.20]
```

The grid is bounded to avoid an unconstrained collapse into a single visible
surface family. The primary gate remains teacher-forced only; passing this route
would not be a natural-output positive result.

## Still Not Unlocked

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
