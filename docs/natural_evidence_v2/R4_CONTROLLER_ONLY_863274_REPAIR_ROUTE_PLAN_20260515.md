# R4 Controller-Only 863274 Repair Route Plan

## Purpose

This is an artifact-only repair-route plan after job `863274` failed the
controller-only teacher-forced selective gate.

The route result is useful because it separates two failure modes:

```text
wrong-control semantics: repaired
positive controlled-base pressure: still too weak
```

Wrong-key and wrong-payload controlled-base arms are now clean (`0/72` basic
gate passes), but controlled-base also has `0/72` basic gate passes. The best
controlled lift vs base is only `+0.0154036601`, far below the `+0.15`
teacher-forced gate.

## Non-Negotiable Scope

This plan does not unlock compute. It records the next repair direction only.

Still not allowed from the current state:

```text
H200 scoring submission
generation
training
Qwen E2E
Llama
same-family null
sanitizer
FAR
payload diversity
paper-facing positive claim
```

These actions remain conditionally authorized once their own recorded
prerequisite gates pass, but `863274` does not satisfy those gates.

## Diagnosis

The best grid was:

```text
bonus_nats: 1.5
penalty_nats: 0.25
max_target_mass: 0.25
max_kl_budget: 0.1
controlled_mean_target_mass: 0.0202354971
controlled_lift_vs_base: +0.0154036601
controlled_rank1_rate: 0.498046875
controlled_median_target_margin: -0.0001098111
```

To reach the `+0.15` lift target from the best observed controlled mass would
require roughly `2.18` additional logit-odds nats under a simple odds-ratio
calculation. A selected row-level cap probe shows the best controlled-base grid
is mostly uncapped:

```text
uncapped controlled-base rows: 8064/8192
max_kl_budget capped rows: 128/8192
controller_scale median: 1.0
controller_scale mean: 0.9988
```

So the immediate bottleneck is not primarily that the KL or target-mass cap is
clipping most rows. The tested controller pressure is intrinsically too weak
for this surface bank and scorer contract.

## Next Artifact-Only Work

The next repair package should answer one question before any new Slurm
submission:

```text
Can a stronger or more targeted controller produce enough positive pressure
while preserving wrong-key/wrong-payload rejection and public-template safety?
```

Minimum artifact-only work:

1. Define a new controller repair design that is not just a rerun of `863274`.
2. Decide whether the repair is a stronger additive controller, a row-adaptive
   pressure controller, or a metric-exact training objective pivot.
3. Record safety bounds for max target mass, KL, per-surface concentration,
   and wrong-control rejection before scoring.
4. Add or update local validators so any future H200 route is plan-only
   reviewed before allowlist enablement.
5. Keep generation blocked until a teacher-forced selective gate passes.

## Future Compute Gate

A future H200 teacher-forced route may be considered only after a new route doc
and config pass local validation. It must remain scoring-only and include:

```text
conditions:
  base
  task_only
  controlled_base
  wrong_key_controlled_base
  wrong_payload_controlled_base

required positive gate:
  controlled lift vs base >= +0.15
  controlled lift vs task-only >= +0.10
  controlled rank1 >= 0.75
  controlled median margin > 0

required null gate:
  wrong-key basic gate passes = 0
  wrong-payload basic gate passes = 0
```

If the next scoring route again fails positive pressure while wrong controls
remain clean, the project should pivot away from this simple controller grid
toward a metric-exact training objective or a more expressive provider-side
controller.
