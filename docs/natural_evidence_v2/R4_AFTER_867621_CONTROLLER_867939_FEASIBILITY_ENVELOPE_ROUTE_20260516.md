# R4 After 867621 Controller 867939 Feasibility-Envelope Route

Status: `ARTIFACT_ONLY_ROUTE_PLAN_NO_SUBMIT`

Source failure:

```text
job: 867939
review: results/natural_evidence_v2/status/r4_after_867621_reliability_controller_safety_bound_score_867939_review/
failure analysis: results/natural_evidence_v2/status/r4_after_867621_controller_867939_failure_analysis_20260516/
```

The safety-bound controller grid improved rank1 but did not produce enough
target-mass lift:

```text
best grid: 23
best lift vs base: +0.041864
best lift vs task_only: +0.046783
best rank1: 0.830566
wrong-control failures: 0
lift deficit vs +0.15 gate: 0.108136
```

This route tests whether the mass-lift failure is an artifact of the previous
controller safety envelope. It remains teacher-forced scoring only.

## Scope

Allowed only after local/remote preflight and single-entry allowlist review:

```text
one H200/pomplun Slurm array
teacher-forced scoring only
condition set: controller_only_controls
rows: after-867621 coordinate-unique reliability rows
contract: a55e
```

Not allowed:

```text
generation
training
Qwen E2E rerun
Llama
same-family null
sanitizer
FAR aggregation
payload-diversity claim
paper-facing positive claim
```

## Feasibility Grid

```text
bonus_nats: [2.25, 2.50, 3.00, 3.50, 4.00]
penalty_nats: [0.50, 1.00]
max_target_mass: [0.50]
max_kl_budget: [0.20, 0.35, 0.50]
grid cells: 30
```

The target-mass cap remains `0.50`. The KL cap is relaxed only for this
diagnostic route to determine whether a stronger but still bounded controller
can reach the teacher-forced mass gate without wrong-key or wrong-payload
controls passing.

## Pass Gate

```text
exists grid with:
  controlled_lift_vs_base >= +0.15
  controlled_lift_vs_task_only >= +0.10
  controlled_rank1_rate >= 0.75
  controlled_median_target_margin > 0
  wrong_key_basic_gate_pass = false
  wrong_payload_basic_gate_pass = false
  target_other_overlap_rate = 0
  scorer_boundary_failures = 0
```

Passing this route would only unlock reviewed small-generation route planning.
It would not itself be a natural-output positive result.

## Control Plane

```text
config: configs/natural_evidence_v2/r4_after_867621_reliability_controller_feasibility_envelope_route.yaml
wrapper: scripts/natural_evidence_v2/slurm/r4_after_867621_reliability_controller_feasibility_envelope_score_h200.sbatch
validator: scripts/natural_evidence_v2/validate_r4_positive_selectivity_pressure_controller_route.py
allowlist entry: v2_r4_after_867621_reliability_controller_feasibility_envelope_score_h200
```
