# R4 Controller-Only Safety-Bound Score 864117 Review

## Decision

Job `864117` is reviewed as:

```text
FAIL_R4_CONTROLLER_ONLY_SAFETY_BOUND_SCORE_864117_NO_SELECTIVE_GATE
```

This is not a Slurm, H200, wrapper, tokenizer, or artifact-sync failure. All
`24/24` H200/pomplun array tasks completed with exit code `0:0`, and all `24`
summary artifacts were synced and reviewed.

## Scope

This was a teacher-forced controller-only scoring route. It did not run
generation, training, Qwen E2E, Llama, same-family null, sanitizer, FAR,
payload diversity, or paper-facing claim work.

## Gate Counts

| Metric | Count |
| --- | ---: |
| Slurm array tasks completed with `0:0` | 24/24 |
| Summary artifacts present | 24/24 |
| Controlled-base basic gate pass | 0/24 |
| Overall selective gate pass | 0/24 |
| Wrong-key basic gate pass | 0/24 |
| Wrong-payload basic gate pass | 0/24 |

Primary pass required at least one grid with `overall_selective_gate_pass=true`
and zero wrong-key/wrong-payload basic-gate passes. The wrong-control side
remains clean, but no controlled-base setting reaches the positive
teacher-forced gate.

## Best Observed Grid

Best controlled lift vs base:

```text
grid_index: 21
bonus_nats: 2.0
penalty_nats: 0.5
max_target_mass: 0.35
max_kl_budget: 0.2
controlled_mean_target_mass: 0.0317901568
controlled_lift_vs_base: +0.0269583198
controlled_lift_vs_task_only: +0.0301177289
controlled_rank1_rate: 0.6015625
controlled_median_target_margin: +0.0033881384
wrong_key_basic_gate_pass: false
wrong_payload_basic_gate_pass: false
```

The safety-bound route improved the best positive pressure relative to job
`863274`, but it is still far below the R4 teacher-forced surface-mass gate:

```text
controlled lift vs base >= +0.15
controlled lift vs task-only >= +0.10
rank1 rate >= 0.75
median target margin > 0
```

The median margin is now positive in the best grid, but lift and rank remain
insufficient.

## Interpretation

Job `864117` confirms that wrong-control selectivity is not the immediate
limiting factor: wrong-key and wrong-payload controlled-base arms remain clean.
The limiting issue is positive pressure. A simple additive controller within
the reviewed safety envelope of `bonus<=2.0`, `KL<=0.20`, and
`max_target_mass<=0.50` does not provide enough average target-cylinder mass
for candidate-v3.

The best controlled mean target mass is `0.03179`, while the `+0.15` lift gate
requires about `0.15483` target mass at the current base mass. A simple
odds-ratio estimate gives roughly `1.72` additional logit-odds nats needed from
the best observed grid to reach the mass gate, before considering rank and
distributional constraints.

## Controlling Next Action

The next action is artifact-only pivot planning. Do not submit another H200
scoring job, generation job, training job, Llama job, null/FAR job, sanitizer
job, payload-diversity route, or paper-facing claim from this state.

Valid artifact-only directions include:

```text
row-adaptive or surface-aware controller design
metric-exact objective/training route planning
surface/channel redesign
```

Any future compute route must be recorded separately with route config,
validator coverage, wrapper plan-only checks, local/remote hash preflight,
Hermes notification, and one-entry allowlist control.

## Reviewed Artifacts

```text
results/natural_evidence_v2/status/r4_controller_only_safety_bound_score_864117_review/
results/natural_evidence_v2/status/r4_controller_only_safety_bound_failure_diagnosis_864117_20260516/
results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_score_864117/
```
