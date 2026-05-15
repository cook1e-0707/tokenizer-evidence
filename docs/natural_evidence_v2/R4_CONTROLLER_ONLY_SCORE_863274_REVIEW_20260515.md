# R4 Controller-Only Score 863274 Review

## Decision

Job `863274` is reviewed as:

```text
FAIL_R4_CONTROLLER_ONLY_SCORE_863274_NO_SELECTIVE_GATE
```

This is not a Slurm, H200, wrapper, tokenizer, or artifact-sync failure. All
`72/72` H200/pomplun array tasks completed with exit code `0:0`, and all `72`
summary artifacts were synced and reviewed.

## Scope

This was a teacher-forced controller-only scoring route. It did not run
generation, training, Qwen E2E, Llama, same-family null, sanitizer, FAR, payload
diversity, or paper-facing claim work.

The route tested the repair where controller arms use the base model instead of
loading the protected adapter:

```text
base
task_only
controlled_base
wrong_key_controlled_base
wrong_payload_controlled_base
```

## Gate Counts

| Metric | Count |
| --- | ---: |
| Slurm array tasks completed with `0:0` | 72/72 |
| Summary artifacts present | 72/72 |
| Controlled-base basic gate pass | 0/72 |
| Overall selective gate pass | 0/72 |
| Wrong-key basic gate pass | 0/72 |
| Wrong-payload basic gate pass | 0/72 |

Primary pass required at least one grid with `overall_selective_gate_pass=true`
and zero wrong-key/wrong-payload basic-gate passes. The wrong-control side is
now clean, but no controlled-base setting reaches the positive teacher-forced
gate.

## Best Observed Grid

Best controlled lift vs base:

```text
grid_index: 67
bonus_nats: 1.5
penalty_nats: 0.25
max_target_mass: 0.25
max_kl_budget: 0.1
controlled_mean_target_mass: 0.0202354971
controlled_lift_vs_base: +0.0154036601
controlled_lift_vs_task_only: +0.0185630693
controlled_rank1_rate: 0.498046875
controlled_median_target_margin: -0.0001098111
```

The configured controller grid is far below the R4 teacher-forced surface-mass
gate:

```text
protected/controlled lift vs base >= +0.15
protected/controlled lift vs task-only >= +0.10
rank1 rate >= 0.70
median target margin > 0
```

## Interpretation

The previous failure mode from job `859672` was wrong-control contamination:
wrong-key and wrong-payload arms inherited protected-adapter pressure while the
scorer still measured committed target ids. Job `863274` fixes that control
semantics issue. Wrong-key and wrong-payload controlled-base arms no longer pass
the basic gate.

However, the controller-only pressure tested here is too weak to create a
recoverable teacher-forced channel. The best controlled-base lift is only about
`+0.0154`, and the best rank-1 rate remains below `0.50`. This does not unlock
generation or downstream claims.

## Controlling Next Action

The next action is artifact-only failure diagnosis and repair-route planning.
Do not submit another H200 scoring job, generation job, training job, Llama job,
same-family null, sanitizer, FAR, payload-diversity route, or paper-facing claim
from this state.

Future compute is conditionally allowed only after a new reviewed route records
its prerequisites and control-plane checks.

## Reviewed Artifacts

```text
results/natural_evidence_v2/status/r4_controller_only_score_863274_review/
results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_score_863274/
```
