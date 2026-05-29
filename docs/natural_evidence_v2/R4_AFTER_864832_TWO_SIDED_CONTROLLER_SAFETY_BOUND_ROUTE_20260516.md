# R4 After 864832 Two-Sided Controller Safety-Bound Route

Status: `ROUTE_PLANNED_NO_SUBMIT`

This route follows the reviewed failure of controller-only job `865434`.
It is a teacher-forced controller scoring route only; it does not run
generation, training, Llama, same-family nulls, sanitizer, FAR, payload
diversity, or paper-claim work.

## Motivation

Job `865434` completed cleanly but failed the selective teacher-forced gate:

```text
best controlled lift vs base: +0.037408
best controlled lift vs task-only: +0.046340
best controlled rank1: 0.651367
best controlled median margin: +0.006919
wrong-key basic gate pass: 0/72
wrong-payload basic gate pass: 0/72
```

Failure attribution found that the best grid was at the previous upper edge
of the controller grid and triggered controller caps:

```text
selected grid: 71
bonus: 1.5
penalty: 0.25
max target mass: 0.45
max KL budget: 0.10
max_kl_budget cap rows: 1132
max_target_mass cap rows: 210
```

The next bounded test is therefore not generation. It is a stricter
teacher-forced safety-bound sweep over the existing two-sided bank, allowing
only the remaining reviewed controller limits:

```text
bonus_nats: [1.50, 1.75, 2.00]
penalty_nats: [0.25, 0.50]
max_target_mass: [0.45, 0.50]
max_kl_budget: [0.10, 0.20]
```

## Scope

```text
score rows: results/natural_evidence_v2/status/r4_after_864832_two_sided_cover_bank_rows_20260516/cover_bank_aligned_target_only_rows.jsonl
contract: a55e
model family: Qwen only
conditions: base, task_only, controlled_base, wrong_key_controlled_base, wrong_payload_controlled_base
wrapper: scripts/natural_evidence_v2/slurm/r4_after_864832_two_sided_controller_safety_bound_score_h200.sbatch
config: configs/natural_evidence_v2/r4_after_864832_two_sided_controller_safety_bound_route.yaml
array: 0-23%4
partition/qos/account: pomplun / pomplun / cs_yinxin.wan
gpu: h200
time: 30-00:00:00
```

## Gate

The route passes only if at least one grid satisfies all controlled gates while
wrong controls remain below their basic gates:

```text
controlled lift vs base >= +0.15
controlled lift vs task-only >= +0.10
controlled rank1 >= 0.75
controlled median margin > 0
wrong-key basic gate pass count = 0
wrong-payload basic gate pass count = 0
target/other overlap = 0
scorer boundary failures = 0
```

## Not Unlocked

This route does not unlock generation or training by itself. If it passes, the
next step is a separate reviewed small generation diagnostic route. If it fails,
the next step remains artifact-only objective/controller repair.
