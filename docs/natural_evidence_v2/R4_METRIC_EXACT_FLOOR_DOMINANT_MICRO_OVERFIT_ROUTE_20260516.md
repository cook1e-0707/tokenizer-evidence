# R4 Metric-Exact Floor-Dominant Micro-Overfit Route

Status: artifact-only route plan, no Slurm submitted by this document.

## Source Failure

Job `864332` completed cleanly but failed the teacher-forced surface-mass gate:

```text
protected mean target mass:   0.0179803
protected lift vs base:      +0.0131485
protected lift vs task-only: +0.0163079
protected rank1 rate:         0.980469
protected median margin:     +0.0059255
```

The failure shape is specific: target tokens usually outrank the immediate
other-token alternatives, but the absolute probability mass on the target token
set remains far below the gate. The final training summary also shows the
target-mass floor remained nearly unsatisfied:

```text
target_mass_floor: 0.20
final_floor_loss:  0.196320
final_ce_loss:     2.206710
```

## Repair Hypothesis

The prior `logsumexp_softplus` route still carried task CE pressure. The next
minimal repair is to test whether a floor-dominant, task-CE-disabled
micro-overfit can make the target mass floor active enough to pass the
teacher-forced gate.

This route does not change the candidate-v3 surface bank, prompt rows, scoring
rows, tokenizer contract, null controls, or downstream claim policy.

## Planned H200 Route

Future allowlist entry:

```text
v2_r4_candidate_v3_floor_dominant_micro_overfit_h200
```

Wrapper:

```text
scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```

Required command:

```text
sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus,TASK_CE_WEIGHT=0.0,TARGET_MASS_FLOOR=0.20,TARGET_MASS_FLOOR_LAMBDA=50.0,TARGET_MASS_CEILING=0.45,TARGET_MASS_CEILING_LAMBDA=5.0,MARGIN_LAMBDA=1.0,MAX_STEPS=128,LEARNING_RATE=1e-4 scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```

Compute policy remains:

```text
partition: pomplun
qos: pomplun
account: cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
```

## Gate

After execution, the route must be reviewed before any generation:

```text
protected lift vs base >= +0.15
protected lift vs task-only >= +0.10
protected rank1 >= 0.75
protected median margin > 0
task-only lift anomaly absent
scorer boundary failures = 0
target/other overlap = 0
```

If the gate fails, do not run generation.

## Current Stop Line

This route plan does not unlock generation, Qwen E2E, Llama, same-family null,
sanitizer, FAR aggregation, payload diversity, or paper-facing claims. A
single H200 submission is allowed only after local/remote preflight and
one-entry allowlist safety pass.
