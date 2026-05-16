# R4 Metric-Exact Micro-Overfit Route Plan

Date: 2026-05-16

## Status

`PASS_R4_METRIC_EXACT_MICRO_OVERFIT_ROUTE_PLAN_ONLY_NO_SUBMIT`

This records the next training route after the 864117 controller failure and metric-exact objective patch review. It does not submit Slurm or start training.

## Route

Future route type:

```text
protected micro-overfit train-and-teacher-forced-score
```

Wrapper:

```text
scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```

Future allowlist entry:

```text
v2_r4_candidate_v3_micro_overfit_h200
```

Required explicit environment for this route:

```text
SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus
VALIDATE_PLAN_ONLY=0
```

Current plan-only validation used:

```text
SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus
VALIDATE_PLAN_ONLY=1
```

The wrapper exited before model/tokenizer loading, CUDA initialization, adapter loading, training, scoring, remote sync, or Slurm submission.

## Compute Policy

Future Slurm route must use:

```text
partition: pomplun
qos: pomplun
account: cs_yinxin.wan
gres: gpu:h200:1
time: 30-00:00:00
```

## Gate

After future execution, the route must be reviewed against the teacher-forced gate before generation:

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

This plan does not unlock generation, Qwen E2E, Llama, null/FAR, sanitizer, payload diversity, or paper-facing claims.
