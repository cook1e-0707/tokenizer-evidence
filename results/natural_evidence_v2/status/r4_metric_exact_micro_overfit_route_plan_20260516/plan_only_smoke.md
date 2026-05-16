# R4 Metric-Exact Micro-Overfit Plan-Only Smoke

Command:

```text
REPO_HOME=$PWD PYTHON=$PWD/.venv/bin/python RUN_ROOT=/tmp/... VALIDATE_PLAN_ONLY=1 SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus bash scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```

Observed:

```text
surface_margin_loss_mode=logsumexp_softplus
partition=pomplun
account=cs_yinxin.wan
qos=pomplun
gres=gpu:h200:1
time_limit=30-00:00:00
VALIDATE_PLAN_ONLY=1: exiting before model/tokenizer loading, CUDA initialization, adapter loading, training, scoring, remote sync, or Slurm submission.
```

Status: `PASS`
