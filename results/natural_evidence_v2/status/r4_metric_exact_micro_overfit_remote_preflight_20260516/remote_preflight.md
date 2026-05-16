# R4 Metric-Exact Micro-Overfit Remote Preflight

Status: `PASS_R4_METRIC_EXACT_MICRO_OVERFIT_REMOTE_PREFLIGHT_NO_SUBMIT`

Remote sync used targeted `rsync` because `~/tokenizer-evidence` has a dirty/untracked worktree and is not safe for `git pull --ff-only`.

Checks:

```text
local/remote hashes matched for route files
remote plan-only wrapper smoke passed
active squeue jobs: none
Slurm job submitted: false
```

Remote plan-only smoke used:

```text
SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus
VALIDATE_PLAN_ONLY=1
```

The wrapper exited before model/tokenizer loading, CUDA initialization, adapter loading, training, scoring, remote sync, or Slurm submission.
