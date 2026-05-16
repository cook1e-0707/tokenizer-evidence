# R4 metric-exact micro-overfit route plan recorded

Phase:

```text
V2_R4_METRIC_EXACT_MICRO_OVERFIT_ROUTE_PLAN_ONLY_PASS_REMOTE_PREFLIGHT_NEXT
```

Blocker:

```text
BLOCK_R4_METRIC_EXACT_MICRO_OVERFIT_REMOTE_PREFLIGHT_NEXT_NO_SUBMIT_YET
```

Summary:

```text
The H200 micro-overfit wrapper now exposes SURFACE_MARGIN_LOSS_MODE.
Plan-only smoke passed with SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus and VALIDATE_PLAN_ONLY=1.
The wrapper exited before model/tokenizer loading, CUDA initialization, adapter loading, training, scoring, remote sync, or Slurm submission.
No allowlist entry is enabled.
```

Artifacts:

```text
docs/natural_evidence_v2/R4_METRIC_EXACT_MICRO_OVERFIT_ROUTE_PLAN_20260516.md
results/natural_evidence_v2/status/r4_metric_exact_micro_overfit_route_plan_20260516/
tests/natural_evidence_v2/test_r4_metric_exact_micro_overfit_wrapper.py
```

Future submission command pattern:

```text
sbatch --export=ALL,SURFACE_MARGIN_LOSS_MODE=logsumexp_softplus scripts/natural_evidence_v2/slurm/r4_candidate_v3_micro_overfit_h200.sbatch
```

Next allowed action:

```text
Remote sync/hash preflight for metric-exact micro-overfit route. Do not submit until local/remote hashes match, active-job preflight is clean, allowlist safety passes, Hermes TG/email notification is sent, and exactly one allowlist entry is enabled for submission.
```
