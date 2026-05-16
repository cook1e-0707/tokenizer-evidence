# R4 metric-exact objective patch reviewed

Phase:

```text
V2_R4_METRIC_EXACT_OBJECTIVE_CODE_PATCH_VALIDATED_ROUTE_PLANNING_NEXT
```

Blocker:

```text
BLOCK_R4_METRIC_EXACT_OBJECTIVE_MICRO_OVERFIT_ROUTE_PLAN_NEXT_NO_SLURM
```

Summary:

```text
Added disabled-by-default trainer option:
--surface-margin-loss-mode logsumexp_softplus

Default remains:
--surface-margin-loss-mode mass_relu

The patch provides metric-exact target/other logsumexp softplus margin helpers for future R4 micro-overfit training routes. No training, model scoring, generation, Slurm, Llama, null/FAR, sanitizer, payload diversity, or paper claim was started.
```

Artifacts:

```text
docs/natural_evidence_v2/R4_METRIC_EXACT_OBJECTIVE_PATCH_REVIEW_20260516.md
results/natural_evidence_v2/status/r4_metric_exact_objective_patch_review_20260516/
tests/natural_evidence_v2/test_r4_metric_exact_objective_helpers.py
```

Validation:

```text
uv run pytest tests/natural_evidence_v2/test_r4_metric_exact_objective_helpers.py tests/natural_evidence_v2/test_r4_training_objective_disabled_by_default.py tests/natural_evidence_v2/test_r4_target_mass_floor_loss.py tests/natural_evidence_v2/test_r4_stratum_weighting_controls.py -q
14 passed

uv run python -m py_compile scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py
PASS

allowlist safety after state update
PASS with zero enabled entries
```

Next allowed action:

```text
Artifact-only micro-overfit route planning for the metric-exact objective patch. Future Slurm/training can proceed only after wrapper/control-plane prerequisites pass.
```
