# R4 Metric-Exact Objective Patch Review

Status: `PASS_R4_METRIC_EXACT_OBJECTIVE_PATCH_STATIC_REVIEW_NO_TRAINING`

The trainer now has an explicit `--surface-margin-loss-mode logsumexp_softplus` option. The default remains `mass_relu`, so existing routes are unchanged unless a reviewed route opts into the metric-exact mode.

Validated:

```text
uv run pytest tests/natural_evidence_v2/test_r4_metric_exact_objective_helpers.py tests/natural_evidence_v2/test_r4_training_objective_disabled_by_default.py tests/natural_evidence_v2/test_r4_target_mass_floor_loss.py tests/natural_evidence_v2/test_r4_stratum_weighting_controls.py -q
14 passed

uv run python -m py_compile scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py
PASS
```

No Slurm, training, scoring, generation, Llama, FAR, sanitizer, payload diversity, or paper claim was started.
