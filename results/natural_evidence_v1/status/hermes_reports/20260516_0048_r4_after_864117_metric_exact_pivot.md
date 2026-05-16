# R4 after-864117 pivot recorded

Phase:

```text
V2_R4_AFTER_864117_METRIC_EXACT_OBJECTIVE_PIVOT_RECORDED_ARTIFACT_ONLY
```

Blocker:

```text
BLOCK_R4_METRIC_EXACT_OBJECTIVE_ROUTE_PLANNING_NEXT_NO_COMPUTE
```

Summary:

```text
Job 864117 was reviewed as a clean Slurm completion but failed the selective teacher-forced gate.
The scalar additive controller line is now recorded as exhausted for candidate-v3 unless a new controller design is created.
The next selected route is metric-exact objective repair planning, artifact-only.
No Slurm, model scoring, generation, training, Llama, null/FAR, sanitizer, payload diversity, or paper claim was unlocked.
Allowlist safety rechecked: PASS with zero enabled entries.
```

Artifacts:

```text
docs/natural_evidence_v2/R4_AFTER_864117_METRIC_EXACT_OBJECTIVE_PIVOT_20260516.md
configs/natural_evidence_v2/r4_after_864117_pivot_package.yaml
scripts/natural_evidence_v2/validate_r4_after_864117_pivot_package.py
tests/natural_evidence_v2/test_r4_after_864117_pivot_package.py
results/natural_evidence_v2/status/r4_after_864117_pivot_package_validation_20260516/
results/natural_evidence_v2/status/r4_after_864117_pivot_package_validation_20260516_allowlist_safety.json
results/natural_evidence_v2/status/r4_after_864117_pivot_post_state_allowlist_safety_20260516.json
```

Validation:

```text
uv run pytest tests/natural_evidence_v2/test_r4_after_864117_pivot_package.py tests/natural_evidence_v2/test_r4_target_mass_floor_loss.py tests/natural_evidence_v2/test_r4_stratum_weighting_controls.py tests/natural_evidence_v2/test_r4_training_objective_disabled_by_default.py -q
14 passed
```

Next allowed action:

```text
Artifact-only metric-exact objective repair code review and route planning.
Future training/scoring may proceed only after its route prerequisites pass:
objective code review, toy-logit tests, wrapper plan-only validation, local/remote hash preflight, zero-enabled allowlist safety, Hermes TG/email notification, exactly-one-entry submission preflight, and immediate allowlist disablement after sbatch.
```
