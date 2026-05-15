# R4 Pressure-Controller Single-Submission Route 20260515 0525

## Decision

Proceed to exactly one Chimera H200/pomplun Slurm array submission for the R4
positive-selectivity pressure-controller teacher-forced scoring route.

This route is scoring-only. It does not start free generation, training, Qwen
E2E, Llama, same-family nulls, sanitizer benchmarks, FAR aggregation, payload
diversity work, or paper-facing positive claims.

## Bound Artifacts

- Route config:
  `configs/natural_evidence_v2/r4_positive_selectivity_pressure_controller_route.yaml`
- Wrapper:
  `scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch`
- Scorer:
  `scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py`
- Controller helper:
  `scripts/natural_evidence_v2/r4_prefix_native_soft_logit_controller.py`
- Candidate rows:
  `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_v3_20260513/r4_prefix_native_surface_probe_rows_v3.jsonl`
- Candidate-row hash:
  `d35e5483ce7f6d3d782ce17961b2c407909afc879a12917c5ccc27090f3c80b7`

## Preflight Evidence

- Local wrong-control mapping and full-wrapper review passed:
  `results/natural_evidence_v2/status/r4_positive_selectivity_pressure_controller_wrong_control_mapping_20260515_0512/`
- Remote wrapper plan-only validation passed:
  `results/natural_evidence_v2/status/r4_pressure_controller_remote_preflight_20260515_0520/`
- Remote zero-enabled allowlist safety passed.
- Local/remote hashes matched.
- Active Chimera job preflight found no active jobs.

## Authorized Submission

Enable exactly one allowlist entry:

```text
v2_r4_positive_selectivity_pressure_controller_score_h200
```

Submit exactly one H200/pomplun Slurm array job:

```text
sbatch --export=ALL,ALLOW_PRESSURE_CONTROLLER_SCORING=1 scripts/natural_evidence_v2/slurm/r4_positive_selectivity_pressure_controller_score_h200.sbatch
```

Then immediately disable the allowlist entry after `sbatch` returns.

## Expected Scope

- Conditions:
  `base`, `task_only`, `controlled_protected`, `wrong_key_controlled`,
  `wrong_payload_controlled`
- Controller grid size: `72`
- Model family: Qwen only
- Contract: same-contract `a55e`
- Primary output:
  `r4_teacher_forced_surface_mass_summary.json`

## Stop Rules

Stop and record a blocker if any of the following occurs:

- More than one allowlist entry is enabled.
- The enabled entry is not
  `v2_r4_positive_selectivity_pressure_controller_score_h200`.
- Local or remote allowlist safety fails.
- Active Chimera jobs appear before submission.
- `sbatch` fails to return a job id.
- The allowlist entry remains enabled after submission.

