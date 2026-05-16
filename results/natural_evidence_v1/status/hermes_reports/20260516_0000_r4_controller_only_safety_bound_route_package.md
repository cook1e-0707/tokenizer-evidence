# R4 controller-only safety-bound route package ready for remote preflight

phase:
V2_R4_CONTROLLER_ONLY_SAFETY_BOUND_ROUTE_PACKAGE_PLAN_ONLY_PASS_NO_SUBMIT

summary:
```text
Codex implemented the artifact-only follow-up route package after 863274 failed positive pressure.
This is not a Slurm submission and not generation.

Changes:
- wrapper now derives controller grid values from route config via emit_r4_pressure_controller_grid.py
- added safety-bound stronger controller route config
- added disabled allowlist entry v2_r4_controller_only_safety_bound_pressure_score_h200
- local route tests pass
- wrapper plan-only smoke pass for grid_23
- allowlist safety pass with zero enabled entries

Future H200 route, if remote preflight passes:
--array=0-23%4
controller-only teacher-forced scoring only
no generation/training/Llama/null/sanitizer/FAR/paper claim
```

artifacts:
```text
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_SAFETY_BOUND_PRESSURE_ROUTE_20260515.md
configs/natural_evidence_v2/r4_controller_only_safety_bound_pressure_route.yaml
results/natural_evidence_v2/status/r4_controller_only_safety_bound_route_package_20260515/
```

next_allowed_action:
Remote sync/hash preflight for the safety-bound route package. If that passes, exactly one reviewed H200/pomplun teacher-forced scoring job may be submitted and the allowlist entry must be disabled immediately after sbatch returns.
