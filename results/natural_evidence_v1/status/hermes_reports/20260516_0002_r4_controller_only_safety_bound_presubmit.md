# R4 controller-only safety-bound H200 presubmit

phase:
V2_R4_CONTROLLER_ONLY_SAFETY_BOUND_ROUTE_PACKAGE_PLAN_ONLY_PASS_NO_SUBMIT

summary:
```text
Remote preflight passed for the safety-bound controller route package.
Local/remote hashes match.
Remote wrapper plan-only passed.
Local and remote single-enabled allowlist preflight passed with exactly:
v2_r4_controller_only_safety_bound_pressure_score_h200

Submitting exactly one H200/pomplun teacher-forced scoring array next:
--array=0-23%4
CONTROLLER_CONDITION_SET=controller_only_controls
ROUTE_CONFIG=configs/natural_evidence_v2/r4_controller_only_safety_bound_pressure_route.yaml
No generation/training/Llama/null/sanitizer/FAR/paper claim.
```

post_submit_rule:
Disable the allowlist entry immediately after sbatch returns.
