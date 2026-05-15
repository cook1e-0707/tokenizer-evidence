# R4 controller-only 863274 repair plan recorded

phase:
V2_R4_CONTROLLER_ONLY_SCORE_863274_REVIEWED_FAIL_NO_GENERATION

summary:
```text
After reviewing job 863274, Codex recorded the artifact-only repair route plan.
The next valid work is design/implementation of a new controller repair package, not a rerun of 863274.

Key diagnosis:
- wrong-control contamination is fixed: wrong-key/wrong-payload basic gate passes are 0/72
- positive pressure remains too weak: controlled-base basic gate passes are 0/72
- best controlled lift vs base is +0.0154, far below +0.15
- selected cap probe shows the best controlled-base grid is mostly uncapped, so the failure is not mainly KL/target-mass clipping
```

artifacts:
```text
docs/natural_evidence_v2/R4_CONTROLLER_ONLY_863274_REPAIR_ROUTE_PLAN_20260515.md
results/natural_evidence_v2/status/r4_controller_only_863274_repair_route_plan_20260515/
```

next_allowed_action:
Artifact-only design/implementation of a stronger or more targeted controller repair package. No Slurm/generation/training/Llama/null/sanitizer/FAR/paper-claim action until a new reviewed route records prerequisites and control-plane checks.
